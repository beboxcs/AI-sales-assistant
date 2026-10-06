from decimal import Decimal
import inspect
import json
from types import SimpleNamespace

import pytest

from benchmarks import run_search_fallback_benchmark as runner


def _config(mode):
    return runner.ManualBenchmarkConfig(
        mode=mode,
        deterministic_revision="12dc8bd",
    )


def test_config_does_not_accept_model_or_pricing_overrides():
    assert set(inspect.signature(runner.ManualBenchmarkConfig).parameters) == {
        "mode",
        "deterministic_revision",
        "repetition_id",
        "caller_supplied_timestamp",
    }


def test_model_profile_derives_conservative_decimal_cost_inputs():
    assert runner.MODEL_PROFILE.max_input_tokens_per_call == 400000 - 2048
    assert isinstance(
        runner.MODEL_PROFILE.input_cost_per_million_tokens_usd,
        Decimal,
    )
    assert isinstance(
        runner.MODEL_PROFILE.output_cost_per_million_tokens_usd,
        Decimal,
    )


def test_dry_run_is_offline_and_reports_auditable_bounds(monkeypatch):
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda **kwargs: pytest.fail("dry-run created an OpenAI client"),
    )

    output = runner.run_manual_benchmark(
        _config(runner.ManualBenchmarkMode.DRY_RUN),
        environ={},
    )
    details = json.loads(output)

    assert details == {
        "dataset_version": "1",
        "eligible_case_count": 15,
        "estimated_worst_case_cost_usd": "4.615200",
        "local_budget_cap_usd": "5.00",
        "context_window_tokens": 400000,
        "max_input_tokens_per_call": 397952,
        "max_output_tokens": 2048,
        "max_provider_calls": 15,
        "max_retries": 0,
        "model_identifier": "gpt-5.4-mini-2026-03-17",
        "pricing_assumptions": {
            "input_cost_per_million_tokens_usd": "0.75",
            "output_cost_per_million_tokens_usd": "4.50",
        },
        "repetitions": 1,
        "timeout_seconds": 45,
        "total_case_count": 54,
    }
    assert "OPENAI_API_KEY" not in output
    assert "New Cairo" not in output


def test_fake_provider_runs_offline_without_api_key(monkeypatch):
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda **kwargs: pytest.fail("fake mode created an OpenAI client"),
    )

    output = runner.run_manual_benchmark(
        _config(runner.ManualBenchmarkMode.FAKE_PROVIDER),
        environ={},
    )
    report = json.loads(output)

    assert report["metadata"]["provider_name"] == "fake"
    assert (
        report["metadata"]["model_identifier"]
        == "gpt-5.4-mini-2026-03-17"
    )
    assert report["overall"]["invocation"] == {
        "denominator": 15,
        "numerator": 15,
    }
    assert "3 bedroom apartment in New Cairo under 6 million cash" not in output
    assert "proposal_json" not in output


def test_missing_api_key_fails_before_client_creation(monkeypatch):
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda **kwargs: pytest.fail("preflight created an OpenAI client"),
    )

    with pytest.raises(
        runner.ManualBenchmarkConfigurationError,
        match="OPENAI_API_KEY is required",
    ):
        runner.run_manual_benchmark(
            _config(runner.ManualBenchmarkMode.REAL_PROVIDER),
            environ={},
        )


def test_conservative_cost_over_cap_fails_before_client_creation(monkeypatch):
    monkeypatch.setattr(runner, "LOCAL_BUDGET_CAP_USD", Decimal("4.00"))
    monkeypatch.setattr(
        runner,
        "OpenAI",
        lambda **kwargs: pytest.fail("preflight created an OpenAI client"),
    )

    with pytest.raises(
        runner.ManualBenchmarkConfigurationError,
        match="exceeds local budget cap",
    ):
        runner.run_manual_benchmark(
            _config(runner.ManualBenchmarkMode.REAL_PROVIDER),
            environ={"OPENAI_API_KEY": "secret"},
        )


def test_real_provider_composes_bounded_client_and_adapter(monkeypatch):
    secret = "sk-secret-never-report"
    captured = {}

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(status="completed", output_parsed=None)

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    def create_client(**kwargs):
        captured["client_kwargs"] = kwargs
        captured["client"] = FakeClient()
        return captured["client"]

    monkeypatch.setattr(runner, "OpenAI", create_client)

    output = runner.run_manual_benchmark(
        _config(runner.ManualBenchmarkMode.REAL_PROVIDER),
        environ={"OPENAI_API_KEY": secret},
    )
    report = json.loads(output)

    assert captured["client_kwargs"] == {
        "api_key": secret,
        "timeout": 45,
        "max_retries": 0,
    }
    assert len(captured["client"].responses.calls) == 15
    assert {
        (call["model"], call["max_output_tokens"], call["store"])
        for call in captured["client"].responses.calls
    } == {("gpt-5.4-mini-2026-03-17", 2048, False)}
    run = report["metadata"]["run_configuration"]
    assert run["timeout_seconds"] == 45
    assert run["max_retries"] == 0
    assert run["context_window_tokens"] == 400000
    assert run["max_input_tokens_per_call"] == 397952
    assert run["max_output_tokens"] == 2048
    assert run["repetitions"] == 1
    assert run["max_provider_calls"] == 15
    assert run["local_budget_cap_usd"] == "5.00"
    assert run["input_cost_per_million_tokens_usd"] == "0.75"
    assert run["output_cost_per_million_tokens_usd"] == "4.50"
    assert run["estimated_worst_case_cost_usd"] == "4.615200"
    assert secret not in output
