import argparse
import json
import os
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Mapping

import openai
from openai import OpenAI

from app.ai_engine.multilingual_search_interpreter import (
    interpret_multilingual_search_query,
)
from app.ai_engine.providers.openai_search_fallback import (
    OpenAISearchFallbackProvider,
)
from app.ai_engine.search_fallback import FallbackDisposition, decide_search_fallback
from app.ai_engine.search_fallback_benchmark import (
    SearchFallbackBenchmarkCase,
    SearchFallbackBenchmarkMetadata,
    SearchFallbackBenchmarkRunConfiguration,
    run_search_fallback_benchmark,
    serialize_search_fallback_benchmark_report,
)
from app.ai_engine.search_fallback_shadow import SearchFallbackProviderRequest
from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_intent import ParsedSearchIntent
from benchmarks.nl_search_golden_cases import (
    NL_SEARCH_GOLDEN_DATASET_VERSION,
    project_golden_benchmark_cases,
)


TIMEOUT_SECONDS = 45
MAX_RETRIES = 0
REPETITIONS = 1
LOCAL_BUDGET_CAP_USD = Decimal("5.00")


class ManualBenchmarkMode(str, Enum):
    DRY_RUN = "dry-run"
    FAKE_PROVIDER = "fake-provider"
    REAL_PROVIDER = "real-provider"


@dataclass(frozen=True)
class _BenchmarkModelProfile:
    model_identifier: str
    context_window_tokens: int
    max_output_tokens: int
    input_cost_per_million_tokens_usd: Decimal
    output_cost_per_million_tokens_usd: Decimal

    @property
    def max_input_tokens_per_call(self) -> int:
        return self.context_window_tokens - self.max_output_tokens


MODEL_PROFILE = _BenchmarkModelProfile(
    model_identifier="gpt-5.4-mini-2026-03-17",
    context_window_tokens=400000,
    max_output_tokens=2048,
    input_cost_per_million_tokens_usd=Decimal("0.75"),
    output_cost_per_million_tokens_usd=Decimal("4.50"),
)


@dataclass(frozen=True)
class ManualBenchmarkConfig:
    mode: ManualBenchmarkMode
    deterministic_revision: str
    repetition_id: str = "1"
    caller_supplied_timestamp: str | None = None


@dataclass(frozen=True)
class _Preflight:
    total_case_count: int
    eligible_case_count: int
    max_provider_calls: int
    estimated_worst_case_cost_usd: Decimal


class ManualBenchmarkConfigurationError(ValueError):
    """Raised before provider creation when benchmark configuration is unsafe."""


class _OfflineNoProposalProvider:
    def propose_search(
        self,
        *,
        request: SearchFallbackProviderRequest,
    ) -> None:
        return None


def run_manual_benchmark(
    config: ManualBenchmarkConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    cases = project_golden_benchmark_cases()
    preflight = _build_preflight(cases)
    if config.mode is ManualBenchmarkMode.DRY_RUN:
        return _serialize_dry_run(preflight)

    metadata = _metadata(config, preflight)
    if config.mode is ManualBenchmarkMode.FAKE_PROVIDER:
        provider = _OfflineNoProposalProvider()
    else:
        provider = _real_provider(preflight, environ)

    report = run_search_fallback_benchmark(
        cases=cases,
        provider=provider,
        metadata=metadata,
    )
    return serialize_search_fallback_benchmark_report(report)


def _real_provider(
    preflight: _Preflight,
    environ: Mapping[str, str] | None,
) -> OpenAISearchFallbackProvider:
    environment = os.environ if environ is None else environ
    api_key = environment.get("OPENAI_API_KEY")
    if not api_key:
        raise ManualBenchmarkConfigurationError("OPENAI_API_KEY is required")
    if preflight.estimated_worst_case_cost_usd > LOCAL_BUDGET_CAP_USD:
        raise ManualBenchmarkConfigurationError(
            "estimated worst-case cost exceeds local budget cap"
        )
    client = OpenAI(
        api_key=api_key,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )
    return OpenAISearchFallbackProvider(
        client=client,
        model=MODEL_PROFILE.model_identifier,
        max_output_tokens=MODEL_PROFILE.max_output_tokens,
    )


def _build_preflight(
    cases: tuple[SearchFallbackBenchmarkCase, ...],
) -> _Preflight:
    eligible_case_count = sum(_is_provider_eligible(case) for case in cases)
    max_provider_calls = eligible_case_count * REPETITIONS
    cost_per_call = (
        MODEL_PROFILE.max_input_tokens_per_call
        * MODEL_PROFILE.input_cost_per_million_tokens_usd
        + MODEL_PROFILE.max_output_tokens
        * MODEL_PROFILE.output_cost_per_million_tokens_usd
    ) / Decimal("1000000")
    return _Preflight(
        total_case_count=len(cases),
        eligible_case_count=eligible_case_count,
        max_provider_calls=max_provider_calls,
        estimated_worst_case_cost_usd=cost_per_call * max_provider_calls,
    )


def _is_provider_eligible(case: SearchFallbackBenchmarkCase) -> bool:
    intent = (
        ParsedSearchIntent(filters=SearchFilters())
        if not case.query.strip()
        else interpret_multilingual_search_query(case.query)
    )
    decision = decide_search_fallback(intent, case.query)
    return decision.disposition in {
        FallbackDisposition.SHADOW_ONLY,
        FallbackDisposition.PROPOSAL_ELIGIBLE,
    }


def _metadata(
    config: ManualBenchmarkConfig,
    preflight: _Preflight,
) -> SearchFallbackBenchmarkMetadata:
    return SearchFallbackBenchmarkMetadata(
        report_schema_version="1",
        dataset_version=NL_SEARCH_GOLDEN_DATASET_VERSION,
        provider_name=(
            "fake"
            if config.mode is ManualBenchmarkMode.FAKE_PROVIDER
            else "openai"
        ),
        model_identifier=MODEL_PROFILE.model_identifier,
        sdk_version=openai.__version__,
        deterministic_revision=config.deterministic_revision,
        sampling_configuration="provider-default",
        repetition_id=config.repetition_id,
        caller_supplied_timestamp=config.caller_supplied_timestamp,
        run_configuration=_run_configuration(preflight),
    )


def _run_configuration(
    preflight: _Preflight,
) -> SearchFallbackBenchmarkRunConfiguration:
    return SearchFallbackBenchmarkRunConfiguration(
        timeout_seconds=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        context_window_tokens=MODEL_PROFILE.context_window_tokens,
        max_input_tokens_per_call=MODEL_PROFILE.max_input_tokens_per_call,
        max_output_tokens=MODEL_PROFILE.max_output_tokens,
        repetitions=REPETITIONS,
        max_provider_calls=preflight.max_provider_calls,
        local_budget_cap_usd=str(LOCAL_BUDGET_CAP_USD),
        input_cost_per_million_tokens_usd=str(
            MODEL_PROFILE.input_cost_per_million_tokens_usd
        ),
        output_cost_per_million_tokens_usd=str(
            MODEL_PROFILE.output_cost_per_million_tokens_usd
        ),
        estimated_worst_case_cost_usd=_money(
            preflight.estimated_worst_case_cost_usd
        ),
    )


def _serialize_dry_run(
    preflight: _Preflight,
) -> str:
    return json.dumps(
        {
            "dataset_version": NL_SEARCH_GOLDEN_DATASET_VERSION,
            "total_case_count": preflight.total_case_count,
            "eligible_case_count": preflight.eligible_case_count,
            "repetitions": REPETITIONS,
            "max_provider_calls": preflight.max_provider_calls,
            "model_identifier": MODEL_PROFILE.model_identifier,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "context_window_tokens": MODEL_PROFILE.context_window_tokens,
            "max_input_tokens_per_call": (
                MODEL_PROFILE.max_input_tokens_per_call
            ),
            "max_output_tokens": MODEL_PROFILE.max_output_tokens,
            "local_budget_cap_usd": str(LOCAL_BUDGET_CAP_USD),
            "pricing_assumptions": {
                "input_cost_per_million_tokens_usd": str(
                    MODEL_PROFILE.input_cost_per_million_tokens_usd
                ),
                "output_cost_per_million_tokens_usd": str(
                    MODEL_PROFILE.output_cost_per_million_tokens_usd
                ),
            },
            "estimated_worst_case_cost_usd": _money(
                preflight.estimated_worst_case_cost_usd
            ),
        },
        indent=2,
        sort_keys=True,
    )


def _money(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001")), "f")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the offline or manual search fallback benchmark.",
    )
    parser.add_argument("mode", choices=[mode.value for mode in ManualBenchmarkMode])
    parser.add_argument("--deterministic-revision", required=True)
    parser.add_argument("--timestamp")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = run_manual_benchmark(
            ManualBenchmarkConfig(
                mode=ManualBenchmarkMode(args.mode),
                deterministic_revision=args.deterministic_revision,
                caller_supplied_timestamp=args.timestamp,
            )
        )
    except ManualBenchmarkConfigurationError as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
