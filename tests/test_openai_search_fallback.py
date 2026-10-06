import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.ai_engine.providers.openai_search_fallback import (
    OpenAISearchFallbackProvider,
)
from app.ai_engine.search_fallback import (
    EvidenceSpan,
    ProposedSearchFilters,
    ProposedTextValue,
    UntrustedSearchProposal,
)
from app.ai_engine.search_fallback_shadow import (
    SearchFallbackProviderError,
    SearchFallbackProviderRequest,
    SearchFallbackProviderResponse,
    SearchFallbackProviderTimeoutError,
)


class FakeResponses:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, responses, secret="unused-test-secret"):
        self.responses = responses
        self.secret = secret


def _proposal(text="New Cairo"):
    return UntrustedSearchProposal(
        filters=ProposedSearchFilters(
            location=ProposedTextValue(
                value="New Cairo",
                evidence=EvidenceSpan(start=0, end=len(text)),
            )
        )
    )


def _provider(
    *,
    response=None,
    error=None,
    model="gpt-6-luna",
    max_output_tokens=2048,
):
    responses = FakeResponses(response=response, error=error)
    return OpenAISearchFallbackProvider(
        client=FakeClient(responses),
        model=model,
        max_output_tokens=max_output_tokens,
    ), responses


def test_provider_sends_only_minimal_structured_request():
    text = "New Cairo for company 7"
    provider, responses = _provider(
        response=SimpleNamespace(status="completed", output_parsed=_proposal(text)),
        max_output_tokens=1536,
    )

    provider.propose_search(request=SearchFallbackProviderRequest(text=text))

    assert len(responses.calls) == 1
    call = responses.calls[0]
    assert call["model"] == "gpt-6-luna"
    assert call["text_format"] is UntrustedSearchProposal
    assert call["store"] is False
    assert call["max_output_tokens"] == 1536
    assert call["input"] == text
    assert set(call) == {
        "model",
        "instructions",
        "input",
        "text_format",
        "store",
        "max_output_tokens",
    }
    assert call["instructions"].strip()
    assert text not in call["instructions"]


def test_valid_parsed_proposal_returns_canonical_provider_response():
    proposal = _proposal()
    provider, _ = _provider(
        response=SimpleNamespace(status="completed", output_parsed=proposal)
    )

    first = provider.propose_search(
        request=SearchFallbackProviderRequest(text="New Cairo")
    )
    second = provider.propose_search(
        request=SearchFallbackProviderRequest(text="New Cairo")
    )

    assert first == second == SearchFallbackProviderResponse(
        proposal_json=proposal.model_dump_json()
    )


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status="completed", output_parsed=None),
        SimpleNamespace(status="completed", output_parsed=None, refusal="refused"),
        SimpleNamespace(status="incomplete", output_parsed=_proposal()),
    ],
)
def test_provider_returns_none_without_a_complete_parsed_proposal(response):
    provider, _ = _provider(response=response)

    assert provider.propose_search(
        request=SearchFallbackProviderRequest(text="query")
    ) is None


def test_timeout_is_mapped_without_provider_details():
    secret = "private-timeout-detail"
    error = openai.APITimeoutError(request=httpx.Request("POST", "https://test"))
    error.args = (secret,)
    provider, _ = _provider(error=error)

    with pytest.raises(SearchFallbackProviderTimeoutError) as raised:
        provider.propose_search(request=SearchFallbackProviderRequest(text="query"))

    assert str(raised.value) == "search fallback provider timed out"
    assert secret not in repr(raised.value)


def test_api_error_is_mapped_without_provider_details():
    secret = "private-api-detail"
    error = openai.APIError(
        secret,
        httpx.Request("POST", "https://test"),
        body={"secret": secret},
    )
    provider, _ = _provider(error=error)

    with pytest.raises(SearchFallbackProviderError) as raised:
        provider.propose_search(request=SearchFallbackProviderRequest(text="query"))

    assert type(raised.value) is SearchFallbackProviderError
    assert str(raised.value) == "search fallback provider failed"
    assert secret not in repr(raised.value)


def test_unexpected_programming_error_propagates():
    provider, _ = _provider(error=RuntimeError("programming defect"))

    with pytest.raises(RuntimeError, match="programming defect"):
        provider.propose_search(request=SearchFallbackProviderRequest(text="query"))


def test_adapter_repr_does_not_expose_injected_client_secret():
    secret = "sk-test-never-expose"
    adapter = OpenAISearchFallbackProvider(
        client=FakeClient(FakeResponses(), secret=secret),
        model="gpt-6-luna",
        max_output_tokens=2048,
    )

    assert secret not in repr(adapter)


def test_sdk_public_parse_surface_generates_strict_proposal_schema():
    captured = {}

    def reject_after_capture(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "message": "intentional test rejection",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(reject_after_capture)),
    )

    with pytest.raises(openai.BadRequestError):
        client.responses.parse(
            model="schema-test-model",
            input="query",
            text_format=UntrustedSearchProposal,
            store=False,
        )

    schema = captured["text"]["format"]["schema"]
    definitions = schema["$defs"]
    assert captured["text"]["format"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"filters", "directives"}
    assert definitions["ProposedSearchFilters"]["additionalProperties"] is False
    assert definitions["ProposedSearchFilters"]["required"] == [
        "location",
        "property_type",
        "bedrooms",
        "bathrooms",
        "min_price",
        "max_price",
        "min_area_sqm",
        "max_area_sqm",
        "payment_method",
        "max_down_payment",
        "min_installment_years",
    ]
    assert schema["properties"]["directives"]["type"] == "array"
    assert definitions["ProposedDecimalValue"]["properties"]["value"]
    assert definitions["SearchField"]["enum"]
    assert definitions["ConstraintStrength"]["enum"] == ["hard", "soft"]


def test_adapter_has_no_execution_or_trusted_boundary_imports():
    path = Path("app/ai_engine/providers/openai_search_fallback.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not imported_modules & {
        "app.api",
        "app.auth",
        "app.search.repository",
        "app.search.service",
        "app.ai_engine.search_application",
        "app.ai_engine.search_orchestrator",
        "app.ai_engine.search_plan_builder",
        "app.ai_engine.search_strategy",
    }
