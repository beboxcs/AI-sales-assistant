from datetime import date
from decimal import Decimal
from json import JSONDecodeError, loads
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
)

from app.ai_engine import search_application
from app.ai_engine.search_filters import SearchFilters
from app.api.dependencies import (
    SearchContextResolver,
    get_search_context_resolver,
    get_search_company_id,
)
from app.api.errors import APIError
from app.api.security import get_current_user
from app.auth.human.types import AuthenticatedUserContext
from app.domain.enums import PaymentMethod
from app.search.exceptions import (
    InvalidSearchScopeError,
    SearchScopeResolutionFailedError,
)

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str | None = None
    property_type: str | None = None
    bedrooms: int | None = Field(default=None, ge=0)
    bathrooms: int | None = Field(default=None, ge=0)
    min_price: Decimal | None = Field(default=None, ge=0)
    max_price: Decimal | None = Field(default=None, ge=0)
    min_area_sqm: Decimal | None = Field(default=None, gt=0)
    max_area_sqm: Decimal | None = Field(default=None, gt=0)
    payment_method: PaymentMethod | None = None
    max_down_payment: Decimal | None = Field(default=None, ge=0)
    min_installment_years: Decimal | None = Field(default=None, gt=0)
    requested_scope_keys: list[str] | None = Field(default=None, min_length=1)

    @field_validator("requested_scope_keys")
    @classmethod
    def requested_scope_keys_must_be_unique(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        if value is None:
            return None
        if len(set(value)) != len(value):
            raise ValueError("requested_scope_keys must not contain duplicates")
        return value

    def to_filters(self) -> SearchFilters:
        return SearchFilters(
            location=self.location,
            property_type=self.property_type,
            bedrooms=self.bedrooms,
            bathrooms=self.bathrooms,
            min_price=self.min_price,
            max_price=self.max_price,
            min_area_sqm=self.min_area_sqm,
            max_area_sqm=self.max_area_sqm,
            payment_method=(
                self.payment_method.value
                if self.payment_method is not None
                else None
            ),
            max_down_payment=self.max_down_payment,
            min_installment_years=self.min_installment_years,
        )

    def requested_scope_tuple(self) -> tuple[str, ...] | None:
        if self.requested_scope_keys is None:
            return None
        return tuple(self.requested_scope_keys)


class PaymentPlanResponse(BaseModel):
    plan_key: str
    source_plan_id: str | None
    plan_name: str | None
    payment_method: str
    down_payment: Decimal | None
    installment_years: Decimal | None
    total_price: Decimal | None
    currency: str


class UnitResponse(BaseModel):
    unit_id: int
    external_unit_id: str
    compound_id: int
    compound_name: str
    location: str
    property_type: str
    bedrooms: int | None
    bathrooms: int | None
    area_sqm: Decimal | None
    base_price: Decimal | None
    base_currency: str
    status: str
    delivery_date: date | None
    source_last_updated: date | None
    matched_scope_keys: list[str]
    payment_plans: list[PaymentPlanResponse]


class RankingResponse(BaseModel):
    rank: int
    score: Decimal
    best_plan_key: str | None


class ExplanationFactResponse(BaseModel):
    dimension: str
    kind: str
    message_key: str
    baseline_value: Any | None
    candidate_value: Any | None
    score: Decimal | None


class ExplanationResponse(BaseModel):
    strengths: list[ExplanationFactResponse]
    tradeoffs: list[ExplanationFactResponse]


class SearchResultResponse(BaseModel):
    unit: UnitResponse
    ranking: RankingResponse
    explanation: ExplanationResponse


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]


class NaturalLanguageSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: StrictStr = Field(max_length=500)


class NaturalLanguageSearchFiltersResponse(BaseModel):
    location: str | None
    property_type: str | None
    bedrooms: int | None
    bathrooms: int | None
    min_price: Decimal | None
    max_price: Decimal | None
    min_area_sqm: Decimal | None
    max_area_sqm: Decimal | None
    payment_method: str | None
    max_down_payment: Decimal | None
    min_installment_years: Decimal | None


class NaturalLanguageSearchDirectiveResponse(BaseModel):
    field: str
    strength: str | None
    preferred: bool
    allow_alternative: bool


class NaturalLanguageSearchAmbiguityResponse(BaseModel):
    field: str | None
    reason: str


class NaturalLanguageSearchIntentResponse(BaseModel):
    filters: NaturalLanguageSearchFiltersResponse
    directives: list[NaturalLanguageSearchDirectiveResponse]
    ambiguities: list[NaturalLanguageSearchAmbiguityResponse]
    unsupported_requirements: list[str]


class NaturalLanguageSearchResponse(BaseModel):
    intent: NaturalLanguageSearchIntentResponse
    executed: bool
    results: list[SearchResultResponse] | None


_SEARCH_REQUEST_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": SearchRequest.model_json_schema(),
            }
        },
    }
}


async def _parse_search_request(
    raw_request: Request,
) -> SearchRequest:
    if not _is_json_content_type(raw_request.headers.get("content-type")):
        raise RequestValidationError(
            [
                {
                    "type": "model_attributes_type",
                    "loc": ("body",),
                    "msg": "Input should be a valid dictionary or object to extract fields from",
                    "input": None,
                }
            ]
        )

    raw_body = await raw_request.body()
    try:
        request_payload = loads(raw_body, parse_float=Decimal)
    except (JSONDecodeError, UnicodeDecodeError) as exc:
        raise RequestValidationError(
            [
                {
                    "type": "json_invalid",
                    "loc": ("body",),
                    "msg": "Invalid JSON",
                    "input": None,
                }
            ]
        ) from exc

    try:
        return SearchRequest.model_validate(request_payload)
    except ValidationError as exc:
        raise RequestValidationError(_body_validation_errors(exc)) from exc


def _is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _body_validation_errors(exc: ValidationError) -> list[dict[str, Any]]:
    errors = []
    for validation_error in exc.errors():
        error_detail = dict(validation_error)
        error_detail["loc"] = ("body", *validation_error["loc"])
        errors.append(error_detail)
    return errors


@router.post(
    "/search",
    response_model=SearchResponse,
    openapi_extra=_SEARCH_REQUEST_OPENAPI,
)
def search_units(
    request: SearchRequest = Depends(_parse_search_request),
    company_id: int = Depends(get_search_company_id),
    resolve_context: SearchContextResolver = Depends(get_search_context_resolver),
) -> SearchResponse:
    filters = request.to_filters()

    try:
        context = resolve_context(
            company_id=company_id,
            requested_scope_keys=request.requested_scope_tuple(),
        )
    except InvalidSearchScopeError as exc:
        raise APIError(
            status_code=404,
            code="SEARCH_SCOPE_NOT_FOUND",
            message="Search scope selection was not found",
        ) from exc
    except SearchScopeResolutionFailedError as exc:
        raise APIError(
            status_code=500,
            code="SEARCH_CONTEXT_RESOLUTION_FAILED",
            message="Search context could not be resolved",
        ) from exc
    except APIError:
        raise
    except Exception as exc:
        raise APIError(
            status_code=500,
            code="SEARCH_CONTEXT_RESOLUTION_FAILED",
            message="Search context could not be resolved",
        ) from exc

    try:
        result = search_application.search_units(filters=filters, context=context)
    except APIError:
        raise
    except Exception as exc:
        raise APIError(
            status_code=500,
            code="SEARCH_FAILED",
            message="Search could not be completed",
        ) from exc

    return SearchResponse(
        results=[_project_search_result(item) for item in result.results]
    )


@router.post(
    "/search/natural-language",
    response_model=NaturalLanguageSearchResponse,
)
def natural_language_search_units(
    request: NaturalLanguageSearchRequest,
    authenticated_user: AuthenticatedUserContext = Depends(get_current_user),
    resolve_context: SearchContextResolver = Depends(get_search_context_resolver),
) -> NaturalLanguageSearchResponse:
    try:
        context = resolve_context(
            company_id=authenticated_user.company_id,
            requested_scope_keys=None,
        )
    except InvalidSearchScopeError as exc:
        raise APIError(
            status_code=404,
            code="SEARCH_SCOPE_NOT_FOUND",
            message="Search scope selection was not found",
        ) from exc
    except SearchScopeResolutionFailedError as exc:
        raise APIError(
            status_code=500,
            code="SEARCH_CONTEXT_RESOLUTION_FAILED",
            message="Search context could not be resolved",
        ) from exc
    except APIError:
        raise
    except Exception as exc:
        raise APIError(
            status_code=500,
            code="SEARCH_CONTEXT_RESOLUTION_FAILED",
            message="Search context could not be resolved",
        ) from exc

    try:
        result = search_application.search_units_from_query(
            text=request.query,
            context=context,
        )
    except APIError:
        raise
    except Exception as exc:
        raise APIError(
            status_code=500,
            code="SEARCH_FAILED",
            message="Search could not be completed",
        ) from exc

    return NaturalLanguageSearchResponse(
        intent=_project_natural_language_intent(result.intent),
        executed=result.search is not None,
        results=(
            None
            if result.search is None
            else [_project_search_result(item) for item in result.search.results]
        ),
    )


def _project_natural_language_intent(intent) -> NaturalLanguageSearchIntentResponse:
    filters = intent.filters
    return NaturalLanguageSearchIntentResponse(
        filters=NaturalLanguageSearchFiltersResponse(
            location=filters.location,
            property_type=filters.property_type,
            bedrooms=filters.bedrooms,
            bathrooms=filters.bathrooms,
            min_price=filters.min_price,
            max_price=filters.max_price,
            min_area_sqm=filters.min_area_sqm,
            max_area_sqm=filters.max_area_sqm,
            payment_method=filters.payment_method,
            max_down_payment=filters.max_down_payment,
            min_installment_years=filters.min_installment_years,
        ),
        directives=[
            NaturalLanguageSearchDirectiveResponse(
                field=_public_value(directive.field),
                strength=_public_value(directive.strength),
                preferred=directive.preferred,
                allow_alternative=directive.allow_alternative,
            )
            for directive in intent.directives
        ],
        ambiguities=[
            NaturalLanguageSearchAmbiguityResponse(
                field=_public_value(ambiguity.field),
                reason=ambiguity.reason,
            )
            for ambiguity in intent.ambiguities
        ],
        unsupported_requirements=list(intent.unsupported_requirements),
    )


def _project_search_result(item) -> SearchResultResponse:
    candidate = item.candidate
    return SearchResultResponse(
        unit=UnitResponse(
            unit_id=candidate.unit_id,
            external_unit_id=candidate.external_unit_id,
            compound_id=candidate.compound_id,
            compound_name=candidate.compound_name,
            location=candidate.location,
            property_type=candidate.property_type,
            bedrooms=candidate.bedrooms,
            bathrooms=candidate.bathrooms,
            area_sqm=candidate.area_sqm,
            base_price=candidate.base_price,
            base_currency=_public_value(candidate.base_currency),
            status=_public_value(candidate.status),
            delivery_date=candidate.delivery_date,
            source_last_updated=candidate.source_last_updated,
            matched_scope_keys=list(candidate.matched_scope_keys),
            payment_plans=[
                PaymentPlanResponse(
                    plan_key=plan.plan_key,
                    source_plan_id=plan.source_plan_id,
                    plan_name=plan.plan_name,
                    payment_method=_public_value(plan.payment_method),
                    down_payment=plan.down_payment,
                    installment_years=plan.installment_years,
                    total_price=plan.total_price,
                    currency=_public_value(plan.currency),
                )
                for plan in candidate.payment_plans
            ],
        ),
        ranking=RankingResponse(
            rank=item.ranking.rank,
            score=item.ranking.score,
            best_plan_key=item.ranking.best_plan_key,
        ),
        explanation=ExplanationResponse(
            strengths=[
                _project_explanation_fact(fact)
                for fact in item.explanation.strengths
            ],
            tradeoffs=[
                _project_explanation_fact(fact)
                for fact in item.explanation.tradeoffs
            ],
        ),
    )


def _project_explanation_fact(fact) -> ExplanationFactResponse:
    return ExplanationFactResponse(
        dimension=_public_value(fact.dimension),
        kind=_public_value(fact.kind),
        message_key=fact.message_key,
        baseline_value=_project_explanation_baseline_value(fact.baseline_value),
        candidate_value=fact.candidate_value,
        score=fact.score,
    )


def _project_explanation_baseline_value(baseline_value):
    if isinstance(baseline_value, Decimal):
        return float(baseline_value)
    if isinstance(baseline_value, tuple):
        return tuple(
            _project_explanation_baseline_value(item_baseline)
            for item_baseline in baseline_value
        )
    if isinstance(baseline_value, list):
        return [
            _project_explanation_baseline_value(item_baseline)
            for item_baseline in baseline_value
        ]
    return _public_value(baseline_value)


def _public_value(value):
    return getattr(value, "value", value)
