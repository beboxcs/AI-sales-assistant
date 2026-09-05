from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.ai_engine.search_filters import SearchFilters


class SearchField(str, Enum):
    LOCATION = "location"
    PROPERTY_TYPE = "property_type"
    BEDROOMS = "bedrooms"
    BATHROOMS = "bathrooms"
    MIN_PRICE = "min_price"
    MAX_PRICE = "max_price"
    MIN_AREA_SQM = "min_area_sqm"
    MAX_AREA_SQM = "max_area_sqm"
    PAYMENT_METHOD = "payment_method"
    MAX_DOWN_PAYMENT = "max_down_payment"
    MIN_INSTALLMENT_YEARS = "min_installment_years"


class ConstraintStrength(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class MatchMode(str, Enum):
    EXACT = "exact"
    NEAR = "near"
    ALTERNATIVE = "alternative"


class RelaxationDirection(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    BOTH = "both"


class SearchConstraint(BaseModel):
    field: SearchField
    strength: ConstraintStrength


class AlternativeDropRule(BaseModel):
    field: SearchField
    priority: int = Field(default=1, gt=0)


class RankingPreference(BaseModel):
    field: SearchField
    weight: float = Field(gt=0, allow_inf_nan=False)


class RelaxationRule(BaseModel):
    field: SearchField
    max_delta: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_percentage: float | None = Field(
        default=None, gt=0, le=100, allow_inf_nan=False
    )
    direction: RelaxationDirection
    priority: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def require_relaxation_limit(self) -> "RelaxationRule":
        if self.max_delta is None and self.max_percentage is None:
            raise ValueError(
                "RelaxationRule requires max_delta or max_percentage"
            )
        return self


NUMERIC_RELAXABLE_FIELDS: frozenset[SearchField] = frozenset(
    {
        SearchField.BEDROOMS,
        SearchField.BATHROOMS,
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
        SearchField.MIN_AREA_SQM,
        SearchField.MAX_AREA_SQM,
        SearchField.MAX_DOWN_PAYMENT,
        SearchField.MIN_INSTALLMENT_YEARS,
    }
)

DELTA_RELAXABLE_FIELDS: frozenset[SearchField] = NUMERIC_RELAXABLE_FIELDS

PERCENTAGE_RELAXABLE_FIELDS: frozenset[SearchField] = frozenset(
    {
        SearchField.MIN_PRICE,
        SearchField.MAX_PRICE,
        SearchField.MIN_AREA_SQM,
        SearchField.MAX_AREA_SQM,
        SearchField.MAX_DOWN_PAYMENT,
    }
)

DISCRETE_DELTA_FIELDS: frozenset[SearchField] = frozenset(
    {SearchField.BEDROOMS, SearchField.BATHROOMS}
)


class SearchStrategy(BaseModel):
    filters: SearchFilters
    constraints: list[SearchConstraint] = Field(default_factory=list)
    match_mode: MatchMode = MatchMode.EXACT
    ranking_preferences: list[RankingPreference] = Field(default_factory=list)
    relaxation_rules: list[RelaxationRule] = Field(default_factory=list)
    alternative_drop_rules: list[AlternativeDropRule] = Field(default_factory=list)
    requested_scope_keys: list[str] | None = None

    @field_validator("requested_scope_keys")
    @classmethod
    def normalize_scope_keys(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("requested_scope_keys must contain at least one key")

        normalized = []
        for scope_key in value:
            normalized_key = scope_key.strip()
            if not normalized_key:
                raise ValueError("requested_scope_keys must not contain empty keys")
            normalized.append(normalized_key)

        if len(normalized) != len(set(normalized)):
            raise ValueError("requested_scope_keys must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def validate_references(self) -> "SearchStrategy":
        populated_fields = {
            field.value
            for field in SearchField
            if getattr(self.filters, field.value) is not None
        }

        constraint_fields = [constraint.field for constraint in self.constraints]
        if len(constraint_fields) != len(set(constraint_fields)):
            raise ValueError("constraint fields must be unique")
        for constraint in self.constraints:
            if constraint.field.value not in populated_fields:
                raise ValueError(
                    "constraint field must reference a populated filter: "
                    f"{constraint.field.value}"
                )

        ranking_fields = [preference.field for preference in self.ranking_preferences]
        if len(ranking_fields) != len(set(ranking_fields)):
            raise ValueError("ranking preference fields must be unique")
        for preference in self.ranking_preferences:
            if preference.field.value not in populated_fields:
                raise ValueError(
                    "ranking preference field must reference a populated filter: "
                    f"{preference.field.value}"
                )

        constraint_strengths = {
            constraint.field: constraint.strength for constraint in self.constraints
        }
        relaxation_fields = [rule.field for rule in self.relaxation_rules]
        if len(relaxation_fields) != len(set(relaxation_fields)):
            raise ValueError("relaxation rule fields must be unique")

        for rule in self.relaxation_rules:
            if rule.field not in NUMERIC_RELAXABLE_FIELDS:
                raise ValueError(
                    "relaxation rules only support numeric fields: "
                    f"{rule.field.value}"
                )
            if rule.field.value not in populated_fields:
                raise ValueError(
                    "relaxation rule field must reference a populated filter: "
                    f"{rule.field.value}"
                )
            if (
                rule.max_delta is not None
                and rule.field not in DELTA_RELAXABLE_FIELDS
            ):
                raise ValueError(
                    "delta relaxation is not supported for field: "
                    f"{rule.field.value}"
                )
            if (
                rule.max_percentage is not None
                and rule.field not in PERCENTAGE_RELAXABLE_FIELDS
            ):
                raise ValueError(
                    "percentage relaxation is not supported for field: "
                    f"{rule.field.value}"
                )
            if (
                rule.max_delta is not None
                and rule.field in DISCRETE_DELTA_FIELDS
                and not rule.max_delta.is_integer()
            ):
                raise ValueError(
                    "max_delta must be a positive whole number for field: "
                    f"{rule.field.value}"
                )
            strength = constraint_strengths.get(rule.field)
            if strength is None:
                raise ValueError(
                    "relaxation rule field must reference a classified constraint: "
                    f"{rule.field.value}"
                )
            if strength is not ConstraintStrength.SOFT:
                raise ValueError(
                    "relaxation rules are only allowed for soft constraints: "
                    f"{rule.field.value}"
                )

        drop_fields = [rule.field for rule in self.alternative_drop_rules]
        if len(drop_fields) != len(set(drop_fields)):
            raise ValueError("alternative drop-rule fields must be unique")
        for rule in self.alternative_drop_rules:
            if rule.field.value not in populated_fields:
                raise ValueError(
                    "alternative drop-rule field must reference a populated filter: "
                    f"{rule.field.value}"
                )
            strength = constraint_strengths.get(rule.field)
            if strength is None:
                raise ValueError(
                    "alternative drop-rule field must reference a classified "
                    f"constraint: {rule.field.value}"
                )
            if strength is not ConstraintStrength.SOFT:
                raise ValueError(
                    "alternative drop rules are only allowed for soft constraints: "
                    f"{rule.field.value}"
                )
        return self
