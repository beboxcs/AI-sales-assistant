from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TypeAlias

from pydantic import BaseModel, Field, field_validator, model_validator

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_strategy import MatchMode, SearchField


class AdjustmentAction(str, Enum):
    WIDENED = "widened"
    DROPPED = "dropped"


ContractValue: TypeAlias = str | bool | int | Decimal | None


class FieldAdjustment(BaseModel):
    field: SearchField
    action: AdjustmentAction
    original_value: ContractValue
    adjusted_value: ContractValue

    @field_validator("original_value", "adjusted_value", mode="before")
    @classmethod
    def normalize_numeric_value(cls, raw_contract_value):
        if raw_contract_value is None or isinstance(
            raw_contract_value, (Decimal, bool)
        ):
            return raw_contract_value
        if isinstance(raw_contract_value, float):
            try:
                return Decimal(str(raw_contract_value))
            except (InvalidOperation, ValueError):
                return raw_contract_value
        return raw_contract_value

    @field_validator("original_value", "adjusted_value")
    @classmethod
    def numeric_values_must_be_finite(cls, contract_value):
        if isinstance(contract_value, Decimal) and not contract_value.is_finite():
            raise ValueError("numeric adjustment values must be finite")
        return contract_value

    @model_validator(mode="after")
    def validate_values(self) -> "FieldAdjustment":
        if self.original_value is None:
            raise ValueError("original_value must not be None")
        if self.action is AdjustmentAction.WIDENED and self.adjusted_value is None:
            raise ValueError("widened adjustments require adjusted_value")
        if (
            self.action is AdjustmentAction.WIDENED
            and self.adjusted_value == self.original_value
        ):
            raise ValueError("widened adjustments must change the value")
        if self.action is AdjustmentAction.DROPPED and self.adjusted_value is not None:
            raise ValueError("dropped adjustments must have adjusted_value=None")
        return self


class SearchAttempt(BaseModel):
    sequence: int = Field(gt=0)
    mode: MatchMode
    filters: SearchFilters
    adjustments: list[FieldAdjustment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_adjustments(self) -> "SearchAttempt":
        adjustment_fields = [adjustment.field for adjustment in self.adjustments]
        if len(adjustment_fields) != len(set(adjustment_fields)):
            raise ValueError("adjustment fields must be unique")

        if self.mode is MatchMode.EXACT and self.adjustments:
            raise ValueError("EXACT attempts must not contain adjustments")
        if self.mode in {MatchMode.NEAR, MatchMode.ALTERNATIVE} and not self.adjustments:
            raise ValueError(f"{self.mode.value.upper()} attempts require adjustments")

        for adjustment in self.adjustments:
            if (
                self.mode is MatchMode.NEAR
                and adjustment.action is AdjustmentAction.DROPPED
            ):
                raise ValueError("NEAR attempts must not contain dropped adjustments")

            filter_value = getattr(self.filters, adjustment.field.value)
            if (
                adjustment.action is AdjustmentAction.WIDENED
                and filter_value is None
            ):
                raise ValueError(
                    "widened adjustment field must remain populated in filters: "
                    f"{adjustment.field.value}"
                )
            if (
                adjustment.action is AdjustmentAction.WIDENED
                and filter_value != adjustment.adjusted_value
            ):
                raise ValueError(
                    "widened adjustment value must match the attempt filter: "
                    f"{adjustment.field.value}"
                )
            if (
                adjustment.action is AdjustmentAction.DROPPED
                and filter_value is not None
            ):
                raise ValueError(
                    "dropped adjustment field must be None in filters: "
                    f"{adjustment.field.value}"
                )
        return self


class SearchPlan(BaseModel):
    attempts: list[SearchAttempt]

    @model_validator(mode="after")
    def validate_attempts(self) -> "SearchPlan":
        if not self.attempts:
            raise ValueError("SearchPlan must contain at least one attempt")

        filter_states = []
        for attempt in self.attempts:
            state = tuple(
                getattr(attempt.filters, field.value) for field in SearchField
            )
            for previous_sequence, previous_state in filter_states:
                if state == previous_state:
                    raise ValueError(
                        "search attempts must have unique resulting filter states; "
                        f"sequences {previous_sequence} and {attempt.sequence} are duplicates"
                    )
            filter_states.append((attempt.sequence, state))

        sequences = [attempt.sequence for attempt in self.attempts]
        if len(sequences) != len(set(sequences)):
            raise ValueError("search attempt sequence values must be unique")
        if sequences != sorted(sequences):
            raise ValueError("search attempts must be ordered by sequence")
        if sequences[0] != 1:
            raise ValueError("first search attempt must have sequence=1")
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError("search attempt sequences must be contiguous")
        if self.attempts[0].mode is not MatchMode.EXACT:
            raise ValueError("first search attempt must use EXACT mode")
        exact_attempts = [
            attempt for attempt in self.attempts if attempt.mode is MatchMode.EXACT
        ]
        if len(exact_attempts) != 1:
            raise ValueError("search plan must contain exactly one EXACT attempt")

        mode_levels = {
            MatchMode.EXACT: 0,
            MatchMode.NEAR: 1,
            MatchMode.ALTERNATIVE: 2,
        }
        levels = [mode_levels[attempt.mode] for attempt in self.attempts]
        if levels != sorted(levels):
            raise ValueError("search attempt modes must not become less permissive")

        baseline_filters = exact_attempts[0].filters
        for attempt in self.attempts[1:]:
            changed_fields = {
                field
                for field in SearchField
                if getattr(baseline_filters, field.value)
                != getattr(attempt.filters, field.value)
            }
            declared_fields = {adjustment.field for adjustment in attempt.adjustments}
            if changed_fields != declared_fields:
                changed_names = sorted(field.value for field in changed_fields)
                declared_names = sorted(field.value for field in declared_fields)
                raise ValueError(
                    "adjustments must describe exactly the changed filter fields; "
                    f"changed={changed_names}, declared={declared_names}"
                )

            for adjustment in attempt.adjustments:
                baseline_value = getattr(baseline_filters, adjustment.field.value)
                if baseline_value is None:
                    raise ValueError(
                        "adjustment field must be populated in the baseline EXACT "
                        f"filters: {adjustment.field.value}"
                    )
                if adjustment.original_value != baseline_value:
                    raise ValueError(
                        "adjustment original_value must match the baseline EXACT "
                        f"filter value: {adjustment.field.value}"
                    )
        return self
