from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_strategy import ConstraintStrength, SearchField


class SearchAmbiguityKind(str, Enum):
    PARSER_ERROR = "parser_error"
    CONTRADICTORY_BOUNDS = "contradictory_bounds"
    EXCLUSION_NOT_REPRESENTABLE = "exclusion_not_representable"
    VAGUE_NUMERIC = "vague_numeric"
    MULTIPLE_EXPLICIT_VALUES = "multiple_explicit_values"
    SEMANTIC_CONTRADICTION = "semantic_contradiction"
    UNREPRESENTABLE_ALTERNATIVE = "unrepresentable_alternative"
    UNRECOGNIZED_SUBJECT = "unrecognized_subject"
    UNCLASSIFIED = "unclassified"


class SearchFieldDirective(BaseModel):
    field: SearchField
    strength: ConstraintStrength | None = None
    preferred: bool = False
    allow_alternative: bool = False

    @model_validator(mode="after")
    def validate_hard_directive_semantics(self) -> "SearchFieldDirective":
        if (
            self.strength is None
            and not self.preferred
            and not self.allow_alternative
        ):
            raise ValueError(
                "directive must define strength, preference, or alternative behavior"
            )
        if self.strength is not ConstraintStrength.HARD:
            return self
        if self.allow_alternative:
            raise ValueError("hard directive cannot allow alternatives")
        if self.preferred:
            raise ValueError("hard directive cannot be preferred")
        return self


class SearchAmbiguity(BaseModel):
    field: SearchField | None = None
    reason: str
    kind: SearchAmbiguityKind = SearchAmbiguityKind.UNCLASSIFIED

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, reason: str) -> str:
        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("ambiguity reason must not be blank")
        return normalized_reason


class ParsedSearchIntent(BaseModel):
    filters: SearchFilters
    directives: list[SearchFieldDirective] = Field(default_factory=list)
    ambiguities: list[SearchAmbiguity] = Field(default_factory=list)
    unsupported_requirements: list[str] = Field(default_factory=list)

    @field_validator("unsupported_requirements")
    @classmethod
    def normalize_unsupported_requirements(
        cls, requirements: list[str]
    ) -> list[str]:
        normalized_requirements = [requirement.strip() for requirement in requirements]
        if any(not requirement for requirement in normalized_requirements):
            raise ValueError("unsupported requirements must not be blank")
        if len(normalized_requirements) != len(set(normalized_requirements)):
            raise ValueError("unsupported requirements must be unique")
        return normalized_requirements

    @model_validator(mode="after")
    def validate_directives(self) -> "ParsedSearchIntent":
        directive_fields = [directive.field for directive in self.directives]
        if len(directive_fields) != len(set(directive_fields)):
            raise ValueError("directive fields must be unique")

        populated_fields = {
            field
            for field in SearchField
            if getattr(self.filters, field.value) is not None
        }
        for directive in self.directives:
            if directive.field not in populated_fields:
                raise ValueError(
                    "directive field must reference a populated filter: "
                    f"{directive.field.value}"
                )
        return self
