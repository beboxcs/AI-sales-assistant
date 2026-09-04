import pytest
from pydantic import ValidationError

from app.ai_engine.search_filters import SearchFilters
from app.ai_engine.search_planner import (
    AdjustmentAction,
    FieldAdjustment,
    SearchAttempt,
    SearchPlan,
)
from app.ai_engine.search_strategy import MatchMode, SearchField


def _attempt(sequence, mode, filters=None, adjustments=None):
    return SearchAttempt(
        sequence=sequence,
        mode=mode,
        filters=filters or SearchFilters(),
        adjustments=adjustments or [],
    )


def _widened_bedroom_adjustment():
    return FieldAdjustment(
        field=SearchField.BEDROOMS,
        action=AdjustmentAction.WIDENED,
        original_value=3,
        adjusted_value=2,
    )


def _dropped_bedroom_adjustment():
    return FieldAdjustment(
        field=SearchField.BEDROOMS,
        action=AdjustmentAction.DROPPED,
        original_value=3,
        adjusted_value=None,
    )


def test_valid_exact_only_plan():
    plan = SearchPlan(attempts=[_attempt(1, MatchMode.EXACT)])

    assert plan.attempts[0].sequence == 1


def test_valid_exact_near_and_alternative_progression():
    near_filters = SearchFilters(bedrooms=2)
    alternative_filters = SearchFilters()
    plan = SearchPlan(
        attempts=[
            _attempt(1, MatchMode.EXACT, SearchFilters(bedrooms=3)),
            _attempt(
                2,
                MatchMode.NEAR,
                near_filters,
                [_widened_bedroom_adjustment()],
            ),
            _attempt(
                3,
                MatchMode.ALTERNATIVE,
                alternative_filters,
                [_dropped_bedroom_adjustment()],
            ),
        ]
    )

    assert [attempt.mode for attempt in plan.attempts] == [
        MatchMode.EXACT,
        MatchMode.NEAR,
        MatchMode.ALTERNATIVE,
    ]


def test_multiple_near_attempts_are_allowed():
    plan = SearchPlan(
        attempts=[
            _attempt(1, MatchMode.EXACT, SearchFilters(bedrooms=3)),
            _attempt(
                2,
                MatchMode.NEAR,
                SearchFilters(bedrooms=2),
                [_widened_bedroom_adjustment()],
            ),
            _attempt(
                3,
                MatchMode.NEAR,
                SearchFilters(bedrooms=1),
                [
                    FieldAdjustment(
                        field="bedrooms",
                        action="widened",
                        original_value=3,
                        adjusted_value=1,
                    )
                ],
            ),
        ]
    )

    assert len(plan.attempts) == 3


def test_exact_attempt_cannot_have_adjustments():
    with pytest.raises(ValidationError, match="EXACT"):
        _attempt(
            1,
            MatchMode.EXACT,
            SearchFilters(bedrooms=2),
            [_widened_bedroom_adjustment()],
        )


def test_near_accepts_widened_but_rejects_dropped_adjustments():
    attempt = _attempt(
        2,
        MatchMode.NEAR,
        SearchFilters(bedrooms=2),
        [_widened_bedroom_adjustment()],
    )
    assert attempt.adjustments[0].action is AdjustmentAction.WIDENED

    with pytest.raises(ValidationError, match="NEAR"):
        _attempt(
            2,
            MatchMode.NEAR,
            SearchFilters(),
            [_dropped_bedroom_adjustment()],
        )


def test_alternative_accepts_dropped_adjustment():
    attempt = _attempt(
        2,
        MatchMode.ALTERNATIVE,
        SearchFilters(),
        [_dropped_bedroom_adjustment()],
    )

    assert attempt.adjustments[0].adjusted_value is None


def test_adjustment_filter_population_must_match_action():
    with pytest.raises(ValidationError, match="remain populated"):
        _attempt(
            2,
            MatchMode.NEAR,
            SearchFilters(),
            [_widened_bedroom_adjustment()],
        )

    with pytest.raises(ValidationError, match="must be None"):
        _attempt(
            2,
            MatchMode.ALTERNATIVE,
            SearchFilters(bedrooms=2),
            [_dropped_bedroom_adjustment()],
        )


def test_widened_adjustment_must_match_resulting_filter_value():
    with pytest.raises(ValidationError, match="match the attempt filter"):
        _attempt(
            2,
            MatchMode.NEAR,
            SearchFilters(bedrooms=1),
            [_widened_bedroom_adjustment()],
        )


def test_widened_adjustment_must_change_the_original_value():
    with pytest.raises(ValidationError, match="must change"):
        FieldAdjustment(
            field="bedrooms",
            action="widened",
            original_value=3,
            adjusted_value=3,
        )


@pytest.mark.parametrize("mode", [MatchMode.NEAR, MatchMode.ALTERNATIVE])
def test_non_exact_attempts_require_adjustments(mode):
    with pytest.raises(ValidationError, match="require adjustments"):
        _attempt(2, mode)


def test_adjustment_fields_must_be_unique():
    with pytest.raises(ValidationError, match="adjustment fields must be unique"):
        _attempt(
            2,
            MatchMode.NEAR,
            SearchFilters(bedrooms=2),
            [_widened_bedroom_adjustment(), _widened_bedroom_adjustment()],
        )


def test_field_adjustment_values_have_required_nullability():
    with pytest.raises(ValidationError, match="original_value"):
        FieldAdjustment(
            field="bedrooms",
            action="widened",
            original_value=None,
            adjusted_value=2,
        )

    with pytest.raises(ValidationError, match="adjusted_value"):
        FieldAdjustment(
            field="bedrooms",
            action="widened",
            original_value=3,
            adjusted_value=None,
        )

    with pytest.raises(ValidationError, match="adjusted_value"):
        FieldAdjustment(
            field="bedrooms",
            action="dropped",
            original_value=3,
            adjusted_value=2,
        )


def test_search_plan_structure_is_validated():
    with pytest.raises(ValidationError, match="at least one"):
        SearchPlan(attempts=[])

    with pytest.raises(ValidationError, match="unique"):
        SearchPlan(
            attempts=[
                _attempt(1, "exact"),
                _attempt(
                    1,
                    "near",
                    SearchFilters(bedrooms=2),
                    [_widened_bedroom_adjustment()],
                ),
            ]
        )

    with pytest.raises(ValidationError, match="sequence=1"):
        SearchPlan(attempts=[_attempt(2, "exact")])

    with pytest.raises(ValidationError, match="contiguous"):
        SearchPlan(
            attempts=[
                _attempt(1, "exact", SearchFilters(bedrooms=3)),
                _attempt(
                    3,
                    "near",
                    SearchFilters(bedrooms=2),
                    [_widened_bedroom_adjustment()],
                ),
            ]
        )

    with pytest.raises(ValidationError, match="ordered"):
        SearchPlan(
            attempts=[
                _attempt(
                    2,
                    "near",
                    SearchFilters(bedrooms=2),
                    [_widened_bedroom_adjustment()],
                ),
                _attempt(1, "exact", SearchFilters(bedrooms=3)),
            ]
        )


def test_search_plan_rejects_multiple_exact_attempts():
    with pytest.raises(ValidationError, match="exactly one EXACT"):
        SearchPlan(
            attempts=[
                _attempt(1, "exact", SearchFilters(bedrooms=3)),
                _attempt(2, "exact", SearchFilters(bedrooms=2)),
            ]
        )


def test_adjustment_original_value_must_match_exact_baseline():
    with pytest.raises(ValidationError, match="baseline EXACT"):
        SearchPlan(
            attempts=[
                _attempt(1, "exact", SearchFilters(bedrooms=3)),
                _attempt(
                    2,
                    "near",
                    SearchFilters(bedrooms=1),
                    [
                        FieldAdjustment(
                            field="bedrooms",
                            action="widened",
                            original_value=2,
                            adjusted_value=1,
                        )
                    ],
                ),
            ]
        )


def test_adjustment_field_must_exist_in_exact_baseline():
    with pytest.raises(ValidationError, match="populated in the baseline"):
        SearchPlan(
            attempts=[
                _attempt(1, "exact", SearchFilters()),
                _attempt(
                    2,
                    "near",
                    SearchFilters(bedrooms=2),
                    [_widened_bedroom_adjustment()],
                ),
            ]
        )


def test_progressive_near_adjustments_stay_anchored_to_exact_baseline():
    plan = SearchPlan(
        attempts=[
            _attempt(1, "exact", SearchFilters(max_price=8_000_000)),
            _attempt(
                2,
                "near",
                SearchFilters(max_price=8_400_000),
                [
                    FieldAdjustment(
                        field="max_price",
                        action="widened",
                        original_value=8_000_000,
                        adjusted_value=8_400_000,
                    )
                ],
            ),
            _attempt(
                3,
                "near",
                SearchFilters(max_price=8_800_000),
                [
                    FieldAdjustment(
                        field="max_price",
                        action="widened",
                        original_value=8_000_000,
                        adjusted_value=8_800_000,
                    )
                ],
            ),
        ]
    )

    assert [adjustment.original_value for attempt in plan.attempts[1:] for adjustment in attempt.adjustments] == [
        8_000_000,
        8_000_000,
    ]


def test_near_cannot_change_a_second_field_without_an_adjustment():
    with pytest.raises(ValidationError, match="exactly the changed filter fields"):
        SearchPlan(
            attempts=[
                _attempt(
                    1,
                    "exact",
                    SearchFilters(bedrooms=3, max_price=8_000_000),
                ),
                _attempt(
                    2,
                    "near",
                    SearchFilters(bedrooms=2, max_price=8_400_000),
                    [_widened_bedroom_adjustment()],
                ),
            ]
        )


def test_alternative_cannot_drop_a_baseline_field_without_an_adjustment():
    with pytest.raises(ValidationError, match="exactly the changed filter fields"):
        SearchPlan(
            attempts=[
                _attempt(
                    1,
                    "exact",
                    SearchFilters(location="New Cairo", bedrooms=3),
                ),
                _attempt(
                    2,
                    "alternative",
                    SearchFilters(bedrooms=2),
                    [_widened_bedroom_adjustment()],
                ),
            ]
        )


def test_multiple_changed_fields_require_matching_adjustments():
    plan = SearchPlan(
        attempts=[
            _attempt(
                1,
                "exact",
                SearchFilters(bedrooms=3, max_price=8_000_000),
            ),
            _attempt(
                2,
                "near",
                SearchFilters(bedrooms=2, max_price=8_400_000),
                [
                    FieldAdjustment(
                        field="max_price",
                        action="widened",
                        original_value=8_000_000,
                        adjusted_value=8_400_000,
                    ),
                    _widened_bedroom_adjustment(),
                ],
            ),
        ]
    )

    assert len(plan.attempts[1].adjustments) == 2


def test_unchanged_baseline_fields_need_no_adjustments():
    plan = SearchPlan(
        attempts=[
            _attempt(1, "exact", SearchFilters(location="New Cairo", bedrooms=3)),
            _attempt(
                2,
                "near",
                SearchFilters(location="New Cairo", bedrooms=2),
                [_widened_bedroom_adjustment()],
            ),
        ]
    )

    assert plan.attempts[1].filters.location == "New Cairo"


def test_direct_exact_to_alternative_is_valid():
    plan = SearchPlan(
        attempts=[
            _attempt(1, "exact", SearchFilters(bedrooms=3)),
            _attempt(
                2,
                "alternative",
                SearchFilters(),
                [_dropped_bedroom_adjustment()],
            ),
        ]
    )

    assert plan.attempts[1].mode is MatchMode.ALTERNATIVE


def test_multiple_alternative_attempts_with_distinct_filter_states_are_valid():
    plan = SearchPlan(
        attempts=[
            _attempt(
                1,
                "exact",
                SearchFilters(location="New Cairo", bedrooms=3),
            ),
            _attempt(
                2,
                "alternative",
                SearchFilters(location="New Cairo"),
                [_dropped_bedroom_adjustment()],
            ),
            _attempt(
                3,
                "alternative",
                SearchFilters(),
                [
                    FieldAdjustment(
                        field="location",
                        action="dropped",
                        original_value="New Cairo",
                        adjusted_value=None,
                    ),
                    FieldAdjustment(
                        field="bedrooms",
                        action="dropped",
                        original_value=3,
                        adjusted_value=None,
                    )
                ],
            ),
        ]
    )

    assert len(plan.attempts) == 3


def test_duplicate_non_exact_resulting_filter_states_are_rejected():
    with pytest.raises(ValidationError, match="unique resulting filter states"):
        SearchPlan(
            attempts=[
                _attempt(1, "exact", SearchFilters(bedrooms=3)),
                _attempt(
                    2,
                    "near",
                    SearchFilters(bedrooms=2),
                    [_widened_bedroom_adjustment()],
                ),
                _attempt(
                    3,
                    "near",
                    SearchFilters(bedrooms=2),
                    [
                        FieldAdjustment(
                            field="bedrooms",
                            action="widened",
                            original_value=3,
                            adjusted_value=2,
                        )
                    ],
                ),
            ]
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_float_adjustment_values_must_be_finite(value):
    with pytest.raises(ValidationError):
        FieldAdjustment(
            field="max_price",
            action="widened",
            original_value=value,
            adjusted_value=9_000_000,
        )

    with pytest.raises(ValidationError):
        FieldAdjustment(
            field="max_price",
            action="widened",
            original_value=8_000_000,
            adjusted_value=value,
        )


@pytest.mark.parametrize(
    "modes",
    [
        ["near"],
        ["exact", "alternative", "near"],
    ],
)
def test_search_plan_rejects_invalid_mode_progression(modes):
    if modes == ["near"]:
        attempts = [
            _attempt(
                1,
                "near",
                SearchFilters(bedrooms=2),
                [_widened_bedroom_adjustment()],
            )
        ]
    else:
        attempts = [
            _attempt(1, "exact", SearchFilters(bedrooms=3)),
            _attempt(
                2,
                "alternative",
                SearchFilters(),
                [_dropped_bedroom_adjustment()],
            ),
            _attempt(
                3,
                "near",
                SearchFilters(bedrooms=2),
                [_widened_bedroom_adjustment()],
            ),
        ]

    with pytest.raises(ValidationError, match="mode"):
        SearchPlan(attempts=attempts)


def test_model_dump_json_is_safe():
    plan = SearchPlan(
        attempts=[
            _attempt(1, "exact", SearchFilters(bedrooms=3)),
            _attempt(
                2,
                "near",
                SearchFilters(bedrooms=2),
                [_widened_bedroom_adjustment()],
            ),
        ]
    )

    dumped = plan.model_dump(mode="json")

    assert dumped["attempts"][1]["mode"] == "near"
    assert dumped["attempts"][1]["adjustments"][0] == {
        "field": "bedrooms",
        "action": "widened",
        "original_value": 3,
        "adjusted_value": 2,
    }
