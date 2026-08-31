from datetime import date
from pathlib import Path

from app.ai_engine.inventory_loader import process_inventory
from app.ai_engine.schemas import UnitStatus


def test_process_inventory_validates_sample_csv():
    file_path = Path("data/sample_inventory.csv")

    valid_units, errors = process_inventory(file_path)

    assert len(valid_units) == 4, _format_errors(errors)
    assert len(errors) == 2, _format_errors(errors)

    statuses_by_unit_id = {unit.unit_id: unit.status for unit in valid_units}
    assert statuses_by_unit_id["A101"] == UnitStatus.AVAILABLE
    assert statuses_by_unit_id["B205"] == UnitStatus.HOLD
    assert statuses_by_unit_id["D412"] == UnitStatus.SOLD

    unit_with_missing_source = next(unit for unit in valid_units if unit.unit_id == "A101")
    assert unit_with_missing_source.source == file_path.name
    assert unit_with_missing_source.last_updated == date.today()

    unit_with_missing_last_updated = next(unit for unit in valid_units if unit.unit_id == "D412")
    assert unit_with_missing_last_updated.last_updated == date.today()

    error_rows = {error["row_number"] for error in errors}
    assert error_rows == {6, 7}, _format_errors(errors)


def _format_errors(errors):
    return "\n".join(
        f"Row {error['row_number']}: {error['message']}"
        for error in errors
    )
