"""Deterministic tests with synthetic data only."""
from pathlib import Path

import pytest

from app.data.catalog import DatasetSpec
from app.data.engine import aggregate
from app.tools.structured import StructuredRequest, query_dataset


@pytest.fixture
def sample(tmp_path: Path) -> DatasetSpec:
    path = tmp_path / "sales.csv"
    path.write_text(
        "store,amount\nNorth,100\nSouth,50\nNorth,25\n", encoding="utf-8"
    )
    return DatasetSpec(
        name="sales", csv_path=path,
        dimensions=("store",), measures=("amount",),
    )


def test_grouped_sum(sample: DatasetSpec) -> None:
    pytest.importorskip("duckdb")
    result = aggregate(sample, metric="amount", operation="sum", group_by="store")
    assert result.rows == [
        {"dimension": "North", "value": 125},
        {"dimension": "South", "value": 50},
    ]


def test_filter_parameterization(sample: DatasetSpec) -> None:
    pytest.importorskip("duckdb")
    result = aggregate(
        sample, metric="amount", operation="sum",
        filters={"store": "North' OR 1=1 --"},
    )
    assert result.rows[0]["value"] is None


def test_reject_unapproved_column(sample: DatasetSpec) -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        aggregate(sample, metric="password", operation="sum")


def test_reject_unapproved_group(sample: DatasetSpec) -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        aggregate(sample, metric="amount", operation="sum", group_by="secret")


def test_reject_unknown_dataset(sample: DatasetSpec) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        query_dataset(
            StructuredRequest(dataset="inventory", metric="amount"),
            {"sales": sample},
        )


def test_reject_bad_identifier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identifier"):
        DatasetSpec(
            name="sales; DROP TABLE x", csv_path=tmp_path / "x.csv",
            dimensions=("store",), measures=("amount",),
        )
