"""Tests for indexed dataset core logic."""

import pytest

from avp_vit.train.data.indexed import SCHEMA_VERSION, parse_index_metadata, split_by_class


def make_raw_metadata(**overrides: str) -> dict[bytes, bytes]:
    defaults = {
        "schema_version": str(SCHEMA_VERSION),
        "root_name": "imagenet",
        "split": "train",
        "val_ratio": "0.1",
        "seed": "42",
        "n_samples": "1000",
        "n_classes": "10",
        "generated_at": "2025-01-01T00:00:00Z",
    }
    defaults.update(overrides)
    return {k.encode(): v.encode() for k, v in defaults.items()}


def test_parse_metadata_valid() -> None:
    meta = parse_index_metadata(make_raw_metadata())
    assert meta.schema_version == SCHEMA_VERSION
    assert meta.root_name == "imagenet"
    assert meta.split == "train"


def test_parse_metadata_wrong_schema_version() -> None:
    with pytest.raises(ValueError, match="Schema version mismatch"):
        parse_index_metadata(make_raw_metadata(schema_version="99"))


def test_split_deterministic() -> None:
    data = {"a": [f"{i}.jpg" for i in range(20)], "b": [f"{i}.jpg" for i in range(20)]}
    t1, v1 = split_by_class(data, 0.2, seed=42)
    t2, v2 = split_by_class(data, 0.2, seed=42)
    assert t1 == t2 and v1 == v2


def test_split_no_data_loss() -> None:
    data = {"x": [f"{i}.jpg" for i in range(100)], "y": [f"{i}.jpg" for i in range(50)]}
    train, val = split_by_class(data, 0.1, seed=0)
    assert len(train) + len(val) == 150
    assert len({r[0] for r in train + val}) == 150  # all unique


def test_split_at_least_one_val_per_class() -> None:
    data = {"tiny": ["a.jpg", "b.jpg"]}
    train, val = split_by_class(data, 0.01, seed=0)  # 0.01 * 2 = 0, but max(1, 0) = 1
    assert len(val) >= 1
