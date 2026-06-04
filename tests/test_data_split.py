"""Tests for the frozen test/pool split.

Run: `pytest tests/test_data_split.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_split import (  # noqa: E402
    ALL_IDS,
    POOL_IDS,
    TEST_IDS,
    HeldOutLeakageError,
    assert_no_test_leakage,
)


def test_test_ids_has_exactly_18_entries():
    assert len(TEST_IDS) == 18


def test_pool_ids_has_exactly_17_entries():
    assert len(POOL_IDS) == 17


def test_test_and_pool_are_disjoint_and_cover_all_ids():
    assert TEST_IDS.isdisjoint(POOL_IDS)
    assert TEST_IDS | POOL_IDS == ALL_IDS
    assert len(TEST_IDS) + len(POOL_IDS) == len(ALL_IDS)


def test_assert_no_test_leakage_passes_on_pool_only():
    assert_no_test_leakage(POOL_IDS)
    assert_no_test_leakage([])
    assert_no_test_leakage(["audio1", "audio2"])


def test_assert_no_test_leakage_raises_on_contaminated_input():
    contaminated = list(POOL_IDS)[:3] + [next(iter(TEST_IDS))]
    with pytest.raises(HeldOutLeakageError):
        assert_no_test_leakage(contaminated)


def test_assert_no_test_leakage_raises_on_single_test_id():
    with pytest.raises(HeldOutLeakageError):
        assert_no_test_leakage(["audio5"])


def test_assert_no_test_leakage_error_message_lists_leaked_ids():
    with pytest.raises(HeldOutLeakageError) as exc:
        assert_no_test_leakage(["audio5", "audio11"], context="ICL selection")
    msg = str(exc.value)
    assert "audio5" in msg and "audio11" in msg
    assert "ICL selection" in msg
