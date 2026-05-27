"""Frozen test/pool split for the chiropractor_ro dataset.

Per the project's locked decisions (see repo-root AGENTS.md):
- 15 conversation IDs are held out as the test set
- the remaining IDs form the pool used for ICL examples + synthetic seeds
- TEST_IDS must NEVER appear in: ICL few-shot examples, synthetic generation
  seeds, fine-tuning training data, or manual prompt iteration material

The list below is hardcoded on purpose so any change shows up cleanly in a
diff. Do not mutate at runtime. Do not regenerate.

Provenance: drawn from the 30 audios available on 2026-05-26
(`audio1`..`audio14`, `audio20`..`audio35`) using
`random.seed(20260526); random.sample(sorted_ids, 15)`.

POOL_IDS will grow when the additional ~20 conversations land. TEST_IDS will
not.
"""

from __future__ import annotations

from typing import Iterable

# ---------------------------------------------------------------------------
# FROZEN TEST SET — DO NOT EDIT WITHOUT TEAM SIGN-OFF
# ---------------------------------------------------------------------------
TEST_IDS: frozenset[str] = frozenset({
    "audio5",
    "audio6",
    "audio7",
    "audio8",
    "audio11",
    "audio14",
    "audio20",
    "audio22",
    "audio25",
    "audio27",
    "audio30",
    "audio31",
    "audio32",
    "audio34",
    "audio35",
})

# ---------------------------------------------------------------------------
# Pool (extend when new conversations land; never add a TEST_ID here)
# ---------------------------------------------------------------------------
POOL_IDS: frozenset[str] = frozenset({
    "audio1",
    "audio2",
    "audio3",
    "audio4",
    "audio9",
    "audio10",
    "audio12",
    "audio13",
    "audio21",
    "audio23",
    "audio24",
    "audio26",
    "audio28",
    "audio29",
    "audio33",
})

ALL_IDS: frozenset[str] = TEST_IDS | POOL_IDS


class HeldOutLeakageError(AssertionError):
    """Raised when held-out test conversation IDs appear where they must not."""


def assert_no_test_leakage(ids: Iterable[str], *, context: str = "") -> None:
    """Crash if any id in `ids` is a held-out test conversation.

    Call this before: building ICL prompts, selecting synthetic seeds,
    assembling fine-tuning data, or running manual prompt iteration.
    """
    leaked = sorted(set(ids) & TEST_IDS)
    if leaked:
        where = f" in {context}" if context else ""
        raise HeldOutLeakageError(
            f"Test set leakage{where}: {leaked} are held-out IDs and must not "
            f"appear here. See src/data_split.py."
        )


# Self-checks run at import time. Cheap; catches accidental edits immediately.
assert len(TEST_IDS) == 15, f"TEST_IDS must have 15 entries, found {len(TEST_IDS)}"
assert TEST_IDS.isdisjoint(POOL_IDS), "TEST_IDS and POOL_IDS overlap"
