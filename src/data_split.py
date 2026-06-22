"""Frozen test/pool split for the chiropractor_ro dataset.

Per the project's locked decisions (see repo-root AGENTS.md):
- 18 conversation IDs are held out as the test set
- the remaining IDs form the pool used for ICL examples + synthetic seeds
- TEST_IDS must NEVER appear in: ICL few-shot examples, synthetic generation
  seeds, fine-tuning training data, or manual prompt iteration material

The list below is hardcoded on purpose so any change shows up cleanly in a
diff. Do not mutate at runtime. Do not regenerate.

Provenance:
- v1 (2026-05-26): 30 audios available (`audio1`..`audio14`, `audio20`..`audio35`).
  15 randomly drawn for TEST via `random.seed(20260526); random.sample(sorted_ids, 15)`.
- v2 (2026-06-04, signed off): the 5 newly-landed audios (audio15..audio19)
  were classified to maintain a 7/5 (TEST/POOL) ratio across the 12
  currently-hand-corrected refs. audio15, 16, 17 → TEST; audio18, 19 → POOL.
  TEST grew from 15 → 18; POOL grew from 15 → 17 (total 35). This is the
  one allowed exception to the original "TEST_IDS will not grow" rule and
  exists because the 30-audio split could not accommodate the 12 refs
  available on 2026-06-04 with the desired eval coverage (7 of 12 refs
  in TEST, 5 in POOL).

- v3 (2026-06-22): audio36/audio37 were found locally with both conversation
  JSON and structured refs. To guarantee 15 immediately evaluable TEST pairs,
  audio36, audio37, and audio28 moved into TEST; audio8, audio11, and audio14
  moved out to POOL because their structured refs are pending locally. TEST
  remains 18 IDs; TEST_COMPLETE_PAIR_IDS is now 15.

Going forward, TEST_IDS is frozen at 18 again. POOL_IDS may grow.

Current local data availability (2026-06-22):
- all TEST_IDS and POOL_IDS have conversation JSON files under
  data/chiropractor_ro/conversations/
- structured refs are still missing for some IDs; use the *_COMPLETE_PAIR_IDS
  sets below when code needs a guaranteed (conversation, ref) pair
- TEST has 15 complete pairs right now; 3 TEST refs are still pending locally
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
    "audio15",
    "audio16",  # added 2026-06-04 (v2)
    "audio17",  # added 2026-06-04 (v2)
    "audio20",
    "audio22",
    "audio25",
    "audio27",
    "audio28",  # moved from POOL in v3 (2026-06-22)
    "audio30",
    "audio31",
    "audio32",  # ref pending locally
    "audio34",  # ref pending locally
    "audio35",  # ref pending locally
    "audio36",  # added to TEST in v3 (2026-06-22)
    "audio37",  # added to TEST in v3 (2026-06-22)
})

# ---------------------------------------------------------------------------
# Pool (extend when new conversations land; never add a TEST_ID here)
# ---------------------------------------------------------------------------
POOL_IDS: frozenset[str] = frozenset({
    "audio1",
    "audio2",  # ref pending locally
    "audio3",  # ref pending locally
    "audio4",
    "audio8",  # moved from TEST in v3; ref pending locally
    "audio9",  # ref pending locally
    "audio10",
    "audio11",  # moved from TEST in v3; ref pending locally
    "audio12",  # ref pending locally
    "audio13",  # ref pending locally
    "audio14",  # moved from TEST in v3; ref pending locally
    "audio18",  # added 2026-06-04 (v2)
    "audio19",  # added 2026-06-04 (v2)
    "audio21",
    "audio23",
    "audio24",
    "audio26",
    "audio29",
    "audio33",  # ref pending locally
})

ALL_IDS: frozenset[str] = TEST_IDS | POOL_IDS

# Guaranteed local (conversation, structured ref) pairs as of 2026-06-22.
# Use these for synthetic seed rotation, ICL examples, and any code path that
# must read both data/chiropractor_ro/conversations/<id>.json and refs/<id>.json.
TEST_COMPLETE_PAIR_IDS: frozenset[str] = frozenset({
    "audio5",
    "audio6",
    "audio7",
    "audio15",
    "audio16",
    "audio17",
    "audio20",
    "audio22",
    "audio25",
    "audio27",
    "audio28",
    "audio30",
    "audio31",
    "audio36",
    "audio37",
})

POOL_COMPLETE_PAIR_IDS: frozenset[str] = frozenset({
    "audio1",
    "audio4",
    "audio10",
    "audio18",
    "audio19",
    "audio21",
    "audio23",
    "audio24",
    "audio26",
    "audio29",
})

TEST_PENDING_REF_IDS: frozenset[str] = TEST_IDS - TEST_COMPLETE_PAIR_IDS
POOL_PENDING_REF_IDS: frozenset[str] = POOL_IDS - POOL_COMPLETE_PAIR_IDS


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
assert len(TEST_IDS) == 18, f"TEST_IDS must have 18 entries, found {len(TEST_IDS)}"
assert len(POOL_IDS) == 19, f"POOL_IDS must have 19 entries after v3, found {len(POOL_IDS)}"
assert TEST_IDS.isdisjoint(POOL_IDS), "TEST_IDS and POOL_IDS overlap"
assert TEST_COMPLETE_PAIR_IDS <= TEST_IDS, "TEST_COMPLETE_PAIR_IDS must be a subset of TEST_IDS"
assert POOL_COMPLETE_PAIR_IDS <= POOL_IDS, "POOL_COMPLETE_PAIR_IDS must be a subset of POOL_IDS"
assert TEST_PENDING_REF_IDS == TEST_IDS - TEST_COMPLETE_PAIR_IDS
assert POOL_PENDING_REF_IDS == POOL_IDS - POOL_COMPLETE_PAIR_IDS
assert TEST_COMPLETE_PAIR_IDS.isdisjoint(POOL_COMPLETE_PAIR_IDS)
assert len(TEST_COMPLETE_PAIR_IDS) == 15, (
    f"TEST_COMPLETE_PAIR_IDS must have 15 immediately evaluable entries, "
    f"found {len(TEST_COMPLETE_PAIR_IDS)}"
)
