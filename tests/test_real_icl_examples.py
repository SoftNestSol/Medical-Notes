from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_real_icl_examples.py"
FEW_SHOT_SCRIPT = ROOT / "src" / "ICL" / "claude_few_shot.py"

spec = importlib.util.spec_from_file_location("build_real_icl_examples", SCRIPT)
assert spec is not None and spec.loader is not None
build_real_icl_examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_real_icl_examples)

few_shot_spec = importlib.util.spec_from_file_location(
    "claude_few_shot", FEW_SHOT_SCRIPT
)
assert few_shot_spec is not None and few_shot_spec.loader is not None
claude_few_shot = importlib.util.module_from_spec(few_shot_spec)
few_shot_spec.loader.exec_module(claude_few_shot)


def test_default_real_icl_examples_are_ready_pool_pairs():
    rows = build_real_icl_examples.read_manifest(
        build_real_icl_examples.DEFAULT_MANIFEST
    )

    ids = [row["conversation_id"] for row in rows]
    assert ids == ["audio18", "audio19"]


def test_build_default_real_icl_examples_validates_notes():
    examples = build_real_icl_examples.build_examples(
        build_real_icl_examples.DEFAULT_MANIFEST
    )

    assert [example["conversation_id"] for example in examples] == [
        "audio18",
        "audio19",
    ]
    assert all(example["transcript"].strip() for example in examples)
    assert all(example["note"]["motivul_prezentarii"] for example in examples)


def test_write_default_real_icl_outputs(tmp_path):
    examples = build_real_icl_examples.build_examples(
        build_real_icl_examples.DEFAULT_MANIFEST
    )

    build_real_icl_examples.write_outputs(examples, tmp_path)

    assert (tmp_path / "real_pool_icl_examples.jsonl").exists()
    prompt = (tmp_path / "real_pool_icl_prompt_block.txt").read_text(
        encoding="utf-8"
    )
    assert "### EXEMPLU ICL 1 (audio18)" in prompt
    assert "### EXEMPLU ICL 2 (audio19)" in prompt
    assert "audio5" not in prompt


def test_claude_few_shot_prompt_contains_audited_examples():
    prompt = claude_few_shot.build_system_prompt(
        build_real_icl_examples.DEFAULT_MANIFEST
    )

    assert "## Exemple ICL reale auditate" in prompt
    assert "### EXEMPLU ICL 1 (audio18)" in prompt
    assert "### EXEMPLU ICL 2 (audio19)" in prompt
    assert "### EXEMPLU ICL 3" not in prompt
