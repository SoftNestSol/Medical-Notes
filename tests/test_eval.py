from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "src" / "SOTA_EVALUATION" / "eval.py"

spec = importlib.util.spec_from_file_location("chiro_eval", EVAL_PATH)
assert spec is not None and spec.loader is not None
chiro_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chiro_eval)


def _note(**overrides):
    base = {
        "motivul_prezentarii": "durere lombara",
        "evaluarea_durerii_vas": None,
        "localizarea_durerii": [],
        "localizarea_durerii_alta": None,
        "antecedente": [],
        "antecedente_altele": None,
        "medicatie_actuala": [],
        "evaluare_functionala_initiala": None,
    }
    base.update(overrides)
    return base


def _write_note(path: Path, note: dict) -> None:
    path.write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")


def test_score_vas_canonicalizes_ints_and_lists():
    assert chiro_eval.score_vas(5, [5])["correct"]
    assert chiro_eval.score_vas([8, 5], [5, 8])["correct"]
    assert chiro_eval.score_vas(None, None)["correct"]
    assert not chiro_eval.score_vas(None, 5)["correct"]
    assert not chiro_eval.score_vas([5, 5], [5])["correct"]


def test_multilabel_aggregation_rewards_empty_sets_without_hiding_misses():
    vocab = ["cervical", "lombar"]
    scores = [
        {
            "per_label": chiro_eval.score_multilabel([], [], vocab),
            **chiro_eval._score_set([], []),
        },
        {
            "per_label": chiro_eval.score_multilabel(["lombar"], ["lombar", "cervical"], vocab),
            **chiro_eval._score_set(["lombar"], ["lombar", "cervical"]),
        },
    ]

    agg = chiro_eval.aggregate_multilabel(scores, vocab)

    assert agg["micro_f1"] == pytest.approx(2 / 3)
    assert agg["macro_f1"] == pytest.approx(0.5)
    assert agg["sample_f1_mean"] == pytest.approx((1.0 + (2 / 3)) / 2)
    assert agg["exact_match_accuracy"] == pytest.approx(0.5)
    assert agg["empty_ref_accuracy"] == pytest.approx(1.0)


def test_medicatie_aggregation_scores_names_and_doses_separately():
    scores = [
        chiro_eval.score_medicatie(
            [
                {"denumire": "Nurofen", "doza": "400mg"},
                {"denumire": "Concor", "doza": None},
            ],
            [
                {"denumire": "nurofen", "doza": "400mg"},
                {"denumire": "aspirin", "doza": None},
            ],
        ),
        chiro_eval.score_medicatie([], []),
    ]

    agg = chiro_eval.aggregate_medicatie(scores)

    assert agg["name_micro_f1"] == pytest.approx(0.5)
    assert agg["name_sample_f1_mean"] == pytest.approx(0.75)
    assert agg["name_exact_match_accuracy"] == pytest.approx(0.5)
    assert agg["empty_ref_accuracy"] == pytest.approx(1.0)
    assert agg["doza_accuracy_on_matched"] == pytest.approx(1.0)


def test_score_text_normalizes_empty_strings_and_uses_lazy_bertscore(monkeypatch):
    monkeypatch.setattr(chiro_eval, "_bertscore_f1", lambda pred, ref: 0.42)

    assert chiro_eval.score_text("   ", None)["bertscore_f1"] == 1.0
    assert chiro_eval.score_text("diagnostic rostit", None)["bertscore_f1"] == 0.0
    assert chiro_eval.score_text(" diagnostic   rostit ", "diagnostic rostit")[
        "bertscore_f1"
    ] == pytest.approx(0.42)


def test_cli_evaluates_prediction_directory_and_writes_reports(tmp_path):
    pred_dir = tmp_path / "preds"
    ref_dir = tmp_path / "refs"
    out_dir = tmp_path / "eval"
    pred_dir.mkdir()
    ref_dir.mkdir()

    _write_note(
        ref_dir / "audio1.json",
        _note(localizarea_durerii=[], evaluarea_durerii_vas=None),
    )
    _write_note(
        pred_dir / "audio1.json",
        _note(localizarea_durerii=[], evaluarea_durerii_vas=None),
    )
    _write_note(
        ref_dir / "audio2.json",
        _note(localizarea_durerii=["lombar"], evaluarea_durerii_vas=5),
    )
    _write_note(
        pred_dir / "audio2.icl.json",
        _note(localizarea_durerii=["cervical"], evaluarea_durerii_vas=[5]),
    )

    exit_code = chiro_eval.main(
        [
            "--pred-dir",
            str(pred_dir),
            "--ref-dir",
            str(ref_dir),
            "--output-dir",
            str(out_dir),
            "--ids",
            "audio1,audio2",
            "--skip-bertscore",
        ]
    )

    assert exit_code == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_cases"] == 2
    assert summary["fields"]["evaluarea_durerii_vas"]["accuracy"] == 1.0
    assert summary["fields"]["localizarea_durerii"]["exact_match_accuracy"] == 0.5
    assert summary["fields"]["evaluare_functionala_initiala"]["skipped"] == 2
    assert (out_dir / "summary.csv").exists()
    assert len((out_dir / "per_case.jsonl").read_text(encoding="utf-8").splitlines()) == 2
