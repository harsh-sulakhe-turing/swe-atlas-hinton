import json
from pathlib import Path
from autoqc.calibrate import Case, build_corpus


def _write_bundle(root: Path):
    (root / "tests").mkdir(parents=True)
    (root / "tests/rubrics.json").write_text(json.dumps([
        {"id": "p" * 32, "title": "1.1: States the port",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "n" * 32, "title": "2.1: Claims that X fails",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}}]))
    (root / "tests/prompt.txt").write_text("q")
    (root / "solution").mkdir(); (root / "solution/answer.txt").write_text("a")
    (root / "environment").mkdir(); (root / "environment/Dockerfile").write_text("FROM x")
    (root / "task.toml").write_text('[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def test_case_expected_not_sound():
    assert Case("c", Path("."), {}).expected_not_sound is False
    assert Case("b", Path("."), {"Q07": {"x"}}).expected_not_sound is True


def test_build_corpus_writes_three_labeled_variants(tmp_path):
    base = tmp_path / "base"; base.mkdir(); _write_bundle(base)
    cases = build_corpus(base, tmp_path / "work")
    by = {c.name: c for c in cases}
    assert set(by) == {"clean", "q07_bad", "q03_bad"}
    # base is untouched
    assert "or similar" not in (base / "tests/rubrics.json").read_text()
    assert "Does not claim" not in (base / "tests/rubrics.json").read_text()
    # clean is a faithful copy with no defect and no expected flags
    assert by["clean"].expected_flags == {}
    assert (by["clean"].bundle_dir / "tests/rubrics.json").exists()
    # q07_bad has the seeded negative + labels Q07
    assert "Does not claim" in (by["q07_bad"].bundle_dir / "tests/rubrics.json").read_text()
    assert set(by["q07_bad"].expected_flags) == {"Q07"}
    # q03_bad has the wildcard + labels Q03
    assert "or similar" in (by["q03_bad"].bundle_dir / "tests/rubrics.json").read_text()
    assert set(by["q03_bad"].expected_flags) == {"Q03"}
