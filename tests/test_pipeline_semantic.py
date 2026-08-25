# tests/test_pipeline_semantic.py
import json
from pathlib import Path
from autoqc.cli import run
from autoqc.llm import FakeLLMClient
from autoqc.model import Verdict


def _good_bundle(root: Path, neg_title: str):
    (root / "tests").mkdir(parents=True)
    (root / "tests/rubrics.json").write_text(json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": neg_title,
         "annotations": {"type": "negative hli verifier", "importance": "must have"}},
    ]))
    (root / "tests/prompt.txt").write_text("q")
    (root / "solution").mkdir()
    (root / "solution/answer.txt").write_text("a")
    (root / "environment").mkdir()
    (root / "environment/Dockerfile").write_text("FROM x")
    (root / "task.toml").write_text(
        '[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def _responder(messages):
    text = " ".join(m["content"] for m in messages)
    return {"passed": ("Does not claim" not in text), "evidence": []}


def test_semantic_runs_when_client_given_good_negative(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _good_bundle(b, "2.1: Claims that X fails")
    out = tmp_path / "out"
    v = run(b, out, llm=FakeLLMClient(_responder))
    assert v is Verdict.SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert any(r["id"] == "Q07" for r in rec["results"])  # semantic ran


def test_semantic_flags_bad_negative(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _good_bundle(b, "2.1: Does not claim that X fails")
    out = tmp_path / "out"
    v = run(b, out, llm=FakeLLMClient(_responder))
    assert v is Verdict.NOT_SOUND  # Q07 is reject-severity
    assert "Q07" in (out / "report.md").read_text()


def test_structural_only_when_no_client(tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EVAL_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    b = tmp_path / "task"; b.mkdir()
    _good_bundle(b, "2.1: Does not claim that X fails")  # would trip Q07 IF semantic ran
    out = tmp_path / "out"
    v = run(b, out)  # no client, no env -> structural only
    assert v is Verdict.SOUND
    rec = json.loads((out / "review_record.json").read_text())
    assert not any(r["id"] == "Q07" for r in rec["results"])  # semantic skipped
