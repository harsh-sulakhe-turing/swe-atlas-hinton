import json
import types
from pathlib import Path
from autoqc.cli import run
from autoqc.llm import FakeLLMClient
from autoqc.model import Verdict


def _bundle(root: Path, neg_title):
    (root / "tests").mkdir(parents=True)
    (root / "tests/rubrics.json").write_text(json.dumps([
        {"id": "a" * 32, "title": "1.1: States X",
         "annotations": {"type": "positive hli verifier", "importance": "must have"}},
        {"id": "b" * 32, "title": neg_title,
         "annotations": {"type": "negative hli verifier", "importance": "must have"}}]))
    (root / "tests/prompt.txt").write_text("q")
    (root / "solution").mkdir(); (root / "solution/answer.txt").write_text("a")
    (root / "environment").mkdir(); (root / "environment/Dockerfile").write_text("FROM x")
    (root / "task.toml").write_text('[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))


def _submit(findings):
    return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": findings}}]}


def test_semantic_runs_with_client_all_pass(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _bundle(b, "2.1: Claims that X fails")
    # every criterion passes both checks
    def responder(messages, tools):
        import re
        ids = re.findall(r"criterion_id=(\w+)", " ".join(m["content"] for m in messages))
        cid = "Q07" if "Q07" in " ".join(m["content"] for m in messages) else "Q03"
        return _submit([{"check_id": cid, "criterion_id": i, "passed": True, "evidence": ["ok"]}
                        for i in dict.fromkeys(ids)])
    v = run(b, tmp_path / "out", llm=FakeLLMClient(responder))
    rec = json.loads((tmp_path / "out/review_record.json").read_text())
    assert {"Q07", "Q03"} <= {r["id"] for r in rec["results"]}
    assert v is Verdict.SOUND


def test_semantic_reject_makes_not_sound(tmp_path):
    b = tmp_path / "task"; b.mkdir()
    _bundle(b, "2.1: Does not claim that X fails")  # a Q07 violation
    def responder(messages, tools):
        import re
        txt = " ".join(m["content"] for m in messages)
        cid = "Q07" if "Q07" in txt else "Q03"
        ids = list(dict.fromkeys(re.findall(r"criterion_id=(\w+)", txt)))
        # fail the negative on Q07, pass everything else
        return _submit([{"check_id": cid, "criterion_id": i,
                         "passed": not (cid == "Q07"), "evidence": ["x"]} for i in ids])
    v = run(b, tmp_path / "out", llm=FakeLLMClient(responder))
    assert v is Verdict.NOT_SOUND
    assert "Q07" in (tmp_path / "out/report.md").read_text()


def test_structural_only_when_no_client(tmp_path, monkeypatch):
    for var in ("EVAL_API_KEY", "EVAL_BASE_URL", "OPENAI_API_KEY", "OPENAI_API_BASE"):
        monkeypatch.delenv(var, raising=False)
    b = tmp_path / "task"; b.mkdir()
    _bundle(b, "2.1: Does not claim that X fails")
    v = run(b, tmp_path / "out")  # no client, no env
    rec = json.loads((tmp_path / "out/review_record.json").read_text())
    assert not any(r["id"] in ("Q07", "Q03") for r in rec["results"])
    assert v is Verdict.SOUND
