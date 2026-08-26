import json
from pathlib import Path
from autoqc.model import CheckResult, Stage, Severity
from autoqc.calibrate import Case, score_case, summarize, run_corpus
from autoqc.llm import FakeLLMClient


def _cr(cid, passed):
    return CheckResult(id=cid, name=cid, stage=Stage.SEMANTIC, severity=Severity.REJECT, passed=passed)


def test_score_case_clean_all_pass():
    case = Case("clean", Path("."), {})
    s = score_case(case, [_cr("Q07", True), _cr("Q03", True)])
    assert s["checks"]["Q07"]["correct"] and s["checks"]["Q03"]["correct"]
    assert s["verdict_correct"] is True


def test_score_case_q07_caught():
    case = Case("q07_bad", Path("."), {"Q07": {"x"}})
    s = score_case(case, [_cr("Q07", False), _cr("Q03", True)])
    assert s["checks"]["Q07"]["got_fail"] is True and s["checks"]["Q07"]["correct"] is True
    assert s["checks"]["Q03"]["got_fail"] is False and s["checks"]["Q03"]["correct"] is True
    assert s["verdict_correct"] is True


def test_score_case_missed_defect():
    case = Case("q07_bad", Path("."), {"Q07": {"x"}})
    s = score_case(case, [_cr("Q07", True), _cr("Q03", True)])  # missed it
    assert s["checks"]["Q07"]["correct"] is False
    assert s["verdict_correct"] is False


def test_score_case_false_reject_on_clean():
    case = Case("clean", Path("."), {})
    s = score_case(case, [_cr("Q07", True), _cr("Q03", False)])  # wrongly failed Q03
    assert s["checks"]["Q03"]["correct"] is False
    assert s["verdict_correct"] is False


def test_summarize_metrics():
    scored = [
        {"name": "clean", "checks": {"Q07": {"expected_fail": False, "got_fail": False, "correct": True},
                                     "Q03": {"expected_fail": False, "got_fail": True, "correct": False}},
         "verdict_correct": False},
        {"name": "q07_bad", "checks": {"Q07": {"expected_fail": True, "got_fail": True, "correct": True},
                                       "Q03": {"expected_fail": False, "got_fail": False, "correct": True}},
         "verdict_correct": True},
    ]
    m = summarize(scored)
    assert m["recall"] == 1.0            # 1 expected-fail, caught
    assert m["false_reject_rate"] == 1 / 3  # 3 clean checks, 1 wrongly failed
    assert m["verdict_accuracy"] == 0.5


def test_run_corpus_reads_records_and_scores(tmp_path):
    # a bundle whose Q07 negative is bad; fake client fails Q07, passes Q03
    import json as _j
    b = tmp_path / "task"; (b / "tests").mkdir(parents=True)
    (b / "tests/rubrics.json").write_text(_j.dumps([
        {"id": "n" * 32, "title": "2.1: Does not claim that X",
         "annotations": {"type": "negative hli verifier", "importance": "must have"}}]))
    (b / "tests/prompt.txt").write_text("q")
    (b / "solution").mkdir(); (b / "solution/answer.txt").write_text("a")
    (b / "environment").mkdir(); (b / "environment/Dockerfile").write_text("FROM x")
    (b / "task.toml").write_text('[metadata]\nrepository="o/r"\nbase_commit="%s"\n' % ("a" * 40))

    def responder(messages, tools):
        import re
        txt = " ".join(m["content"] for m in messages)
        cid = "Q07" if "Q07" in txt else "Q03"
        ids = list(dict.fromkeys(re.findall(r"criterion_id=(\w+)", txt)))
        return {"tool_calls": [{"id": "s", "name": "submit_findings", "args": {"findings": [
            {"check_id": cid, "criterion_id": i, "passed": cid != "Q07", "evidence": ["x"]} for i in ids]}}]}

    case = Case("q07_bad", b, {"Q07": {"n" * 32}})
    scored = run_corpus([case], FakeLLMClient(responder), tmp_path / "out", k=1)
    assert scored[0]["checks"]["Q07"]["got_fail"] is True
    assert scored[0]["checks"]["Q07"]["correct"] is True
