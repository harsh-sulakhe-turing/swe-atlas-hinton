from autoqc.model import Severity, Stage
from autoqc.semantic.base import SemanticCheck, SemanticJudgment, Unit
from autoqc.llm import FakeLLMClient
from autoqc.semantic.engine import run_check


class StubCheck(SemanticCheck):
    """One unit; proposer verdict driven by the responder; role tag in messages lets the
    responder distinguish proposer vs adversary calls."""
    id = "QX"
    name = "Stub"
    severity = Severity.REJECT

    def units(self, bundle):
        return [Unit(key="u1", payload={})]

    def proposer_messages(self, bundle, unit):
        return [{"role": "system", "content": "PROPOSER"}]

    def adversary_messages(self, bundle, unit, agg_passed):
        return [{"role": "system", "content": f"ADVERSARY agg={agg_passed}"}]

    def parse(self, raw):
        return SemanticJudgment(passed=bool(raw.get("passed")), evidence=raw.get("evidence", []))


def _responder(proposer_passed, adversary_passed):
    def r(messages):
        tag = messages[0]["content"]
        if tag.startswith("PROPOSER"):
            return {"passed": proposer_passed, "evidence": ["p"]}
        return {"passed": adversary_passed, "evidence": ["a"]}
    return r


def test_unanimous_pass_unrefuted_is_sound():
    llm = FakeLLMClient(_responder(True, True))  # adversary attacks a pass but agrees it passes
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert r.id == "QX" and r.stage is Stage.SEMANTIC
    assert r.passed is True and r.needs_human is False


def test_unanimous_reject_unrefuted_stays_reject():
    llm = FakeLLMClient(_responder(False, False))  # adversary defends but agrees it fails
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert r.passed is False and r.needs_human is False
    assert "u1" in r.detail


def test_adversary_overturn_flags_needs_human():
    # proposers unanimously pass, adversary finds it should fail -> overturn
    llm = FakeLLMClient(_responder(True, False))
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert r.passed is True and r.needs_human is True


def test_split_vote_flags_needs_human():
    # alternate proposer verdicts to force a split; adversary agrees with majority
    state = {"n": 0}
    def r(messages):
        if messages[0]["content"].startswith("PROPOSER"):
            state["n"] += 1
            return {"passed": state["n"] % 2 == 0, "evidence": []}  # T/F/T over 3 calls -> 1 pass? adjust
        return {"passed": False, "evidence": []}
    llm = FakeLLMClient(r)
    res = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert res.needs_human is True


def test_evidence_is_capped():
    llm = FakeLLMClient(lambda m: {"passed": True, "evidence": ["x"] * 50})
    r = run_check(StubCheck(), bundle=None, llm=llm, k=3)
    assert len(r.evidence) <= 20
