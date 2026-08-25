import pytest
from autoqc.model import Severity
from autoqc.semantic.base import SemanticJudgment, Unit, SemanticCheck


def test_judgment_defaults():
    j = SemanticJudgment(passed=True)
    assert j.evidence == [] and j.reason == ""


def test_unit_holds_payload():
    u = Unit(key="abc", payload={"title": "1.1: x"})
    assert u.key == "abc" and u.payload["title"] == "1.1: x"


def test_base_check_methods_abstract():
    c = SemanticCheck()
    assert c.severity is Severity.REJECT
    for call in (lambda: c.units(None),
                 lambda: c.proposer_messages(None, None),
                 lambda: c.adversary_messages(None, None, True),
                 lambda: c.parse({})):
        with pytest.raises(NotImplementedError):
            call()
