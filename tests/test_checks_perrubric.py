from autoqc.model import Severity
from autoqc.agent.checks import Q04, Q08, Q10, Q11, SEMANTIC_CHECKS


def test_ids_severities_and_rubric_mode():
    assert (Q04.id, Q04.severity, Q04.unit_mode) == ("Q04", Severity.REJECT, "rubric")
    assert (Q08.id, Q08.severity, Q08.unit_mode) == ("Q08", Severity.WARN, "rubric")
    assert (Q10.id, Q10.severity, Q10.unit_mode) == ("Q10", Severity.WARN, "rubric")
    assert (Q11.id, Q11.severity, Q11.unit_mode) == ("Q11", Severity.WARN, "rubric")
    assert all(c in SEMANTIC_CHECKS for c in (Q04, Q08, Q10, Q11))


def test_registry_full_set():
    ids = [c.id for c in SEMANTIC_CHECKS]
    assert set(ids) == {"Q01", "Q02", "Q03", "Q04", "Q05", "Q07", "Q08", "Q10", "Q11", "P02", "P03", "A01", "A02", "A03", "A04", "A05"}
    assert len(ids) == len(set(ids))  # no dupes


def test_guidance_specific():
    assert "obligation" in Q04.guidance.lower() or "cover" in Q04.guidance.lower()
    assert "inverse" in Q08.guidance.lower() or "redundant" in Q08.guidance.lower()
    assert "observed" in Q10.guidance.lower() or "empirical" in Q10.guidance.lower()
    assert "lookup" in Q11.guidance.lower() or "trivia" in Q11.guidance.lower()
