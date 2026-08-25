from __future__ import annotations
from autoqc.model import Severity
from autoqc.semantic.base import SemanticCheck, SemanticJudgment, Unit


def _is_negative(it: dict) -> bool:
    return isinstance(it, dict) and "negative" in str(
        it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")


class NegativeSemanticsCheck(SemanticCheck):
    """Q07: a negative criterion must state the FALSE assertion whose presence should fail
    the answer ("Claims that ..."), never "Does not claim...", an omission, or the correct
    behavior."""
    id = "Q07"
    name = "Negative score-flip semantics"
    severity = Severity.REJECT

    def units(self, bundle):
        items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
        return [Unit(key=str(it.get("id", "?")), payload={"title": str(it.get("title", ""))})
                for it in items if _is_negative(it)]

    def proposer_messages(self, bundle, unit):
        system = (
            "You judge whether a NEGATIVE rubric criterion is phrased correctly. A negative "
            "criterion must state a FALSE assertion whose PRESENCE in an answer should fail it "
            "(for example: 'Claims that every retry uses exponential backoff'). It is INCORRECT "
            "if phrased as an omission statement, as a required absence, or as the correct behavior. "
            'Reply ONLY as JSON: {"passed": <bool>, "reason": <str>, "evidence": [<str>]}. '
            "passed=true means it is correctly phrased as a false assertion to catch."
        )
        user = f"Negative criterion title:\n{unit.payload['title']}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def adversary_messages(self, bundle, unit, agg_passed):
        if agg_passed:
            stance = ("A prior review judged this negative criterion CORRECTLY phrased. Adversarially "
                      "look for ANY way it actually reads as an omission, a 'does not claim' form, or the "
                      "correct behavior rather than a false assertion to catch.")
        else:
            stance = ("A prior review judged this negative criterion INCORRECTLY phrased. Adversarially "
                      "check whether it actually IS a valid false assertion whose presence should fail an answer.")
        system = (
            "You are an adversarial reviewer of negative-rubric phrasing. " + stance +
            ' Reply ONLY as JSON: {"passed": <bool>, "reason": <str>, "evidence": [<str>]}, '
            "where passed=true means the criterion IS correctly phrased as a false assertion to catch."
        )
        user = f"Negative criterion title:\n{unit.payload['title']}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse(self, raw):
        ev = raw.get("evidence") or []
        if not isinstance(ev, list):
            ev = [str(ev)]
        return SemanticJudgment(
            passed=bool(raw.get("passed")),
            evidence=[str(x) for x in ev],
            reason=str(raw.get("reason", "")),
        )


SEMANTIC_CHECKS = [NegativeSemanticsCheck()]
