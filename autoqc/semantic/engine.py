from __future__ import annotations
from autoqc.model import CheckResult, Stage
from autoqc.semantic.base import SemanticCheck


def run_check(check: SemanticCheck, bundle, llm, k: int = 3) -> CheckResult:
    unit_passed: list[bool] = []
    unit_needs_human: list[bool] = []
    failing_keys: list[str] = []
    evidence: list[str] = []

    for unit in check.units(bundle):
        judgments = [check.parse(llm.judge(check.proposer_messages(bundle, unit)))
                     for _ in range(k)]
        votes = sum(1 for j in judgments if j.passed)
        agg_passed = votes * 2 > k
        split = votes not in (0, k)

        adv = check.parse(llm.judge(check.adversary_messages(bundle, unit, agg_passed)))
        overturn = adv.passed != agg_passed
        needs_human = split or overturn

        unit_passed.append(agg_passed)
        unit_needs_human.append(needs_human)
        if not agg_passed:
            failing_keys.append(unit.key)
        for j in judgments + [adv]:
            evidence.extend(j.evidence)

    passed = all(unit_passed)  # True when there are no units
    needs_human = any(unit_needs_human)
    detail = "" if passed else "failing units: " + ", ".join(failing_keys)
    return CheckResult(
        id=check.id, name=check.name, stage=Stage.SEMANTIC, severity=check.severity,
        passed=passed, needs_human=needs_human, evidence=evidence[:20], detail=detail,
    )
