from __future__ import annotations
from autoqc.model import CheckResult, Stage, Severity


def _criteria(bundle):
    items = getattr(bundle, "rubrics", None)
    return [it for it in items if isinstance(it, dict)] if isinstance(items, list) else []


def _is_negative(it):
    ann = it.get("annotations")
    return "negative" in str(ann.get("type", "") if isinstance(ann, dict) else "")


def check_q09(bundle) -> CheckResult:
    has_neg = any(_is_negative(it) for it in _criteria(bundle))
    return CheckResult(id="Q09", name="Negative present", stage=Stage.SEMANTIC,
                       severity=Severity.WARN, passed=has_neg,
                       detail="" if has_neg else "no negative criterion present")


def check_q12_size(bundle, limit: int = 18) -> CheckResult:
    n = len(_criteria(bundle))
    ok = n <= limit
    return CheckResult(id="Q12", name="Economical size", stage=Stage.SEMANTIC,
                       severity=Severity.WARN, passed=ok,
                       detail="" if ok else f"{n} criteria (>{limit}) — routes to human review")


DETERMINISTIC_CHECKS = [check_q09, check_q12_size]
