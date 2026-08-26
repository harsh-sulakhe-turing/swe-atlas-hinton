from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from autoqc.model import Severity
from autoqc.agent.runner import Role
from autoqc.agent.tools import default_tools


@dataclass
class SemanticCheck:
    id: str
    name: str
    severity: Severity
    scope: Callable          # (items) -> list[dict]
    guidance: str
    unit_mode: str = "criterion"


def _is_dict_crit(it):
    return isinstance(it, dict) and it.get("id")


def _negatives(items):
    return [it for it in (items or [])
            if _is_dict_crit(it) and "negative" in str(
                it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")]


def _all_criteria(items):
    return [it for it in (items or []) if _is_dict_crit(it)]


Q07 = SemanticCheck(
    id="Q07", name="Negative score-flip semantics", severity=Severity.REJECT, scope=_negatives,
    guidance=("A NEGATIVE criterion must state the FALSE assertion whose PRESENCE in an answer "
              "should fail it (for example 'Claims that every retry uses exponential backoff'). "
              "It VIOLATES this check if phrased as 'Does not claim...', as a required omission, "
              "or as the correct behavior. passed=true means correctly phrased."))

Q03 = SemanticCheck(
    id="Q03", name="No wildcard / escape hatch", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this check if it can be satisfied by open-ended wording: a "
              "bare trailing '*', 'or similar', 'or other', 'and so on', or an e.g./such-as list "
              "whose items are NOT interchangeable full answers. IMPORTANT: an e.g./such as list is "
              "FINE and common when every listed item is an interchangeable way to satisfy the same "
              "requirement — do NOT flag interchangeable examples. passed=true means no escape hatch."))

SEMANTIC_CHECKS = [Q07, Q03]

_PROPOSER_SYS = (
    "You are a strict, evidence-based QC reviewer of grading rubrics. Judge only the check and "
    "criteria you are given. You may read bundle files with the tools if helpful. Finish by "
    "calling submit_findings with exactly one finding per listed criterion; evidence must quote "
    "the criterion text you relied on.")

_ADVERSARY_SYS = (
    "You are an adversarial second reviewer of grading-rubric QC. For each criterion a prior "
    "review marked FAIL, argue whether it is actually acceptable (defend it). For each marked "
    "PASS, look for a violation the first reviewer missed (attack it). Finish by calling "
    "submit_findings with your verdict per criterion; passed=true means the criterion is fine.")


def proposer_role() -> Role:
    return Role(name="proposer", system_prompt=_PROPOSER_SYS, tools=default_tools())


def adversary_role() -> Role:
    return Role(name="adversary", system_prompt=_ADVERSARY_SYS, tools=default_tools())


def _criteria_block(criteria) -> str:
    return "\n".join(f"- criterion_id={c['id']}  title={c.get('title','')!r}" for c in criteria)


def _full_rubric_block(bundle) -> str:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        typ = "neg" if "negative" in str(it.get("annotations", {}).get("type", "")) else "pos"
        lines.append(f"- [{typ}] criterion_id={it.get('id')}  title={it.get('title', '')!r}")
    return "\n".join(lines)


def proposer_context(bundle, check, criteria) -> str:
    if check.unit_mode == "rubric":
        return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
                f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"The full rubric:\n{_full_rubric_block(bundle)}\n\n"
                "Judge the rubric AS A WHOLE for this check. Submit EXACTLY ONE finding with "
                f"check_id={check.id} and criterion_id=\"rubric\": passed=true if the rubric "
                "SATISFIES the check, passed=false if it VIOLATES it; evidence must justify it.")
    return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            f"Judge each of these criteria (submit one finding per criterion, check_id={check.id}):\n"
            f"{_criteria_block(criteria)}\n\n"
            "passed=true if the criterion SATISFIES the check, passed=false if it VIOLATES it.")


def adversary_context(bundle, check, criteria, agg) -> str:
    if check.unit_mode == "rubric":
        v = agg.get("rubric", {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"The full rubric:\n{_full_rubric_block(bundle)}\n\n"
                f"A prior review judged the whole rubric: {verdict}. Challenge that per the rules "
                f"above, then submit EXACTLY ONE finding (check_id={check.id}, criterion_id=\"rubric\").")
    lines = []
    for c in criteria:
        v = agg.get(c["id"], {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        lines.append(f"- criterion_id={c['id']}  prior_verdict={verdict}  title={c.get('title','')!r}")
    return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            "A prior review produced these verdicts. Challenge them per the rules above, then "
            f"submit_findings with your verdict per criterion (check_id={check.id}):\n"
            + "\n".join(lines))
