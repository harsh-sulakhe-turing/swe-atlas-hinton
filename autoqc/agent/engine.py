from __future__ import annotations

from autoqc.model import CheckResult, Stage
from autoqc.agent.runner import run_agent
from autoqc.agent.checks import (SEMANTIC_CHECKS, proposer_role, adversary_role,
                                 proposer_context, adversary_context)
from autoqc.agent.tools import validate_findings


def aggregate(finding_sets, allowed_ids):
    out = {}
    n = len(finding_sets)
    # collapse each pass to at most one verdict per criterion (first occurrence)
    per_pass = []
    for fs in finding_sets:
        seen = {}
        for f in fs:
            cid = f.get("criterion_id")
            if cid in allowed_ids and cid not in seen:
                seen[cid] = (bool(f.get("passed")), list(f.get("evidence") or []))
        per_pass.append(seen)
    for cid in allowed_ids:
        votes, evidence = [], []
        for seen in per_pass:
            if cid in seen:
                p, ev = seen[cid]
                votes.append(p)
                evidence.extend(ev)
        if not votes:
            out[cid] = {"passed": False, "split": True, "evidence": evidence}
            continue
        passed = sum(votes) * 2 > len(votes)
        split = (len(set(votes)) > 1) or (len(votes) < n)
        out[cid] = {"passed": passed, "split": split, "evidence": evidence}
    return out


def adjudicate(agg, adversary_findings):
    adv = {f.get("criterion_id"): bool(f.get("passed"))
           for f in (adversary_findings or []) if f.get("criterion_id")}
    out = {}
    for cid, v in agg.items():
        overturn = cid in adv and adv[cid] != v["passed"]
        out[cid] = {"passed": v["passed"], "needs_human": bool(v["split"] or overturn)}
    return out


def _own(findings, check_id, allowed_ids):
    valid, _ = validate_findings(findings, allowed_ids)
    return [f for f in valid if f.get("check_id") == check_id]


def run_check(check, bundle, client, ctx, k=3, votes_log=None) -> CheckResult:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    criteria = check.scope(items)
    if not criteria:
        return CheckResult(id=check.id, name=check.name, stage=Stage.SEMANTIC,
                           severity=check.severity, passed=True)
    allowed = {c["id"] for c in criteria}

    finding_sets = []
    p_ctx = proposer_context(bundle, check, criteria)
    for _ in range(k):
        res = run_agent(proposer_role(), p_ctx, client, ctx)
        if votes_log is not None:
            votes_log.append({"check": check.id, "role": "proposer",
                              "ok": res.ok, "findings": res.findings})
        finding_sets.append(_own(res.findings, check.id, allowed) if res.ok else [])

    agg = aggregate(finding_sets, allowed)
    adv_res = run_agent(adversary_role(), adversary_context(bundle, check, criteria, agg), client, ctx)
    if votes_log is not None:
        votes_log.append({"check": check.id, "role": "adversary",
                          "ok": adv_res.ok, "findings": adv_res.findings})
    adv_findings = _own(adv_res.findings, check.id, allowed) if adv_res.ok else []

    adj = adjudicate(agg, adv_findings)
    passed = all(v["passed"] for v in adj.values()) if adj else True
    needs_human = any(v["needs_human"] for v in adj.values())
    problem_ids = [cid for cid, v in adj.items() if (not v["passed"]) or v["needs_human"]]
    evidence = []
    for v in agg.values():
        evidence.extend(v.get("evidence") or [])
    detail = "" if passed and not needs_human else "criteria needing attention: " + ", ".join(problem_ids)
    return CheckResult(id=check.id, name=check.name, stage=Stage.SEMANTIC, severity=check.severity,
                       passed=passed, needs_human=needs_human, evidence=evidence[:20], detail=detail)


def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None):
    return [run_check(c, bundle, client, ctx, k=k, votes_log=votes_log) for c in checks]
