from __future__ import annotations

import re

from autoqc.model import CheckResult, Stage, Severity
from autoqc.agent.runner import run_agent
from autoqc.agent.checks import (SEMANTIC_CHECKS, proposer_role, adversary_role,
                                 proposer_context, adversary_context,
                                 Q06, factual_role, factual_context)
from autoqc.agent.tools import validate_findings, AgentContext
from autoqc.agent.deterministic import DETERMINISTIC_CHECKS
from autoqc.agent.container import ContainerSession, docker_available, ContainerError


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
    allowed = {"rubric"} if getattr(check, "unit_mode", "criterion") == "rubric" else {c["id"] for c in criteria}

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


def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None, factual=True):
    results = [run_check(c, bundle, client, ctx, k=k, votes_log=votes_log) for c in checks]
    results += [fn(bundle) for fn in DETERMINISTIC_CHECKS]
    if factual:
        results.append(run_factual_stage(bundle, client, votes_log=votes_log))
    return results


def _files(evidence) -> set[str]:
    """Path prefixes cited in evidence. Scans the whole string for a
    path:line-shaped token (evidence only has to CONTAIN one, not start
    with one); falls back to the pre-':' first token if none is found.
    Used to require same-file agreement."""
    out = set()
    for e in evidence or []:
        s = str(e).strip()
        if not s:
            continue
        m = re.search(r'([^\s:]+):\d+', s)
        if m:
            out.add(m.group(1))
        else:
            head = s.split()[0]
            out.add(head.split(":", 1)[0])
    return out


def _round_map(findings, allowed_ids):
    """First (passed, files) per criterion in a round's own Q06 findings."""
    m = {}
    for f in _own(findings, "Q06", allowed_ids):
        cid = f.get("criterion_id")
        if cid in allowed_ids and cid not in m:
            m[cid] = (bool(f.get("passed")), _files(f.get("evidence")))
    return m


def adjudicate_factual(r1_findings, r2_findings, allowed_ids) -> dict[str, dict]:
    m1 = _round_map(r1_findings, allowed_ids)
    m2 = _round_map(r2_findings, allowed_ids)
    out = {}
    for cid in allowed_ids:
        v1, v2 = m1.get(cid), m2.get(cid)
        if v1 is None or v2 is None:
            out[cid] = {"passed": False, "needs_human": True}
            continue
        p1, f1 = v1
        p2, f2 = v2
        if p1 != p2:
            out[cid] = {"passed": False, "needs_human": True}
        elif p1:  # both pass
            out[cid] = {"passed": True, "needs_human": False}
        else:      # both reject: require same-file overlap
            same = bool(f1 & f2)
            out[cid] = {"passed": False, "needs_human": not same}
    return out


def run_factual(bundle, client, ctx, votes_log=None) -> CheckResult:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    criteria = Q06.scope(items)
    if not criteria:
        return CheckResult(id="Q06", name=Q06.name, stage=Stage.FACTUAL,
                           severity=Severity.REJECT, passed=True)
    allowed = {c["id"] for c in criteria}
    p_ctx = factual_context(bundle, criteria)
    rounds = []
    for _ in range(2):
        res = run_agent(factual_role(), p_ctx, client, ctx)
        if votes_log is not None:
            votes_log.append({"check": "Q06", "role": "factual",
                              "ok": res.ok, "findings": res.findings})
        rounds.append(res.findings if res.ok else [])

    adj = adjudicate_factual(rounds[0], rounds[1], allowed)
    passed = all(v["passed"] for v in adj.values()) if adj else True
    needs_human = any(v["needs_human"] for v in adj.values())
    problems = [cid for cid, v in adj.items() if (not v["passed"]) or v["needs_human"]]
    evidence = []
    for fs in rounds:
        for f in _own(fs, "Q06", allowed):
            evidence.extend(f.get("evidence") or [])
    detail = "" if passed and not needs_human else "criteria needing attention: " + ", ".join(problems)
    return CheckResult(id="Q06", name=Q06.name, stage=Stage.FACTUAL, severity=Severity.REJECT,
                       passed=passed, needs_human=needs_human, evidence=evidence[:20], detail=detail)


def _q06_needs_human(reason: str) -> CheckResult:
    return CheckResult(id="Q06", name=Q06.name, stage=Stage.FACTUAL, severity=Severity.REJECT,
                       passed=False, needs_human=True, detail=f"Q06 not run: {reason}")


def run_factual_stage(bundle, client, votes_log=None, limits=None,
                      docker=docker_available) -> CheckResult:
    if not getattr(bundle, "files_present", {}).get("environment/Dockerfile", True):
        return _q06_needs_human("no environment/Dockerfile in bundle")
    if not docker():
        return _q06_needs_human("Docker is not available")
    try:
        session = ContainerSession(bundle, limits=limits)
        session.ensure_image()
        session.start()
    except ContainerError as e:
        return _q06_needs_human(str(e))
    except Exception as e:  # defensive: never crash the run
        return _q06_needs_human(f"container setup error: {e}")
    try:
        ctx = AgentContext(bundle_dir=bundle.root, container=session)
        return run_factual(bundle, client, ctx, votes_log=votes_log)
    except Exception as e:
        return _q06_needs_human(f"factual pass error: {e}")
    finally:
        session.stop()
