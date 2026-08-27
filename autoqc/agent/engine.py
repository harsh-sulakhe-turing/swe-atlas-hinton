from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from autoqc.model import CheckResult, Stage, Severity
from autoqc.agent.runner import run_agent
from autoqc.agent.checks import (SEMANTIC_CHECKS, proposer_role, adversary_role,
                                 proposer_context, adversary_context,
                                 Q06, factual_role, factual_context)
from autoqc.agent.tools import validate_findings, AgentContext
from autoqc.agent.deterministic import DETERMINISTIC_CHECKS
from autoqc.agent.container import ContainerSession, docker_available, ContainerError


def _engine_workers(default: int = 8) -> int:
    try:
        return int(os.environ.get("AUTOQC_ENGINE_WORKERS", str(default)))
    except (TypeError, ValueError):
        return default


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


TEXT_MAX_TURNS = 3


def _factual_max_turns(default: int = 40) -> int:
    try:
        return int(os.environ.get("AUTOQC_FACTUAL_MAX_TURNS", str(default)))
    except (TypeError, ValueError):
        return default


# The container agent is exploratory (reads source + runs repros across many
# turns before it can submit), so it needs a far larger turn budget than the
# single-turn text checks. The default 12 (inherited by omission) starved it and
# degraded real verifications to needs_human. Env-overridable for calibration.
FACTUAL_MAX_TURNS = _factual_max_turns()


def _empty_scope_result(check) -> CheckResult:
    return CheckResult(id=check.id, name=check.name, stage=Stage.SEMANTIC,
                       severity=check.severity, passed=True)


def _check_prep(check, bundle):
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    criteria = check.scope(items)
    allowed = ({"rubric"} if getattr(check, "unit_mode", "criterion") == "rubric"
               else {c["id"] for c in criteria})
    return criteria, allowed


def proposer_pass(check, bundle, client, ctx, criteria, allowed):
    res = run_agent(proposer_role(), proposer_context(bundle, check, criteria),
                    client, ctx, max_turns=TEXT_MAX_TURNS)
    log = {"check": check.id, "role": "proposer", "ok": res.ok, "findings": res.findings}
    return (_own(res.findings, check.id, allowed) if res.ok else []), log


def adversary_pass(check, bundle, client, ctx, criteria, allowed, agg):
    res = run_agent(adversary_role(), adversary_context(bundle, check, criteria, agg),
                    client, ctx, max_turns=TEXT_MAX_TURNS)
    log = {"check": check.id, "role": "adversary", "ok": res.ok, "findings": res.findings}
    return (_own(res.findings, check.id, allowed) if res.ok else []), log


def finalize_check(check, agg, adv_findings) -> CheckResult:
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


def run_check(check, bundle, client, ctx, k=3, votes_log=None) -> CheckResult:
    criteria, allowed = _check_prep(check, bundle)
    if not criteria:
        return _empty_scope_result(check)
    finding_sets = []
    for _ in range(k):
        own, log = proposer_pass(check, bundle, client, ctx, criteria, allowed)
        if votes_log is not None:
            votes_log.append(log)
        finding_sets.append(own)
    agg = aggregate(finding_sets, allowed)
    adv_own, adv_log = adversary_pass(check, bundle, client, ctx, criteria, allowed, agg)
    if votes_log is not None:
        votes_log.append(adv_log)
    return finalize_check(check, agg, adv_own)


def run_checks_parallel(checks, bundle, client, ctx, k, votes_log=None) -> list:
    """Run every text check over one bounded thread pool. All proposer passes are
    submitted up front; each check's adversary is launched when its k proposers
    finish. Verdicts are identical to serial run_check (aggregate is order-free).
    Bookkeeping runs only on this (main) thread; pool workers are pure passes."""
    w = _engine_workers()
    preps, results_by_id = {}, {}
    for check in checks:
        criteria, allowed = _check_prep(check, bundle)
        if not criteria:
            results_by_id[check.id] = _empty_scope_result(check)
        else:
            preps[check.id] = (check, criteria, allowed)

    prop_logs = {cid: [] for cid in preps}     # cid -> [log dict] (completion order)
    prop_sets = {cid: [] for cid in preps}     # cid -> [own findings]
    remaining = {cid: k for cid in preps}
    aggs, adv_logs = {}, {}

    with ThreadPoolExecutor(max_workers=max(1, w)) as pool:
        fut_meta, pending = {}, set()
        for cid, (check, criteria, allowed) in preps.items():
            for _ in range(k):
                fut = pool.submit(proposer_pass, check, bundle, client, ctx, criteria, allowed)
                fut_meta[fut] = ("proposer", cid)
                pending.add(fut)
            if remaining[cid] <= 0:
                # k<=0: no proposer completion will ever fire the remaining[cid]==0
                # check below, so match serial run_check by aggregating zero votes
                # and launching the adversary right away.
                agg = aggregate(prop_sets[cid], allowed)
                aggs[cid] = agg
                afut = pool.submit(adversary_pass, check, bundle, client, ctx,
                                   criteria, allowed, agg)
                fut_meta[afut] = ("adversary", cid)
                pending.add(afut)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                kind, cid = fut_meta.pop(fut)
                check, criteria, allowed = preps[cid]
                if kind == "proposer":
                    try:
                        own, log = fut.result()
                    except Exception as e:  # never sink the run
                        own, log = [], {"check": cid, "role": "proposer", "ok": False,
                                        "findings": [], "error": repr(e)}
                    prop_sets[cid].append(own)
                    prop_logs[cid].append(log)
                    remaining[cid] -= 1
                    if remaining[cid] == 0:
                        agg = aggregate(prop_sets[cid], allowed)
                        aggs[cid] = agg
                        afut = pool.submit(adversary_pass, check, bundle, client, ctx,
                                           criteria, allowed, agg)
                        fut_meta[afut] = ("adversary", cid)
                        pending.add(afut)
                else:  # adversary
                    try:
                        adv_own, adv_log = fut.result()
                    except Exception as e:
                        adv_own, adv_log = [], {"check": cid, "role": "adversary", "ok": False,
                                               "findings": [], "error": repr(e)}
                    adv_logs[cid] = adv_log
                    results_by_id[cid] = finalize_check(check, aggs[cid], adv_own)

    if votes_log is not None:  # deterministic order: check order, proposers then adversary
        # (within a single check, the proposer entries themselves are in
        # completion order, not submission order -- this doesn't affect
        # verdicts since aggregate() is order-independent over its votes)
        for check in checks:
            cid = check.id
            if cid in preps:
                votes_log.extend(prop_logs[cid])
                if cid in adv_logs:
                    votes_log.append(adv_logs[cid])

    return [results_by_id[check.id] for check in checks]


def run_semantic(bundle, client, ctx, checks=SEMANTIC_CHECKS, k=3, votes_log=None, factual=True):
    results = run_checks_parallel(checks, bundle, client, ctx, k, votes_log=votes_log)
    results += [fn(bundle) for fn in DETERMINISTIC_CHECKS]
    if factual:
        results.append(run_factual_stage(bundle, client, votes_log=votes_log))
    return results


def _files(evidence) -> set[str]:
    """Path prefixes cited in evidence. Scans the whole string for a
    path:line-shaped token (evidence only has to CONTAIN one, not start
    with one). Evidence with no such token contributes nothing -- there is
    no fallback to a first-whitespace-token guess, since that could make two
    citation-less rejects look like the same file by coincidence and turn
    into a false hard reject. Used to require same-file agreement."""
    out = set()
    for e in evidence or []:
        s = str(e).strip()
        if not s:
            continue
        m = re.search(r'([^\s:]+):\d+', s)
        if m:
            out.add(m.group(1))
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
        res = run_agent(factual_role(), p_ctx, client, ctx, max_turns=FACTUAL_MAX_TURNS)
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
