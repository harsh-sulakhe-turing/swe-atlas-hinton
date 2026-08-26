from __future__ import annotations


def aggregate(finding_sets, allowed_ids):
    out = {}
    n = len(finding_sets)
    for cid in allowed_ids:
        votes, evidence = [], []
        for fs in finding_sets:
            for f in fs:
                if f.get("criterion_id") == cid:
                    votes.append(bool(f.get("passed")))
                    evidence.extend(f.get("evidence") or [])
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
