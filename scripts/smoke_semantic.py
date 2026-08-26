"""Live smoke: run the full AutoQC pipeline (structural + semantic Q07/Q03) against the
real gateway on one bundle. Reads .env, prints verdict + per-criterion semantic outcomes
and how many agent passes ran. Makes real model calls — operator-run, not a unit test.

Usage: python3 scripts/smoke_semantic.py <bundle_dir> [K]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
from autoqc.bundle import load_bundle
from autoqc.structural import run_structural
from autoqc.agent.engine import run_semantic
from autoqc.agent.tools import AgentContext
from autoqc.verdict import compute_verdict
from autoqc.llm import GatewayLLMClient

# load .env into the environment (do not print secrets)
for line in open(".env"):
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

bundle_dir = Path(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 3

bundle = load_bundle(bundle_dir)
client = GatewayLLMClient()
assert client.available(), ".env missing EVAL_API_KEY / EVAL_BASE_URL"
print(f"model={client.model}  bundle={bundle_dir.name}  K={k}")
rubric = bundle.rubrics if isinstance(bundle.rubrics, list) else []
print(f"rubric: {len(rubric)} criteria")

results = run_structural(bundle)
votes = []
results += run_semantic(bundle, client, AgentContext(bundle_dir=bundle_dir), k=k, votes_log=votes)
verdict = compute_verdict(results)

print("\n--- semantic checks ---")
for r in results:
    if r.stage.value == "semantic":
        print(f"{r.id} {r.name}: passed={r.passed}  needs_human={r.needs_human}")
        if r.detail:
            print(f"    {r.detail}")
        for e in r.evidence[:3]:
            print(f"    evidence: {e[:120]}")

print("\n--- agent passes (per-vote log) ---")
by = {}
for v in votes:
    key = (v["check"], v["role"])
    by.setdefault(key, {"n": 0, "ok": 0, "findings": 0})
    by[key]["n"] += 1
    by[key]["ok"] += 1 if v["ok"] else 0
    by[key]["findings"] += len(v.get("findings") or [])
for (check, role), s in sorted(by.items()):
    print(f"  {check} {role}: {s['n']} passes, {s['ok']} ok, {s['findings']} findings total")

print(f"\nSTRUCTURAL+SEMANTIC VERDICT: {verdict.value}")
