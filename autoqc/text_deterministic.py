from __future__ import annotations
from pathlib import Path
from autoqc.model import CheckResult, Stage, Severity

# Verbatim fragments of the authoring template left in un-rendered instruction.md,
# taken from the Batch-1 placeholder bundles. A case-insensitive substring match is
# an exact-marker test, not a heuristic quality judgment.
PLACEHOLDER_MARKERS = (
    "describe the developer's realistic, multi-part question here",
    "without telegraphing the measured result",
)

def check_p01(bundle) -> CheckResult:
    text = getattr(bundle, "instruction", None)
    n, s = "Instruction rendered, not placeholder", Severity.REJECT
    if not isinstance(text, str) or not text.strip():
        return CheckResult(id="P01", name=n, stage=Stage.STRUCTURAL, severity=s,
                           passed=False, detail="instruction.md is missing or empty")
    low = text.lower()
    hit = next((m for m in PLACEHOLDER_MARKERS if m in low), None)
    if hit:
        return CheckResult(id="P01", name=n, stage=Stage.STRUCTURAL, severity=s,
                           passed=False,
                           detail=f"instruction.md still contains authoring placeholder text: {hit!r}",
                           evidence=[f"marker matched: {hit!r}"])
    return CheckResult(id="P01", name=n, stage=Stage.STRUCTURAL, severity=s, passed=True)

def check_h01(bundle) -> CheckResult:
    n, s = "No committed cache artifacts", Severity.WARN
    root = Path(getattr(bundle, "root", "."))
    hits = []
    try:
        hits = [str(p.relative_to(root)) for p in root.rglob("*")
                if p.name == "__pycache__" or p.suffix == ".pyc"]
    except OSError:
        pass
    if hits:
        return CheckResult(id="H01", name=n, stage=Stage.STRUCTURAL, severity=s,
                           passed=False, detail=f"cache artifacts committed: {', '.join(sorted(hits)[:5])}",
                           evidence=sorted(hits)[:5])
    return CheckResult(id="H01", name=n, stage=Stage.STRUCTURAL, severity=s, passed=True)

TEXT_DETERMINISTIC_CHECKS = [check_p01, check_h01]
