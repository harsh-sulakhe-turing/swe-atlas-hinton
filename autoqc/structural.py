from __future__ import annotations
import re
from autoqc.bundle import Bundle, REQUIRED_FILES
from autoqc.model import CheckResult, Stage, Severity

ID_RE = re.compile(r"^[0-9a-f]{32}$")
TITLE_RE = re.compile(r"^\s*([12])\.(\d+)\s*:")
VALID_TYPES = {"positive hli verifier", "negative hli verifier"}


def _ok(id, name, sev, detail="", evidence=None):
    return CheckResult(id=id, name=name, stage=Stage.STRUCTURAL, severity=sev,
                       passed=True, evidence=evidence or [], detail=detail)


def _fail(id, name, sev, detail, evidence=None):
    return CheckResult(id=id, name=name, stage=Stage.STRUCTURAL, severity=sev,
                       passed=False, evidence=evidence or [], detail=detail)


def _items(bundle: Bundle) -> list[dict]:
    return bundle.rubrics if isinstance(bundle.rubrics, list) else []


def _s01(b: Bundle) -> CheckResult:
    n, s = "Parses as JSON array", Severity.REJECT
    if b.rubrics_error is not None:
        return _fail("S01", n, s, f"rubrics.json does not parse: {b.rubrics_error}")
    if not isinstance(b.rubrics, list) or len(b.rubrics) == 0:
        return _fail("S01", n, s, "rubrics.json must be a non-empty JSON array")
    return _ok("S01", n, s)


def _s02(b: Bundle) -> CheckResult:
    n, s = "Item shape", Severity.REJECT
    for i, it in enumerate(_items(b)):
        if not isinstance(it, dict):
            return _fail("S02", n, s, f"item {i} is not an object")
        if "id" not in it or "title" not in it:
            return _fail("S02", n, s, f"item {i} missing id/title")
        ann = it.get("annotations")
        if not isinstance(ann, dict) or "type" not in ann or "importance" not in ann:
            return _fail("S02", n, s, f"item {i} missing annotations.type/importance")
    return _ok("S02", n, s)


def _s03(b: Bundle) -> CheckResult:
    n, s = "ID format & uniqueness", Severity.REJECT
    seen = set()
    for i, it in enumerate(_items(b)):
        iid = it.get("id") if isinstance(it, dict) else None
        if not isinstance(iid, str) or not ID_RE.match(iid):
            return _fail("S03", n, s, f"item {i} id {iid!r} is not 32 lowercase hex")
        if iid in seen:
            return _fail("S03", n, s, f"duplicate id {iid}")
        seen.add(iid)
    return _ok("S03", n, s)


def _s04(b: Bundle) -> CheckResult:
    n, s = "Type/number consistency", Severity.REJECT
    for it in _items(b):
        if not isinstance(it, dict):
            continue
        m = TITLE_RE.match(str(it.get("title", "")))
        if not m:
            return _fail("S04", n, s, f"title not numbered N.x: {it.get('title')!r}")
        num, typ = m.group(1), str(it.get("annotations", {}).get("type", ""))
        if num == "1" and "positive" not in typ:
            return _fail("S04", n, s, f"1.x must be positive: {it.get('title')!r}")
        if num == "2" and "negative" not in typ:
            return _fail("S04", n, s, f"2.x must be negative: {it.get('title')!r}")
    return _ok("S04", n, s)


def _s05(b: Bundle) -> CheckResult:
    n, s = "Has a positive", Severity.REJECT
    for it in _items(b):
        if isinstance(it, dict) and "positive" in str(it.get("annotations", {}).get("type", "")):
            return _ok("S05", n, s)
    return _fail("S05", n, s, "no positive (1.x) criterion present")


def _s06(b: Bundle) -> CheckResult:
    n, s = "Bundle completeness", Severity.REJECT
    missing = [f for f in REQUIRED_FILES if not b.files_present.get(f)]
    if missing:
        return _fail("S06", n, s, f"missing files: {', '.join(missing)}")
    if not b.repository:
        return _fail("S06", n, s, "task.toml missing [metadata].repository")
    if not (isinstance(b.base_commit, str) and len(b.base_commit) == 40):
        return _fail("S06", n, s, "task.toml base_commit must be 40 chars")
    return _ok("S06", n, s)


def _s07(b: Bundle) -> CheckResult:
    n, s = "Type vocabulary", Severity.WARN
    for it in _items(b):
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations", {})
        if ann.get("type") not in VALID_TYPES:
            return _fail("S07", n, s, f"unexpected type {ann.get('type')!r}")
        if ann.get("importance") != "must have":
            return _fail("S07", n, s, f"unexpected importance {ann.get('importance')!r}")
    return _ok("S07", n, s)


def _s08(b: Bundle) -> CheckResult:
    n, s = "Sequential numbering", Severity.WARN
    nums = {"1": [], "2": []}
    for it in _items(b):
        if not isinstance(it, dict):
            continue
        m = TITLE_RE.match(str(it.get("title", "")))
        if m:
            nums[m.group(1)].append(int(m.group(2)))
    for prefix, seq in nums.items():
        if not seq:
            continue
        expected = list(range(1, len(seq) + 1))
        if sorted(seq) != expected:
            return _fail("S08", n, s,
                         f"{prefix}.x numbering not sequential: got {sorted(seq)}")
    return _ok("S08", n, s)


def run_structural(bundle: Bundle) -> list[CheckResult]:
    return [_s01(bundle), _s02(bundle), _s03(bundle),
            _s04(bundle), _s05(bundle), _s06(bundle),
            _s07(bundle), _s08(bundle)]
