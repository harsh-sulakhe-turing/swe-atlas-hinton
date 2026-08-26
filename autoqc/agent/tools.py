from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CHECK_IDS = {f"Q{n:02d}" for n in range(1, 13)}
ALLOWED_READ = {"tests/prompt.txt", "tests/rubrics.json", "solution/answer.txt", "task.toml"}


@dataclass
class AgentContext:
    bundle_dir: Path


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable | None = None
    terminal: bool = False

    def schema(self) -> dict:
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


def _read_bundle_file(args: dict, ctx: AgentContext) -> str:
    path = str(args.get("path", ""))
    if path not in ALLOWED_READ:
        return f"error: '{path}' is not allowed (readable: {sorted(ALLOWED_READ)})"
    p = Path(ctx.bundle_dir) / path
    try:
        return p.read_text()
    except FileNotFoundError:
        return f"error: '{path}' not found in bundle"
    except OSError as e:
        return f"error: could not read '{path}': {e}"
    except (UnicodeDecodeError, ValueError) as e:
        return f"error: could not decode '{path}': {e}"


def _list_dir(args: dict, ctx: AgentContext) -> str:
    rel = str(args.get("path", "."))
    root = Path(ctx.bundle_dir).resolve()
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        return f"error: '{rel}' is outside the bundle"
    try:
        return "\n".join(sorted(x.name for x in p.iterdir()))
    except OSError as e:
        return f"error: could not list '{rel}': {e}"


read_bundle_file = Tool(
    name="read_bundle_file",
    description="Read one file from the task bundle (prompt, rubric, answer, or task.toml).",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    run=_read_bundle_file)

list_dir = Tool(
    name="list_dir",
    description="List the entries of a directory inside the task bundle.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    run=_list_dir)

SUBMIT_FINDINGS = Tool(
    name="submit_findings",
    description="Submit the final structured verdicts and finish. Call this exactly once.",
    parameters={"type": "object", "properties": {"findings": {"type": "array", "items": {
        "type": "object",
        "required": ["check_id", "criterion_id", "passed", "evidence"],
        "properties": {
            "check_id": {"type": "string", "enum": sorted(CHECK_IDS)},
            "criterion_id": {"type": "string"},
            "passed": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "reason": {"type": "string"}}}}}, "required": ["findings"]},
    run=None, terminal=True)


def default_tools() -> list[Tool]:
    return [read_bundle_file, list_dir, SUBMIT_FINDINGS]


def validate_findings(findings, allowed_criterion_ids: set[str]):
    """Return (valid, problems). A finding is valid iff check_id is a known check,
    criterion_id is a known rubric id (or 'rubric'), passed is bool, evidence non-empty."""
    valid, problems = [], []
    for i, f in enumerate(findings if isinstance(findings, list) else []):
        if not isinstance(f, dict):
            problems.append(f"finding {i}: not an object"); continue
        if f.get("check_id") not in CHECK_IDS:
            problems.append(f"finding {i}: bad check_id {f.get('check_id')!r}"); continue
        cid = f.get("criterion_id")
        if cid != "rubric" and cid not in allowed_criterion_ids:
            problems.append(f"finding {i}: unknown criterion_id {cid!r}"); continue
        if not isinstance(f.get("passed"), bool):
            problems.append(f"finding {i}: passed not a bool"); continue
        ev = f.get("evidence")
        if not (isinstance(ev, list) and ev and all(isinstance(x, str) and x.strip() for x in ev)):
            problems.append(f"finding {i}: evidence must be a non-empty list of non-empty strings"); continue
        valid.append(f)
    return valid, problems
