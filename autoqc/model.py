from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    REJECT = "reject"
    WARN = "warn"


class Stage(Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    FACTUAL = "factual"


class Verdict(Enum):
    NOT_SOUND = "not_sound"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    SOUND = "sound"


@dataclass
class CheckResult:
    id: str
    name: str
    stage: Stage
    severity: Severity
    passed: bool
    needs_human: bool = False
    evidence: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class ReviewRecord:
    task_name: str
    guideline_version: str
    results: list[CheckResult]
    verdict: Verdict
