from __future__ import annotations
from dataclasses import dataclass, field
from autoqc.model import Severity


@dataclass
class SemanticJudgment:
    passed: bool
    evidence: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class Unit:
    key: str
    payload: dict


class SemanticCheck:
    id: str = ""
    name: str = ""
    severity: Severity = Severity.REJECT

    def units(self, bundle) -> list[Unit]:
        raise NotImplementedError

    def proposer_messages(self, bundle, unit: Unit) -> list[dict]:
        raise NotImplementedError

    def adversary_messages(self, bundle, unit: Unit, agg_passed: bool) -> list[dict]:
        raise NotImplementedError

    def parse(self, raw: dict) -> SemanticJudgment:
        raise NotImplementedError
