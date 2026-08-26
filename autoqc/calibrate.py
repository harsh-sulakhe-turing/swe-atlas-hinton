from __future__ import annotations
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from autoqc.seed import seed_bad_negative, seed_wildcard


@dataclass
class Case:
    name: str
    bundle_dir: Path
    expected_flags: dict = field(default_factory=dict)  # check_id -> set(criterion_ids)

    @property
    def expected_not_sound(self) -> bool:
        return bool(self.expected_flags)


def _copy_bundle(base: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(base, dest)
    return dest


def _seed_into(bundle_dir: Path, seed_fn):
    rf = bundle_dir / "tests" / "rubrics.json"
    items = json.loads(rf.read_text())
    mutated, changed = seed_fn(items)
    rf.write_text(json.dumps(mutated, indent=2))
    return changed


def build_corpus(base_bundle_dir, work_dir) -> list[Case]:
    base = Path(base_bundle_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    clean = _copy_bundle(base, work / "clean")
    q07 = _copy_bundle(base, work / "q07_bad")
    q07_id = _seed_into(q07, seed_bad_negative)
    q03 = _copy_bundle(base, work / "q03_bad")
    q03_id = _seed_into(q03, seed_wildcard)

    return [
        Case("clean", clean, {}),
        Case("q07_bad", q07, {"Q07": {q07_id}} if q07_id else {"Q07": set()}),
        Case("q03_bad", q03, {"Q03": {q03_id}} if q03_id else {"Q03": set()}),
    ]
