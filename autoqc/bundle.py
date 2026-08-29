from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

REQUIRED_FILES = [
    "tests/prompt.txt",
    "tests/rubrics.json",
    "solution/answer.txt",
    "task.toml",
    "environment/Dockerfile",
]


@dataclass
class Bundle:
    root: Path
    rubrics_raw: str | None
    rubrics: object | None
    rubrics_error: str | None
    prompt: str | None
    answer: str | None
    instruction: str | None
    task_toml: dict | None
    repository: str | None
    base_commit: str | None
    files_present: dict[str, bool]


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text()
    except (OSError, UnicodeDecodeError):
        return None


def load_bundle(root: Path) -> Bundle:
    root = Path(root)
    files_present = {rel: (root / rel).is_file() for rel in REQUIRED_FILES}

    rubrics_raw = _read_text(root / "tests/rubrics.json")
    rubrics, rubrics_error = None, None
    if rubrics_raw is None:
        if (root / "tests/rubrics.json").exists():
            rubrics_error = "tests/rubrics.json is unreadable"
        else:
            rubrics_error = "tests/rubrics.json is missing"
    else:
        try:
            rubrics = json.loads(rubrics_raw)
        except json.JSONDecodeError as e:
            rubrics_error = f"invalid JSON: {e}"

    task_toml, repository, base_commit = None, None, None
    raw_toml = _read_text(root / "task.toml")
    if raw_toml is not None:
        try:
            task_toml = tomllib.loads(raw_toml)
            meta = task_toml.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            repository = meta.get("repository")
            if not isinstance(repository, str):
                repository = None
            base_commit = meta.get("base_commit")
            if not isinstance(base_commit, str):
                base_commit = None
        except tomllib.TOMLDecodeError:
            task_toml = None

    return Bundle(
        root=root,
        rubrics_raw=rubrics_raw,
        rubrics=rubrics,
        rubrics_error=rubrics_error,
        prompt=_read_text(root / "tests/prompt.txt"),
        answer=_read_text(root / "solution/answer.txt"),
        instruction=_read_text(root / "instruction.md"),
        task_toml=task_toml,
        repository=repository,
        base_commit=base_commit,
        files_present=files_present,
    )
