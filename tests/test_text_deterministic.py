import types
from pathlib import Path
from autoqc.model import Stage, Severity
from autoqc.text_deterministic import check_p01, check_h01, TEXT_DETERMINISTIC_CHECKS

def _b(instruction):
    return types.SimpleNamespace(instruction=instruction, root=".")

def _broot(tmp_path):
    return types.SimpleNamespace(instruction="ok", root=tmp_path)

def test_p01_rejects_placeholder_marker():
    r = check_p01(_b("...\n<question>\nDescribe the developer's realistic, multi-part "
                     "question here without telegraphing the measured result.\n</question>"))
    assert r.id == "P01" and r.stage is Stage.STRUCTURAL and r.severity is Severity.REJECT
    assert r.passed is False and "placeholder" in r.detail.lower()

def test_p01_rejects_missing_instruction():
    r = check_p01(_b(None))
    assert r.passed is False

def test_p01_passes_rendered_prompt():
    r = check_p01(_b("I have a service that POSTs file-like bodies with aiohttp and the "
                     "connection pool is misbehaving. Why does a zero-length file body ..."))
    assert r.passed is True

def test_h01_warns_on_pycache(tmp_path):
    (tmp_path / "tests" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tests" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    r = check_h01(_broot(tmp_path))
    assert r.id == "H01" and r.severity is Severity.WARN and r.passed is False
    assert "__pycache__" in r.detail

def test_h01_passes_clean_bundle(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "prompt.txt").write_text("q")
    assert check_h01(_broot(tmp_path)).passed is True

def test_text_deterministic_registry():
    assert TEXT_DETERMINISTIC_CHECKS == [check_p01, check_h01]
