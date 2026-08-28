import types
from autoqc.model import Stage, Severity
from autoqc.text_deterministic import check_p01

def _b(instruction):
    return types.SimpleNamespace(instruction=instruction, root=".")

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
