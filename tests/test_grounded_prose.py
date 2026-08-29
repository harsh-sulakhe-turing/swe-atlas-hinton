import types
from autoqc.agent.checks import (grounded_prompt_role, grounded_answer_role,
                                 grounded_prompt_context, grounded_answer_context)

def _b():
    return types.SimpleNamespace(prompt="Why does the aiohttp pool reopen sockets?",
                                 answer="I traced conn.close() in the cancel handler ...",
                                 repository="aio-libs/aiohttp", base_commit="a" * 40)

def test_grounded_roles_have_run_bash():
    names = {t.name for t in grounded_prompt_role().tools}
    assert "run_bash" in names and "submit_findings" in names

def test_grounded_prompt_context_mentions_p04_and_testbed():
    c = grounded_prompt_context(_b())
    assert "P04" in c and "/testbed" in c and 'criterion_id="prompt"' in c

def test_grounded_answer_context_mentions_a06():
    c = grounded_answer_context(_b())
    assert "A06" in c and 'criterion_id="answer"' in c and "trajectory" in c.lower()
