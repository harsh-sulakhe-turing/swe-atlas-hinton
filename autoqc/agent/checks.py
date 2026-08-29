from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from autoqc.model import Severity
from autoqc.agent.runner import Role
from autoqc.agent.tools import factual_tools, text_tools


@dataclass
class SemanticCheck:
    id: str
    name: str
    severity: Severity
    scope: Callable          # (items) -> list[dict]
    guidance: str
    unit_mode: str = "criterion"
    role_kind: str = "rubric"


def _is_dict_crit(it):
    return isinstance(it, dict) and it.get("id")


def _negatives(items):
    return [it for it in (items or [])
            if _is_dict_crit(it) and "negative" in str(
                it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")]


def _all_criteria(items):
    return [it for it in (items or []) if _is_dict_crit(it)]


def _positives(items):
    return [it for it in (items or [])
            if _is_dict_crit(it) and "positive" in str(
                it.get("annotations", {}).get("type", "") if isinstance(it.get("annotations"), dict) else "")]


def _prompt_unit(items): return [{"id": "prompt"}]
def _answer_unit(items): return [{"id": "answer"}]
def _bundle_unit(items): return [{"id": "bundle"}]


Q07 = SemanticCheck(
    id="Q07", name="Negative score-flip semantics", severity=Severity.REJECT, scope=_negatives,
    guidance=("A NEGATIVE criterion must state the FALSE assertion whose PRESENCE in an answer "
              "should fail it (that is, phrased as 'Claims that <the incorrect behavior>' — an "
              "assertion the correct answer disproves). It VIOLATES this check if phrased as "
              "'Does not claim...', as a required omission, or as the correct behavior. "
              "passed=true means correctly phrased."))

Q03 = SemanticCheck(
    id="Q03", name="No wildcard / escape hatch", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this check if it can be satisfied by open-ended wording: a "
              "bare trailing '*', 'or similar', 'or other', 'and so on', or an e.g./such-as list "
              "whose items are NOT interchangeable full answers. IMPORTANT: an e.g./such as list is "
              "FINE and common when every listed item is an interchangeable way to satisfy the same "
              "requirement — do NOT flag interchangeable examples. passed=true means no escape hatch."))

Q01 = SemanticCheck(
    id="Q01", name="Atomicity", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this if it bundles TWO OR MORE independently gradable facts "
              "(each could be independently true or false and separately meaningful). A cause and "
              "its direct effect may stay together only when partial satisfaction would be "
              "meaningless. A coherent cluster answering one sub-question is acceptable. passed=true "
              "if the criterion tests one gradable proposition."))

Q02 = SemanticCheck(
    id="Q02", name="Binary judgeability", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this if a grader cannot decide it met/not-met from an answer's "
              "text alone: subjective quality words ('thorough', 'clearly', 'well', 'adequately', "
              "'good'), or an undefined completeness claim. Naming specific facts/values/paths is "
              "fine. passed=true if the criterion is objectively decidable."))

Q05 = SemanticCheck(
    id="Q05", name="No unrequested scope", severity=Severity.REJECT, scope=_positives,
    guidance=("A POSITIVE criterion VIOLATES this if it grades a fact the prompt neither requests "
              "nor requires as an indispensable causal link to answer it. Use the task prompt "
              "(shown above) to judge. passed=true if the criterion is requested by the prompt or is "
              "a necessary causal step toward the requested answer."))

Q04 = SemanticCheck(
    id="Q04", name="Prompt coverage", severity=Severity.REJECT, scope=_all_criteria, unit_mode="rubric",
    guidance=("The rubric VIOLATES this if any explicit obligation in the task prompt has NO positive "
              "criterion grading it (especially a central runtime result buried or absent). passed=true "
              "if every prompt obligation maps to at least one positive criterion. Evidence: name any "
              "uncovered obligation."))

Q08 = SemanticCheck(
    id="Q08", name="Discriminating negatives", severity=Severity.WARN, scope=_all_criteria, unit_mode="rubric",
    guidance=("Grading is all-or-nothing: a negative adds power only if NO positive already forces the "
              "correct version of the same fact. The rubric VIOLATES this (warn) if its negatives are "
              "mostly REDUNDANT inverses of positives. passed=true if at least one negative is "
              "discriminating — it catches a plausible wrong answer that no positive already forces. "
              "Evidence: cite a redundant negative, or the discriminating one."))

Q10 = SemanticCheck(
    id="Q10", name="Empirical result graded", severity=Severity.WARN, scope=_all_criteria, unit_mode="rubric",
    guidance=("If the task prompt requires running the software, the rubric VIOLATES this (warn) if NO "
              "positive grades the SPECIFIC observed result (a value, comparison, state transition, "
              "generated artifact, or error). A criterion that merely says the answer 'ran' or "
              "'reported empirical evidence' is insufficient. passed=true if a specific empirical "
              "result is graded, OR the prompt does not require running anything."))

Q11 = SemanticCheck(
    id="Q11", name="Not lookup-dominated", severity=Severity.WARN, scope=_all_criteria, unit_mode="rubric",
    guidance=("The rubric VIOLATES this (warn) if it is mostly trivia lookups (ports, constants, default "
              "values, file lists, single line numbers) with little causal or synthesis content. "
              "passed=true if the criteria as a whole require meaningful reasoning, not just lookups."))

P02 = SemanticCheck(
    id="P02", name="Single coherent goal", severity=Severity.REJECT, scope=_prompt_unit,
    unit_mode="prompt", role_kind="prose",
    guidance=("The prompt VIOLATES this if it bundles two or more independent tasks/goals "
              "(each separately deliverable) into one request. A single question with several "
              "closely-related sub-parts that build to one answer is FINE. passed=true if the "
              "prompt pursues one coherent goal."))

P03 = SemanticCheck(
    id="P03", name="Natural conversational request", severity=Severity.REJECT, scope=_prompt_unit,
    unit_mode="prompt", role_kind="prose",
    guidance=("The prompt VIOLATES this if it reads as a rigid enumerated checklist / numbered "
              "instruction list rather than a natural developer question, OR if it telegraphs the "
              "measured result the answer is supposed to discover. Prose with a few inline "
              "sub-questions is FINE. passed=true if it reads as a natural, non-spoiling request."))

A01 = SemanticCheck(
    id="A01", name="Investigation-first opening", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it OPENS with the conclusion / a 'short answer' lede "
              "before any acknowledgment of investigation. A brief summary is fine AFTER an opening "
              "that acknowledges exploring the codebase. passed=true if it opens investigation-first."))

A02 = SemanticCheck(
    id="A02", name="Continuous narrative", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it is organized under bold or numbered SECTION HEADERS "
              "(e.g. '**Root cause**', '1. Summary') rather than continuous prose that interleaves "
              "reasoning and evidence. Inline code blocks and their output are fine. passed=true if "
              "it reads as continuous narrative."))

A03 = SemanticCheck(
    id="A03", name="First-person voice", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it is NOT written in sustained first-person ('I traced', "
              "'I confirmed'). An impersonal report voice ('The system does X') throughout violates "
              "it. passed=true if first-person narration is sustained."))

A04 = SemanticCheck(
    id="A04", name="Evidence shown inline", severity=Severity.REJECT, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if it CLAIMS an empirical result (ran a command, observed a "
              "value/state/error) but does NOT show the actual command AND its output inline. Pure "
              "code-reading conclusions with file:line citations do not need command output. "
              "passed=true if every claimed run shows command + output."))

A05 = SemanticCheck(
    id="A05", name="Bash-only method", severity=Severity.WARN, scope=_answer_unit,
    unit_mode="answer", role_kind="prose",
    guidance=("The answer VIOLATES this if its investigation METHOD is creating/committing a "
              "script or source file as the deliverable, or modifying the repo, rather than "
              "read-only bash exploration (grep/cat/find/git plus temporary, cleaned-up probes). "
              "passed=true if the method is read-only bash investigation."))

AL01 = SemanticCheck(
    id="AL01", name="Files describe one task", severity=Severity.WARN, scope=_bundle_unit,
    unit_mode="bundle", role_kind="prose",
    guidance=("VIOLATES this if instruction.md, tests/prompt.txt, and solution/answer.txt do not "
              "all describe the SAME task (different subject, repo, or question). Minor wording "
              "differences are fine. passed=true if all three correspond to one task."))

Q13 = SemanticCheck(
    id="Q13", name="Rubric verifies exploration", severity=Severity.REJECT, scope=_all_criteria,
    unit_mode="rubric",
    guidance=("The rubric VIOLATES this if NO must-have criterion verifies that the model actually "
              "EXPLORED the codebase — i.e. grades a repo-derived fact, path, mechanism, or observed "
              "runtime result that could only be produced by investigating the code, not by general "
              "knowledge. passed=true if at least one criterion forces demonstrated exploration."))

# All semantic-text checks run through the shared proposer(K=3)+adversary ensemble
# in autoqc/agent/engine.py. Phase 1 rubric-criterion checks: Q07, Q03, Q01, Q02,
# Q05, Q04, Q08, Q10, Q11. Phase 1.5 whole-document/rubric checks added on top:
# P02, P03 (prompt), A01-A05 (answer), AL01 (cross-file alignment), Q13 (rubric
# exploration coverage). Q06, P04, and A06 are NOT here — they are grounded checks
# run separately by run_grounded_stage against a live repo checkout.
SEMANTIC_CHECKS = [Q07, Q03, Q01, Q02, Q05, Q04, Q08, Q10, Q11,
                   P02, P03, A01, A02, A03, A04, A05, AL01, Q13]

_PROPOSER_SYS = (
    "You are a strict, evidence-based QC reviewer of grading rubrics. Judge only the check and "
    "criteria you are given. Judge only from the bundle context provided below. Finish by "
    "calling submit_findings with exactly one finding per listed criterion; evidence must quote "
    "the criterion text you relied on.")

_ADVERSARY_SYS = (
    "You are an adversarial second reviewer of grading-rubric QC. For each criterion a prior "
    "review marked FAIL, argue whether it is actually acceptable (defend it). For each marked "
    "PASS, look for a violation the first reviewer missed (attack it). Finish by calling "
    "submit_findings with your verdict per criterion; passed=true means the criterion is fine.")


def proposer_role() -> Role:
    return Role(name="proposer", system_prompt=_PROPOSER_SYS, tools=text_tools())


def adversary_role() -> Role:
    return Role(name="adversary", system_prompt=_ADVERSARY_SYS, tools=text_tools())


_PROSE_PROPOSER_SYS = (
    "You are a strict QC reviewer of a SWE task's authored text (its prompt or its "
    "reference answer). Judge only the one check and the one document you are given, "
    "from the text shown below. Finish by calling submit_findings with exactly one "
    "finding for the listed unit; evidence must quote the text you relied on.")

_PROSE_ADVERSARY_SYS = (
    "You are an adversarial second reviewer of authored SWE task text. If a prior "
    "review marked the document FAIL, argue whether it is actually acceptable; if PASS, "
    "look for a violation it missed. Finish by calling submit_findings with your verdict "
    "for the one unit; passed=true means the document is fine.")


def prose_proposer_role() -> Role:
    return Role(name="prose_proposer", system_prompt=_PROSE_PROPOSER_SYS, tools=text_tools())


def prose_adversary_role() -> Role:
    return Role(name="prose_adversary", system_prompt=_PROSE_ADVERSARY_SYS, tools=text_tools())


def _criteria_block(criteria) -> str:
    return "\n".join(f"- criterion_id={c['id']}  title={c.get('title','')!r}" for c in criteria)


def _full_rubric_block(bundle) -> str:
    items = bundle.rubrics if isinstance(getattr(bundle, "rubrics", None), list) else []
    lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ann = it.get("annotations")
        typ = "neg" if "negative" in str(
            ann.get("type", "") if isinstance(ann, dict) else "") else "pos"
        lines.append(f"- [{typ}] criterion_id={it.get('id')}  title={it.get('title', '')!r}")
    return "\n".join(lines)


def _doc_text(bundle, unit_mode) -> str:
    if unit_mode == "prompt":
        return f"Task prompt (tests/prompt.txt):\n{getattr(bundle, 'prompt', '') or ''}"
    if unit_mode == "answer":
        return (f"Task prompt (for context):\n{getattr(bundle, 'prompt', '') or ''}\n\n"
                f"Reference answer (solution/answer.txt):\n{getattr(bundle, 'answer', '') or ''}")
    # bundle: alignment across all three files
    return (f"instruction.md:\n{getattr(bundle, 'instruction', '') or ''}\n\n"
            f"tests/prompt.txt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"solution/answer.txt:\n{getattr(bundle, 'answer', '') or ''}")


def _doc_context(bundle, check) -> str:
    unit = check.unit_mode
    return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            f"{_doc_text(bundle, unit)}\n\n"
            f"Judge this document AS A WHOLE for the check. Submit EXACTLY ONE finding "
            f"with check_id={check.id} and criterion_id=\"{unit}\": passed=true if it "
            f"SATISFIES the check, passed=false if it VIOLATES it; evidence must quote the text.")


def proposer_context(bundle, check, criteria) -> str:
    if check.unit_mode in ("prompt", "answer", "bundle"):
        return _doc_context(bundle, check)
    if check.unit_mode == "rubric":
        return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
                f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"The full rubric:\n{_full_rubric_block(bundle)}\n\n"
                "Judge the rubric AS A WHOLE for this check. Submit EXACTLY ONE finding with "
                f"check_id={check.id} and criterion_id=\"rubric\": passed=true if the rubric "
                "SATISFIES the check, passed=false if it VIOLATES it; evidence must justify it.")
    return (f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            f"Judge each of these criteria (submit one finding per criterion, check_id={check.id}):\n"
            f"{_criteria_block(criteria)}\n\n"
            "passed=true if the criterion SATISFIES the check, passed=false if it VIOLATES it.")


Q06 = SemanticCheck(
    id="Q06", name="Factual soundness", severity=Severity.REJECT, scope=_all_criteria,
    guidance=("A criterion VIOLATES this if the repo at base_commit does NOT support what it "
              "grades on. A POSITIVE criterion must assert a fact that is TRUE in the repo; a "
              "NEGATIVE criterion asserts a FALSE claim, so it is sound only if that claim is "
              "actually FALSE in the repo. passed=true means the code backs the criterion. "
              "Judge only criteria that make a repo-checkable claim; mark passed=true for "
              "criteria that make no code claim (subjective/phrasing is out of scope)."))

_FACTUAL_SYS = (
    "You verify a grading rubric against the actual repository, which is checked out at "
    "/testbed at base_commit inside a network-isolated container. For each criterion, decide "
    "whether the code supports what it grades on: a POSITIVE criterion is sound iff its fact is "
    "TRUE in the repo; a NEGATIVE criterion states a FALSE assertion and is sound iff that "
    "assertion is actually FALSE in the repo. Use run_bash to read source (cat/grep/rg/find/ls/"
    "git log|show) and, only when a claim needs it, to build/test (writes go under /scratch; no "
    "network). If a criterion makes no repo-checkable claim, mark it passed=true. Finish by "
    "calling submit_findings with exactly one finding per criterion; every finding's evidence "
    "must include at least one path:line citation from the repo.")


def factual_role() -> Role:
    return Role(name="factual", system_prompt=_FACTUAL_SYS, tools=factual_tools())


def factual_context(bundle, criteria) -> str:
    return (f"Repository: {getattr(bundle, 'repository', '') or ''} at base_commit "
            f"{getattr(bundle, 'base_commit', '') or ''} (checked out at /testbed).\n\n"
            f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Check Q06 — Factual soundness.\n{Q06.guidance}\n\n"
            f"Verify each of these criteria against the repo (submit one finding per criterion, "
            f"check_id=Q06):\n{_criteria_block(criteria)}\n\n"
            "passed=true if the repo supports the criterion, passed=false if it contradicts or "
            "lacks it. Evidence must cite path:line.")


_GROUNDED_PROMPT_SYS = (
    "You verify, against the real repository checked out at /testbed (base_commit, "
    "network-isolated), whether a SWE task PROMPT genuinely requires exploring the codebase. "
    "Use run_bash (cat/grep/rg/find/ls/git) to judge whether a correct answer could be given "
    "from general knowledge alone, or truly needs repo-specific investigation. Finish with "
    "submit_findings: exactly one finding; evidence must include a path:line citation.")

_GROUNDED_ANSWER_SYS = (
    "You verify, against the real repository at /testbed (base_commit, network-isolated), "
    "whether a task's reference ANSWER is trajectory-like: it must demonstrate genuine codebase "
    "exploration and must NOT be a bare direct answer NOR a raw full-trajectory dump. Use "
    "run_bash to confirm the answer's cited paths/mechanisms are real and that answering "
    "required exploration. Finish with submit_findings: one finding; evidence must cite path:line.")

def grounded_prompt_role() -> Role:
    return Role(name="grounded_prompt", system_prompt=_GROUNDED_PROMPT_SYS, tools=factual_tools())

def grounded_answer_role() -> Role:
    return Role(name="grounded_answer", system_prompt=_GROUNDED_ANSWER_SYS, tools=factual_tools())

def _grounded_head(bundle) -> str:
    return (f"Repository: {getattr(bundle, 'repository', '') or ''} at base_commit "
            f"{getattr(bundle, 'base_commit', '') or ''} (checked out at /testbed).\n\n")

def grounded_prompt_context(bundle) -> str:
    return (_grounded_head(bundle) +
            f"Task prompt (tests/prompt.txt):\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            "Check P04 — Requires codebase exploration. VIOLATES (passed=false) if a correct "
            "answer could be produced without exploring THIS repo (general knowledge / direct "
            "answer). passed=true if answering demands repo-specific investigation. Submit EXACTLY "
            "ONE finding: check_id=P04, criterion_id=\"prompt\"; evidence must cite path:line.")

def grounded_answer_context(bundle) -> str:
    return (_grounded_head(bundle) +
            f"Task prompt:\n{getattr(bundle, 'prompt', '') or ''}\n\n"
            f"Reference answer (solution/answer.txt):\n{getattr(bundle, 'answer', '') or ''}\n\n"
            "Check A06 — Trajectory-like exploration. VIOLATES (passed=false) if the answer is a "
            "bare direct answer with no exploration, OR a raw full-trajectory dump, OR its cited "
            "exploration does not hold up against the repo. passed=true if it shows genuine, real "
            "codebase exploration. Submit EXACTLY ONE finding: check_id=A06, "
            "criterion_id=\"answer\"; evidence must cite path:line.")


def adversary_context(bundle, check, criteria, agg) -> str:
    if check.unit_mode in ("prompt", "answer", "bundle"):
        unit = check.unit_mode
        v = agg.get(unit, {})
        verdict = "PASS" if v.get("passed") else "FAIL"
        return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"{_doc_text(bundle, unit)}\n\n"
                f"A prior review judged this document: {verdict}. Challenge that per the "
                f"rules above, then submit EXACTLY ONE finding (check_id={check.id}, "
                f"criterion_id=\"{unit}\").")
    if check.unit_mode == "rubric":
        v = agg.get("rubric", {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
                f"The full rubric:\n{_full_rubric_block(bundle)}\n\n"
                f"A prior review judged the whole rubric: {verdict}. Challenge that per the rules "
                f"above, then submit EXACTLY ONE finding (check_id={check.id}, criterion_id=\"rubric\").")
    lines = []
    for c in criteria:
        v = agg.get(c["id"], {})
        verdict = "PASS" if v.get("passed") else "FAIL (reject)"
        lines.append(f"- criterion_id={c['id']}  prior_verdict={verdict}  title={c.get('title','')!r}")
    return (f"Check {check.id} — {check.name}.\n{check.guidance}\n\n"
            "A prior review produced these verdicts. Challenge them per the rules above, then "
            f"submit_findings with your verdict per criterion (check_id={check.id}):\n"
            + "\n".join(lines))
