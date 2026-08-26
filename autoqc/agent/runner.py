from __future__ import annotations
import json
from dataclasses import dataclass, field
from autoqc.agent.tools import Tool


@dataclass
class Role:
    name: str
    system_prompt: str
    tools: list[Tool]


@dataclass
class AgentResult:
    findings: list[dict] = field(default_factory=list)
    ok: bool = False
    reason: str = ""


def _assistant_msg(resp) -> dict:
    return {"role": "assistant", "content": resp.text or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.args)}}
                           for tc in resp.tool_calls]}


def run_agent(role: Role, context_text: str, client, ctx, max_turns: int = 12) -> AgentResult:
    by_name = {t.name: t for t in role.tools}
    schemas = [t.schema() for t in role.tools]
    messages = [{"role": "system", "content": role.system_prompt},
                {"role": "user", "content": context_text}]
    for _ in range(max_turns):
        try:
            resp = client.chat(messages, tools=schemas)
        except Exception as e:  # gateway/network/parse error
            return AgentResult(ok=False, reason=f"chat error: {e}")

        if not resp.tool_calls:
            messages.append({"role": "assistant", "content": resp.text or ""})
            messages.append({"role": "user", "content": "Call submit_findings to finish."})
            continue

        messages.append(_assistant_msg(resp))
        for call in resp.tool_calls:
            if call.name == "submit_findings":
                findings = call.args.get("findings", []) if isinstance(call.args, dict) else []
                return AgentResult(findings=findings, ok=True)
            tool = by_name.get(call.name)
            try:
                out = tool.run(call.args, ctx) if (tool and tool.run) else f"error: unknown tool {call.name!r}"
            except Exception as e:
                out = f"error: tool {call.name!r} failed: {e}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
    return AgentResult(ok=False, reason=f"exceeded max_turns ({max_turns}) without submit_findings")
