# -*- coding: utf-8 -*-
from acp_sdk.server.app import create_app
from acp_sdk.server.agent import agent
from acp_sdk.models import Message, MessagePart

try:
    from script.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


@agent(name="ACP_Doctor_A", description="Doctor A (LLM)")
def doctor_a(input: list[Message], context) -> Message:
    text = ""
    for m in input or []:
        for p in m.parts or []:
            if getattr(p, 'content', None):
                text = p.content
    reply = generate_doctor_reply('doctor_a', text or '')
    return Message(role="agent", parts=[MessagePart(content_type="text/plain", content=reply)])


app = create_app(doctor_a)


