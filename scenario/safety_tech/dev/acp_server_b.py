# -*- coding: utf-8 -*-
from acp_sdk.server.app import create_app
from acp_sdk.server.agent import agent
from acp_sdk.models import Message, MessagePart


@agent(name="ACP_Doctor_B", description="Doctor B (echo)")
def doctor_b(input: list[Message], context) -> Message:
    # Simple echo with prefix
    text = ""
    for m in input or []:
        for p in m.parts or []:
            if p.content:
                text = p.content
    return Message(role="agent", parts=[MessagePart(content_type="text/plain", content=f"Doctor B echoed: {text}")])


app = create_app(doctor_b)
