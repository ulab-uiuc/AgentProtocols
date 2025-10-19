# -*- coding: utf-8 -*-
from acp_sdk.server.app import create_app
from acp_sdk.server.agent import agent
from acp_sdk.models import Message, MessagePart


@agent(name="ACP_Doctor_A", description="Doctor A (echo)")
def doctor_a(input: list[Message], context) -> Message:
    # Simple echo of the last user text
    text = ""
    for m in input or []:
        for p in m.parts or []:
            if p.content:
                text = p.content
    return Message(role="agent", parts=[MessagePart(content_type="text/plain", content=f"Doctor A received: {text}")])


app = create_app(doctor_a)
