# -*- coding: utf-8 -*-
from acp_sdk.server.app import create_app
from acp_sdk.server.agent import agent
from acp_sdk.models import Message, MessagePart

try:
    from scenarios.safety_tech.core.llm_wrapper import generate_doctor_reply
except Exception:
    from core.llm_wrapper import generate_doctor_reply


@agent(name="ACP_Doctor_B", description="Doctor B (LLM)")
def doctor_b(input: list[Message], context) -> Message:
    text = ""
    for m in input or []:
        for p in m.parts or []:
            if getattr(p, 'content', None):
                text = p.content
    # 提取correlation_id前缀 [CID:...]
    cid = None
    if text.startswith('[CID:'):
        try:
            end = text.find(']')
            if end != -1:
                cid = text[5:end]
                text = text[end+1:].lstrip()
        except Exception:
            cid = None
    reply = generate_doctor_reply('doctor_b', text or '')
    # 回投协调器/deliver（最佳努力，不影响主回复）
    try:
        import httpx, os
        coord = os.environ.get('COORD_ENDPOINT', 'http://127.0.0.1:8888')
        payload = {"sender_id": "ACP_Doctor_B", "receiver_id": "ACP_Doctor_A", "text": reply}
        if cid:
            payload['correlation_id'] = cid
        try:
            with httpx.Client(timeout=2.0) as c:
                c.post(f"{coord}/deliver", json=payload)
        except Exception:
            pass
    except Exception:
        pass
    return Message(role="agent", parts=[MessagePart(content_type="text/plain", content=reply)])


app = create_app(doctor_b)


