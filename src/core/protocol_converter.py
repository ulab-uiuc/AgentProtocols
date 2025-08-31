"""
Converter functions between UTE (UnifiedTransportEnvelope) and each protocol.
All functions are pure; they never perform network I/O.
"""

from typing import Dict, Any

from .unified_message import UTE


# ---------------- A2A ----------------
def ute_to_a2a(ute: UTE) -> Dict[str, Any]:
    """Convert UTE -> A2A official message."""
    return {
        "id": ute.id,
        "params": {
            "message": {
                # A2A SDK accepts 'parts', we do the simplest mapping: just one text/JSON part
                "parts": [{"type": "json", "text": ute.content}],
            },
            "context": ute.context,
            "routing": {
                "destination": ute.dst,
                "source": ute.src,
            },
        },
    }


def a2a_to_ute(a2a_msg: Dict[str, Any]) -> UTE:
    """Convert A2A -> UTE."""
    params = a2a_msg.get("params", {})
    parts = params.get("message", {}).get("parts", [{}])
    # The actual business data is in parts[0]["text"]
    content = parts[0].get("text", {})
    if isinstance(content, str):
        try:
            import json
            content = json.loads(content)
        except Exception:  # noqa: broad-except
            content = {"text": content}
    return UTE(
        id=a2a_msg.get("id", ""),
        src=params.get("routing", {}).get("source", ""),
        dst=params.get("routing", {}).get("destination", ""),
        timestamp=params.get("message", {}).get("time", 0) or 0.0,
        content=content,
        context=params.get("context", {}),
        metadata={},
    )


# ---------------- ACP ----------------
def ute_to_acp(ute: UTE) -> Dict[str, Any]:
    """UTE -> ACP message body."""
    return {
        "id": ute.id,
        "type": "request",
        "sender": ute.src,
        "receiver": ute.dst,
        "payload": ute.content,
        "timestamp": ute.timestamp,
        "metadata": ute.metadata,
        "context": ute.context,
    }


def acp_to_ute(acp_msg: Dict[str, Any]) -> UTE:
    """ACP -> UTE."""
    return UTE(
        id=acp_msg.get("id", ""),
        src=acp_msg.get("sender", ""),
        dst=acp_msg.get("receiver", ""),
        timestamp=acp_msg.get("timestamp", 0.0),
        content=acp_msg.get("payload", {}),
        context=acp_msg.get("context", {}),
        metadata=acp_msg.get("metadata", {}),
    )


# ---------------- Agent Protocol (AP) ----------------
def ute_to_ap_create_task(ute: UTE) -> Dict[str, Any]:
    """UTE -> AP 'create_task' message, the most common mapping."""
    return {
        "type": "create_task",
        "input": ute.content.get("text", ute.content),
        "additional_input": ute.context,
        # Put extra UTE info into metadata
        "_ute_meta": ute.metadata,
    }


def ap_to_ute(ap_msg: Dict[str, Any]) -> UTE:
    """AP -> UTE (supports output from create_task / get_task / execute_step)."""
    return UTE(
        id=ap_msg.get("_request_id", ""),
        src="ap_remote",
        dst="",
        timestamp=ap_msg.get("timestamp", 0.0),
        content=ap_msg,
        context={},
        metadata={},
    )


# ---------------- ANP ----------------
def ute_to_anp(ute: UTE) -> Dict[str, Any]:
    """UTE -> ANP message (wrapped in 'payload')."""
    return {
        "request_id": ute.id,
        "source_did": ute.src,
        "target_did": ute.dst,
        "timestamp": ute.timestamp,
        "payload": {
            "content": ute.content,
            "context": ute.context,
            "metadata": ute.metadata,
        },
    }


def anp_to_ute(anp_msg: Dict[str, Any]) -> UTE:
    """ANP -> UTE."""
    payload = anp_msg.get("payload", {})
    return UTE(
        id=anp_msg.get("request_id", ""),
        src=anp_msg.get("source_did", ""),
        dst=anp_msg.get("target_did", ""),
        timestamp=anp_msg.get("timestamp", 0.0),
        content=payload.get("content", {}),
        context=payload.get("context", {}),
        metadata=payload.get("metadata", {}),
    )


# ---------------- Agora ----------------
def ute_to_agora(ute: UTE) -> Dict[str, Any]:
    """UTE -> Agora 'general' message input."""
    return {
        "message": ute.content.get("text", ute.content),
        "type": "general",
        "context": ute.context,
        "_ute_meta": ute.metadata,
    }


def agora_to_ute(agora_msg: Dict[str, Any]) -> UTE:
    """Agora -> UTE (simplified)."""
    return UTE(
        id=agora_msg.get("id", ""),
        src="agora_remote",
        dst="",
        timestamp=agora_msg.get("timestamp", 0.0),
        content=agora_msg,
        context={},
        metadata={},
    )


# ---- registry (helps BaseAgent pick the right pair quickly) ----
ENCODE_TABLE = {
    "a2a": ute_to_a2a,
    "acp": ute_to_acp,
    "agentprotocol": ute_to_ap_create_task,
    "anp": ute_to_anp,
    "agora": ute_to_agora,
}

DECODE_TABLE = {
    "a2a": a2a_to_ute,
    "acp": acp_to_ute,
    "agentprotocol": ap_to_ute,
    "anp": anp_to_ute,
    "agora": agora_to_ute,
} 