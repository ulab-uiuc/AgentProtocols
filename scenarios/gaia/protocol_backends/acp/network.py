"""
ACP Network implementation for GAIA framework.
Uses acp_sdk client/server HTTP APIs to send messages and orchestrate agents.
The communication model is different from dummy/agora and follows the
provided acp client/server demo style.
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional, List
import os
import sys
import re

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from core.schema import Message, ExecutionStatus, Colors
from protocol_backends.acp.agent import ACPAgent

# ACP SDK imports
try:
    from acp_sdk.client import Client as ACPClient
    from acp_sdk.models import Message as ACPMessage, MessagePart as ACPMessagePart
except ImportError as e:
    raise ImportError(f"ACP SDK components required but not available: {e}")


def _sanitize_agent_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name or "agent")


class ACPCommBackend:
    """acp_sdk HTTP communication backend.

    - Registers per-agent base_url and acp agent name
    - Sends Run requests using ACP client
    - Extracts text outputs for GAIA
    """

    def __init__(self) -> None:
        # id -> { base_url, agent }
        self._registry: Dict[str, Dict[str, str]] = {}
        self._clients: Dict[str, Any] = {}

    async def register_agent(self, agent_id: str, base_url: str, agent_name: str) -> None:
        self._registry[agent_id] = {
            "base_url": base_url,
            "agent": _sanitize_agent_name(agent_name),
        }

    def _get_client(self, agent_id: str):
        client = self._clients.get(agent_id)
        if client is None:
            base = self._registry[agent_id]["base_url"].replace("localhost", "127.0.0.1")
            client = ACPClient(base_url=base)
            self._clients[agent_id] = client
        return client

    async def close(self) -> None:
        for k, c in list(self._clients.items()):
            try:
                await c.__aexit__(None, None, None)  # ensure close if used as context
            except Exception:
                try:
                    await c.aclose()
                except Exception:
                    pass
        self._clients.clear()

    async def send(self, src_id: str, dst_id: str, text: str) -> str:
        reg = self._registry.get(dst_id)
        if not reg:
            raise RuntimeError(f"Unknown ACP agent id: {dst_id}")
        client = self._get_client(dst_id)
        agent_name = reg["agent"]

        last_err = None
        for attempt in range(1, 5):
            try:
                async with client:  # type: ignore
                    run = await client.run_sync(
                        agent=agent_name,
                        input=[
                            ACPMessage(
                                role="user",
                                parts=[ACPMessagePart(content=text, content_type="text/plain")],
                            )
                        ],
                    )
                    out_text = self._extract_text(run.output)
                    if not out_text:
                        print(f"[ACPCommBackend] empty output from agent '{agent_name}' (dst_id={dst_id}); raw={getattr(run, 'output', None)}")
                    return out_text
            except Exception as e:
                last_err = e
                sleep_s = min(0.1 * (2 ** (attempt - 1)), 0.8)
                print(f"[ACPCommBackend] attempt {attempt} failed for {dst_id} ({agent_name}): {e}; retrying in {sleep_s:.2f}s")
                await asyncio.sleep(sleep_s)
        print(f"[ACPCommBackend] all attempts failed for {dst_id} ({agent_name})")
        raise last_err  # type: ignore

    def _extract_text(self, output: Any) -> str:
        try:
            if output is None:
                return ""
            if isinstance(output, list):
                texts: List[str] = []
                for m in output:
                    parts = getattr(m, "parts", []) or []
                    for p in parts:
                        ctype = getattr(p, "content_type", "text/plain")
                        if ctype == "text/plain":
                            texts.append(str(getattr(p, "content", "")))
                return "\n".join([t for t in texts if t])
            if isinstance(output, dict):
                if "text" in output:
                    return str(output["text"]) or ""
                if "body" in output:
                    return str(output["body"]) or ""
            return str(output)
        except Exception:
            return ""


class ACPNetwork(MeshNetwork):
    """ACP Network implementation."""

    def __init__(self, config: Dict[str, Any]):
        if isinstance(config, dict):
            config = {**config, "protocol": config.get("protocol", "acp")}
        super().__init__(config=config)
        self.comm_backend = ACPCommBackend()
        self.register_agents_from_config()

    def create_acp_agent(self, agent_config: Dict[str, Any], task_id: str, agent_prompts: Optional[Dict[str, Any]] = None) -> ACPAgent:
        agent_id = agent_config["id"]
        system_prompt = None
        if agent_prompts and str(agent_id) in agent_prompts:
            system_prompt = agent_prompts[str(agent_id)].get("system_prompt")
        return ACPAgent(
            node_id=agent_id,
            name=agent_config["name"],
            tool=agent_config["tool"],
            port=agent_config["port"],
            config={
                "max_tokens": agent_config.get("max_tokens", 500),
                "role": agent_config.get("role", "agent"),
                "priority": agent_config.get("priority", 1),
                "system_prompt": system_prompt,
                "protocol": "acp",
            },
            task_id=task_id,
        )

    def register_agents_from_config(self) -> Dict[str, Any]:
        if "workflow" not in self.config:
            raise ValueError("Full configuration must contain 'workflow' key")
        agent_configs = self.config.get("agents", [])
        agent_prompts = self.config.get("agent_prompts", {})
        print(f"📝 Preparing to create {len(agent_configs)} ACP agents")
        for agent_info in agent_configs:
            agent = self.create_acp_agent(agent_info, self.task_id, agent_prompts)
            self.register_agent(agent)
            endpoint = f"http://127.0.0.1:{agent_info['port']}"
            # Dynamically set ACP-side name: `<id>_<name>` to match workspace
            acp_name = f"{agent.id}_{agent.name}"
            try:
                agent._acp_agent_name = acp_name
            except Exception:
                pass
            # Register with communication backend (base_url, agent_name)
            asyncio.create_task(self.comm_backend.register_agent(str(agent_info["id"]), endpoint, acp_name))
            # Agent also stores endpoint for health checks and diagnostics
            try:
                agent._endpoint = endpoint
            except Exception:
                pass
            # Start each agent's ACP server
            try:
                agent.run_server_background()
            except Exception as e:
                print(f"❌ Failed to start ACP agent server {agent_info['id']}: {e}")
            print(f"✅ ACP Agent {agent_info['name']} (ID: {agent_info['id']}) has been created and registered; ACP name: {acp_name}")
        print(f"🎉 Successfully registered {len(agent_configs)} ACP agents in total")

    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        try:
            text = msg.get("content") or json.dumps(msg, ensure_ascii=False)
            resp = await self.comm_backend.send(src_id="network", dst_id=str(dst), text=text)
            self.pkt_cnt += 1
            self.bytes_tx += len(json.dumps(msg).encode("utf-8"))
            print(f"📤 ACPNetwork -> {dst}: {msg.get('type','unknown')} | resp: {str(resp)[:80] if resp else 'ok'}")
        except Exception as e:
            print(f"❌ Failed to deliver ACP message to agent {dst}: {e}")

    async def start(self):
        print("🌐 Starting ACP multi-agent network...")
        await super().start()
        print("🔗 ACP communication backend initialized")
        try:
            ok = await self.monitor_agent_health()
            if not ok:
                print(f"{Colors.YELLOW}⚠️  Post-start health check: some agents may be unreachable yet{Colors.RESET}")
        except Exception as _e:
            print(f"[ACPNetwork] Post-start health check failed: {_e}")

    async def stop(self):
        print("🛑 Stopping ACP network...")
        try:
            await self.comm_backend.close()
            print("✅ ACP communication backend closed")
        except Exception as e:
            print(f"⚠️ Error closing ACP communication backend: {e}")
        await super().stop()
        print("✅ ACP network stopped")

    def get_network_stats(self) -> Dict[str, Any]:
        stats = {
            "network_type": "acp",
            "total_agents": len(self.agents),
            "performance_metrics": {
                "bytes_tx": self.bytes_tx,
                "bytes_rx": self.bytes_rx,
                "pkt_cnt": self.pkt_cnt,
                "header_overhead": self.header_overhead,
                "token_sum": self.token_sum,
            },
            "agent_stats": [],
        }
        for agent in self.agents:
            try:
                stats["agent_stats"].append(agent.get_connection_status())
            except Exception as e:
                stats["agent_stats"].append({"agent_id": agent.id, "error": str(e)})
        return stats

    async def _execute_agent_step(self, agent_id: int, context_message: str, step_idx: int) -> str:
        agent = self.get_agent_by_id(agent_id)
        if not agent:
            raise Exception(f"Agent {agent_id} not found")
        self.network_memory.add_step_execution(
            step=step_idx,
            agent_id=str(agent_id),
            agent_name=agent.name,
            task_id=self.task_id,
            user_message=context_message,
        )
        self.network_memory.update_step_status(step_idx, ExecutionStatus.PROCESSING)
        try:
            resp_text = await self.comm_backend.send(src_id="network", dst_id=str(agent_id), text=context_message)
            messages = [
                Message.user_message(context_message),
                Message.assistant_message(resp_text or ""),
            ]
            if not resp_text or not str(resp_text).strip():
                self.network_memory.update_step_status(step_idx, ExecutionStatus.ERROR, error_message="Empty result from agent")
                return "No meaningful result generated by agent"
            else:
                self.network_memory.update_step_status(step_idx, ExecutionStatus.SUCCESS, messages=messages)
                return resp_text
        except Exception as e:
            self.network_memory.update_step_status(step_idx, ExecutionStatus.ERROR, error_message=str(e))
            return f"Error: {e}"
