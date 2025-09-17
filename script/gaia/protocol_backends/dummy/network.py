"""
Dummy Network implementation for GAIA framework.
Implements broadcast and polling mechanisms for dummy protocol.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.network import MeshNetwork
from protocol_backends.dummy.agent import DummyAgent


class DummyNetwork(MeshNetwork):
    """
    Dummy Network implementation with broadcast and enhanced polling capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Ensure protocol is set for workspace namespacing
        if isinstance(config, dict):
            config = {**config, "protocol": config.get("protocol", "dummy")}
        super().__init__(config=config)
        self.register_agents_from_config()
    
    def create_dummy_agent(self,
        agent_config: Dict[str, Any], task_id: str, 
        agent_prompts: Optional[Dict[str, Any]] = None
    ) -> DummyAgent:
        """Create a DummyAgent from configuration."""
        agent_id = agent_config['id']

        # Get system prompt from agent_prompts[agent_id] if available
        system_prompt = None
        if agent_prompts and str(agent_id) in agent_prompts:
            system_prompt = agent_prompts[str(agent_id)].get('system_prompt')

        return DummyAgent(
            node_id=agent_id,
            name=agent_config['name'],
            tool=agent_config['tool'],
            port=agent_config['port'],
            config={
                'max_tokens': agent_config.get('max_tokens', 500),
                'role': agent_config.get('role', 'agent'),
                'priority': agent_config.get('priority', 1),
                'system_prompt': system_prompt,  # Add system prompt to config
                'message_loss_rate': 0.05,  # 5% message loss for testing
                'delivery_delay': 0.1,      # 100ms delay
                'max_message_size': 1024,   # 1KB messages
                'protocol': 'dummy'
            },
            task_id=task_id,
            router_url="dummy://localhost:11008"
        )

    def register_agents_from_config(self) -> Dict[str, Any]:
        """
        Create and register multiple agents from configuration.
        
        Args:
            task_id: Task ID to assign to all agents
            self.config: Full configuration including agent_prompts (optional)
            
        Returns:
            Complete configuration with updated task_id
        """     
        # Update task_id in workflow
        if "workflow" not in self.config:
            raise ValueError("Full configuration must contain 'workflow' key")
        
        # Extract agent configurations and prompts
        agent_configs = self.config.get('agents', [])
        agent_prompts = self.config.get('agent_prompts', {})
        
        print(f"📝 准备创建 {len(agent_configs)} 个Agent")
        
        for agent_info in agent_configs:
            try:
                # Create dummy agent from configuration with proper system prompt
                agent = self.create_dummy_agent(
                    agent_config=agent_info, 
                    task_id=self.task_id, 
                    agent_prompts=agent_prompts
                )
                
                # Register agent to network
                self.register_agent(agent)
                
                print(f"✅ Agent {agent_info['name']} (ID: {agent_info['id']}) 已创建并注册")
                
            except Exception as e:
                print(f"❌ 创建和注册Agent {agent_info.get('name', 'unknown')} 失败: {e}")
                raise
        
        print(f"🎉 总共成功注册了 {len(agent_configs)} 个Agent")

    # ==================== Communication Methods ====================
    
    async def deliver(self, dst: int, msg: Dict[str, Any]) -> None:
        """
        Deliver message to specific agent using dummy protocol.
        
        Args:
            dst: Destination agent ID
            msg: Message payload
        """
        # Find the destination agent
        target_agent = self.get_agent_by_id(dst)
        if target_agent:
            try:
                # For DummyAgent, directly put message into agent's mailbox
                if hasattr(target_agent, '_client') and hasattr(target_agent._client, '_mailbox'):
                    # Add network metadata
                    enhanced_msg = {
                        **msg,
                        "_network_meta": {
                            "delivered_by": "dummy_network",
                            "target_agent": dst,
                            "delivery_timestamp": time.time(),
                            "message_id": f"net_{int(time.time() * 1000000)}"
                        }
                    }
                    
                    # Directly deliver to agent's mailbox
                    await target_agent._client._mailbox.put(enhanced_msg)
                    
                    # Update network metrics
                    self.pkt_cnt += 1
                    msg_size = len(json.dumps(msg).encode('utf-8'))
                    self.bytes_tx += msg_size
                    
                    # Update sender agent statistics
                    if hasattr(target_agent, '_client'):
                        # This message is being delivered, so increment received count
                        target_agent._client.received_messages += 1
                    
                    print(f"📤 DummyNetwork delivered message to agent {dst}: {msg.get('type', 'unknown')}")
                else:
                    print(f"❌ Agent {dst} does not support dummy message delivery")
            except Exception as e:
                print(f"❌ Failed to deliver message to agent {dst}: {e}")
        else:
            print(f"❌ Agent {dst} not found in network")