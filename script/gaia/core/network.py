"""Enhanced MeshNetwork with dynamic agent management and intelligent routing."""
import asyncio
import json
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .agent import MeshAgent
from protocols.base_adapter import ProtocolAdapter


async def eval_runner(pred: str, truth_path: str) -> Dict[str, Any]:
    """Calculate quality score for final answer."""
    try:
        with open(truth_path, encoding='utf-8') as f:
            truth = json.load(f)
        
        # Simple exact match and basic similarity
        em = int(pred.strip().lower() == truth.get("answer", "").strip().lower())
        
        # Basic similarity score (can be enhanced with ROUGE-L)
        pred_words = set(pred.lower().split())
        truth_words = set(truth.get("answer", "").lower().split())
        
        if pred_words and truth_words:
            similarity = len(pred_words & truth_words) / len(pred_words | truth_words)
        else:
            similarity = 0.0
        
        quality_score = (em + similarity) / 2
        
        return {
            "quality_score": quality_score,
            "exact_match": em,
            "similarity": similarity
        }
    except Exception as e:
        return {
            "quality_score": 0.0,
            "exact_match": 0,
            "similarity": 0.0,
            "error": str(e)
        }


class MeshNetwork:
    """Enhanced MeshNetwork with dynamic agent creation and intelligent routing."""
    
    def __init__(self, adapter: ProtocolAdapter):
        self.adapter = adapter
        self.agents: List[MeshAgent] = []
        self.connections: List[asyncio.StreamWriter] = []
        self.readers: List[asyncio.StreamReader] = []
        self.config: Dict[str, Any] = {}
        
        # Metrics
        self.bytes_tx = 0
        self.bytes_rx = 0
        self.header_overhead = 0
        self.pkt_cnt = 0
        self.token_sum = 0
        self.start_ts = time.time() * 1000
        self.done_ts: Optional[float] = None
        self.done_payload: Optional[str] = None
        
        # Control flags
        self.running = False
        self.relay_tasks: List[asyncio.Task] = []
    
    async def load_and_create_agents(self, config_path: str):
        """Load configuration and dynamically create agents."""
        print(f"📋 Loading configuration from {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        print(f"🤖 Creating {len(self.config['agents'])} agents...")
        
        # Create agents according to configuration
        for agent_config in self.config["agents"]:
            agent = MeshAgent(
                node_id=agent_config["id"],
                name=agent_config["name"],
                tool=agent_config["tool"],
                adapter=self.adapter,
                port=agent_config["port"],
                config=agent_config
            )
            self.agents.append(agent)
            
            # Start agent server
            asyncio.create_task(agent.serve())
        
        # Wait for agents to start
        print("⏳ Waiting for agents to initialize...")
        await asyncio.sleep(2.0)
        print("✅ All agents started successfully")
    
    async def start(self):
        """Connect to configured agent ports and start relay tasks."""
        print("🔗 Connecting to agent ports...")
        
        ports = [agent.port for agent in self.agents]
        
        for port in ports:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                self.connections.append(writer)
                self.readers.append(reader)
                
                # Start relay task for this connection
                task = asyncio.create_task(self._relay(reader, port))
                self.relay_tasks.append(task)
                
                print(f"✅ Connected to agent on port {port}")
            except Exception as e:
                print(f"❌ Failed to connect to port {port}: {e}")
        
        # Start monitoring task
        asyncio.create_task(self._monitor_done())
        self.running = True
        print("🚀 Network started and monitoring for completion")
    
    async def stop(self):
        """Stop the network and all connections."""
        print("🛑 Stopping network...")
        self.running = False
        
        # Cancel relay tasks
        for task in self.relay_tasks:
            task.cancel()
        
        # Close connections
        for writer in self.connections:
            writer.close()
            await writer.wait_closed()
        
        # Stop agents
        for agent in self.agents:
            await agent.stop()
        
        print("✅ Network stopped")
    
    async def broadcast_init(self, doc: str):
        """Broadcast initial document to all agents."""
        print("📄 Broadcasting initial document to all agents...")
        
        # Split document into chunks if too large
        chunk_size = 1000
        chunks = [doc[i:i+chunk_size] for i in range(0, len(doc), chunk_size)]
        
        packet = {
            "type": "doc_init",
            "chunks": chunks,
            "total_length": len(doc)
        }
        
        blob = self.adapter.encode(packet)
        
        for writer in self.connections:
            try:
                writer.write(len(blob).to_bytes(4, "big") + blob)
                await writer.drain()
                self.bytes_tx += 4 + len(blob)
            except Exception as e:
                print(f"Error broadcasting to agent: {e}")
        
        print(f"✅ Broadcasted document ({len(doc)} chars) to {len(self.connections)} agents")
    
    async def _relay(self, reader: asyncio.StreamReader, src_port: int):
        """Relay messages according to configuration routing rules."""
        try:
            while self.running:
                # Read packet size
                size_data = await reader.readexactly(4)
                size = int.from_bytes(size_data, "big")
                
                # Read packet data
                data = await reader.readexactly(size)
                
                # Update metrics
                self.bytes_rx += 4 + size
                packet = self.adapter.decode(data)
                self.pkt_cnt += 1
                self.header_overhead += self.adapter.header_size(packet)
                self.token_sum += packet.get("token_used", 0)
                
                # Check for final answer
                if packet.get("tag") == "final_answer" and not self.done_ts:
                    self.done_ts = time.time() * 1000
                    self.done_payload = packet.get("payload", "")
                    print(f"🎯 Final answer received: {self.done_payload[:100]}...")
                
                # Route message according to configuration
                await self._route_message(packet, src_port, data)
                
        except asyncio.IncompleteReadError:
            # Connection closed
            pass
        except Exception as e:
            print(f"Error in relay for port {src_port}: {e}")
    
    async def _route_message(self, packet: Dict[str, Any], src_port: int, data: bytes):
        """Route message according to configuration rules."""
        message_type = packet.get("type", "unknown")
        
        # Check for broadcast types
        broadcast_types = self.config.get("communication_rules", {}).get("broadcast_types", [])
        if message_type in broadcast_types:
            # Broadcast to all agents except sender
            for writer in self.connections:
                try:
                    peer_port = writer.get_extra_info("peername")[1]
                    if peer_port != src_port:
                        writer.write(len(data).to_bytes(4, "big") + data)
                        await writer.drain()
                        self.bytes_tx += 4 + len(data)
                except Exception as e:
                    print(f"Error broadcasting message: {e}")
        else:
            # Direct routing based on configuration
            await self._direct_route(packet, src_port, data)
    
    async def _direct_route(self, packet: Dict[str, Any], src_port: int, data: bytes):
        """Handle direct routing based on workflow configuration."""
        # Find source agent
        src_agent_id = None
        for agent in self.agents:
            if agent.port == src_port:
                src_agent_id = agent.id
                break
        
        if src_agent_id is None:
            return
        
        # Find target agents based on workflow
        workflow = self.config.get("workflow", {})
        message_flow = workflow.get("message_flow", [])
        
        target_agents = []
        for flow in message_flow:
            if flow["from"] == src_agent_id:
                if flow["to"] == "final":
                    # This is a final answer, no routing needed
                    return
                target_agents.extend(flow["to"])
        
        # Route to target agents
        for target_id in target_agents:
            writer = self._get_writer_by_agent_id(target_id)
            if writer:
                try:
                    writer.write(len(data).to_bytes(4, "big") + data)
                    await writer.drain()
                    self.bytes_tx += 4 + len(data)
                except Exception as e:
                    print(f"Error routing to agent {target_id}: {e}")
    
    def _get_writer_by_agent_id(self, agent_id: int) -> Optional[asyncio.StreamWriter]:
        """Get writer for specific agent ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                # Find corresponding writer
                for i, a in enumerate(self.agents):
                    if a.id == agent_id and i < len(self.connections):
                        return self.connections[i]
        return None
    
    async def _monitor_done(self):
        """Monitor for completion and trigger evaluation."""
        print("👀 Monitoring for final answer...")
        
        # Wait for final answer or timeout
        timeout = self.config.get("performance_targets", {}).get("max_execution_time", 300000)
        timeout_seconds = timeout / 1000
        
        start_time = time.time()
        while self.done_ts is None and self.running:
            await asyncio.sleep(1)
            
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                print("⏰ Execution timeout reached")
                self.done_ts = time.time() * 1000
                self.done_payload = "TIMEOUT: No final answer received within time limit"
                break
        
        if self.done_payload:
            print("🔄 Starting evaluation...")
            await self._evaluate()
    
    async def _evaluate(self):
        """Run final evaluation and archive artifacts."""
        print("📊 Running evaluation...")
        
        # Run quality evaluation
        try:
            quality = await eval_runner(self.done_payload or "", "ground_truth.json")
        except Exception as e:
            print(f"Evaluation error: {e}")
            quality = {"quality_score": 0.0, "exact_match": 0, "error": str(e)}
        
        # Compile metrics report
        report = {
            "performance_metrics": {
                "bytes_tx": self.bytes_tx,
                "bytes_rx": self.bytes_rx,
                "pkt_cnt": self.pkt_cnt,
                "header_overhead": self.header_overhead,
                "token_sum": self.token_sum,
                "elapsed_ms": self.done_ts - self.start_ts if self.done_ts else 0
            },
            "quality_metrics": quality,
            "configuration": {
                "num_agents": len(self.agents),
                "task_id": self.config.get("task_id", "unknown"),
                "complexity": self.config.get("task_analysis", {}).get("complexity", "unknown")
            },
            "final_answer": self.done_payload
        }
        
        # Save metrics
        with open("metrics.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("💾 Metrics saved to metrics.json")
        
        # Archive workspaces
        await self._archive_artifacts()
        
        print("✅ Evaluation complete!")
        print(f"📈 Quality Score: {quality.get('quality_score', 0):.2f}")
        print(f"⚡ Total Time: {report['performance_metrics']['elapsed_ms']:.0f}ms")
        print(f"🔢 Total Tokens: {report['performance_metrics']['token_sum']}")
    
    async def _archive_artifacts(self):
        """Archive workspace artifacts."""
        try:
            workspaces_path = Path("workspaces")
            if workspaces_path.exists():
                with tarfile.open("run_artifacts.tar.gz", "w:gz") as tar:
                    for workspace in workspaces_path.iterdir():
                        if workspace.is_dir():
                            tar.add(workspace, arcname=workspace.name)
                print("📦 Artifacts archived to run_artifacts.tar.gz")
            else:
                print("📁 No workspaces found to archive")
        except Exception as e:
            print(f"Error archiving artifacts: {e}")
