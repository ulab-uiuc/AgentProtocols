# **Gaia Technical Design Document**

## **0. Design Goals & Hard Constraints**

This document outlines the architecture for **Gaia**, a multi-agent benchmark system designed for local execution and evaluation. The system adheres to the following core principles and constraints.

| # | Constraint | Description |
| --- | --- | --- |
| 1 | **Initial Document Broadcast** | The `MeshNetwork` process must broadcast the external Gaia document to all agents at `t=0` via `MeshNetwork.broadcast_init()`. |
| 2 | **Asynchronous P2P Communication** | Each agent operates as an independent asynchronous process. All communication is point-to-point, relayed by the central `MeshNetwork`. |
| 3 | **One Tool per Agent** | Each agent is bound to a single, specific tool (e.g., `search`, `extract`) defined at startup. |
| 4 | **Local-Only Execution**| All components run locally (`127.0.0.1`) using a dedicated range of ports (e.g., 9000-900N). |
| 5 | **Pluggable Communication Protocol** | The underlying communication protocol (e.g., JSON, ANP, A2A) must be interchangeable via a `ProtocolAdapter` abstraction layer. |
| 6 | **Private Agent Workspaces** | Each agent maintains a private, isolated workspace for logs, memory, and other artifacts, located at `workspaces/<node_id>/`. |
| 7 | **Integrated Evaluation & Logging** | The `MeshNetwork` is responsible for final evaluation, metric aggregation (`metrics.json`), and archiving all run artifacts (`run_artifacts.tar.gz`). |

---

## **1. Overall Architecture**

The system is designed around a **star topology** with a central `MeshNetwork` acting as a message relay and controller. Agents do not communicate directly; instead, they publish messages to the network, which then broadcasts them to all other agents. This decouples agents and allows the network to handle monitoring, logging, and evaluation seamlessly.

```
                 ┌─────────────── MeshNetwork (Transport + Metrics) ────────────────┐
                 │  (1) broadcast_init(doc)                                         │
                 │  (2) TCP relay ⤷ count bytes / tokens / pkt                      │
┌─────────────┐  │  (3) detect tag="final_answer"                                   │
│  Agent#0    │◀─┼─────────┐                                                       │
│  tool=search│  │         ▼                                                       │
│  port:9000  │  │  eval_runner → metrics.json & artifacts archive                 │
└─────────────┘  └──────────────────────────────────────────────────────────────────┘
      ▲                                ▲
      ▼ “search_docs”                 ▼ “events”
┌─────────────┐                ┌─────────────┐
│  Agent#1    │                │  Agent#2    │
│ tool=extract│                │ tool=triple │
│ port:9001   │                │ port:9002   │
└─────────────┘  ▲             └─────────────┘  ▲
      ▼ “triples”│                    ▼ “final_answer”
┌─────────────┐  │
│  Agent#3    │──┘
│ tool=reason │
│ port:9003   │
└─────────────┘
```

**Workflow Sequence:**

1.  **Initialization**: The `MeshNetwork` starts and broadcasts the initial Gaia document to all connected agents.
2.  **Task Processing**: The `search` agent receives the document, processes it, and emits its findings.
3.  **Message Relay**: The `MeshNetwork` relays this message to all other agents.
4.  **Pipeline Execution**: The `extract`, `triple`, and `reason` agents listen for relevant incoming messages, execute their respective tools in sequence, and publish their results.
5.  **Final Answer**: The `reason` agent produces the final answer and tags it for the network.
6.  **Evaluation**: The `MeshNetwork` detects the `final_answer` tag, stops the clock, runs the evaluation script, and archives all logs and workspaces.

---

## **2. Directory Structure**

To maintain a clean and scalable codebase, the project will adhere to the following directory structure, separating concerns between core logic, protocols, and execution scripts.

```
gaia_benchmark/
├── core/                     # ★ Core system logic
│   ├── network.py            # MeshNetwork implementation
│   └── agent.py              # MeshAgent template
├── protocols/                # 🔌 Pluggable protocol adapters
│   ├── base_adapter.py       # ProtocolAdapter abstract base class
│   └── json_adapter.py       # Default JSON implementation
├── tools/                    # 🛠️ Agent tool definitions
│   └── registry.py           # Tool registration logic
├── workspaces/               # 📂 Agent-specific runtime directories (auto-generated)
│   ├── 0/
│   └── 1/
├── scripts/                  # 🚀 Entry points and evaluation
│   ├── run_benchmark.py      # Main script to launch the system
│   └── evaluate.py           # Evaluation logic (eval_runner)
├── config/                   # ⚙️ Configuration files
│   └── default.yaml          # Defines agents, tools, and ports
└── docs/                     # 📖 System documentation
    └── README.md
```

---

## **3. ProtocolAdapter Interface**

To satisfy the pluggable protocol requirement, all communication encoding and decoding will be handled through a `ProtocolAdapter`. This allows the system to switch from a simple JSON protocol to a more efficient or secure binary protocol without changing the core agent or network logic.

```python
# located in protocols/base_adapter.py
import abc

class ProtocolAdapter(abc.ABC):
    """Abstract interface for encoding and decoding network packets."""

    @abc.abstractmethod
    def encode(self, packet: dict) -> bytes:
        """Encodes a dictionary packet into bytes."""
        ...

    @abc.abstractmethod
    def decode(self, blob: bytes) -> dict:
        """Decodes bytes into a dictionary packet."""
        ...

    @abc.abstractmethod
    def header_size(self, packet: dict) -> int:
        """Calculates the size of the protocol's header for metrics."""
        ...
```

#### **Baseline JSON Implementation**

A simple, human-readable implementation using JSON will serve as the baseline.

```python
# located in protocols/json_adapter.py
import json
from .base_adapter import ProtocolAdapter

class JsonAdapter(ProtocolAdapter):
    def encode(self, p: dict) -> bytes:
        return json.dumps(p).encode('utf-8')

    def decode(self, b: bytes) -> dict:
        return json.loads(b)

    def header_size(self, p: dict) -> int:
        # JSON has no separate header, so overhead is zero.
        return 0
```

---

## **4. Core Component Logic**

The system consists of two primary components: the `MeshNetwork` and the `MeshAgent`.

### **4.1. MeshNetwork**

The `MeshNetwork` is the central hub. It does not contain any application logic but is responsible for reliable message delivery, metrics collection, and orchestrating the benchmark lifecycle.

```python
# Core logic for core/network.py
class MeshNetwork:
    def __init__(self, ports: list[int], adapter: ProtocolAdapter):
        self.ports = ports
        self.adapter = adapter
        self.conns: list[asyncio.StreamWriter] = []
        # Metrics
        self.bytes_tx = self.bytes_rx = self.header_overhead = 0
        self.pkt_cnt = self.token_sum = 0
        self.start_ts = time.time() * 1000
        self.done_ts = None
        self.done_payload = None

    async def start(self):
        """Connects to all agent ports and starts the relay tasks."""
        for p in self.ports:
            r, w = await asyncio.open_connection("127.0.0.1", p)
            self.conns.append(w)
            asyncio.create_task(self._relay(r, p))
        asyncio.create_task(self._monitor_done())

    async def broadcast_init(self, doc: str):
        """Broadcasts the initial document to all agents."""
        chunks = [doc[i:i+400] for i in range(0, len(doc), 400)]
        pkt = {"type": "doc_init", "chunks": chunks}
        blob = self.adapter.encode(pkt)
        for w in self.conns:
            w.write(len(blob).to_bytes(4, "big") + blob)
            await w.drain()

    async def _relay(self, reader, src_port):
        """Reads from one agent and broadcasts to all others."""
        while True:
            size = int.from_bytes(await reader.readexactly(4), "big")
            data = await reader.readexactly(size)
            # Update RX metrics
            self.bytes_rx += 4 + size
            pkt = self.adapter.decode(data)
            self.pkt_cnt += 1
            self.header_overhead += self.adapter.header_size(pkt)
            self.token_sum += pkt.get("token_used", 0)

            # Check for final answer
            if pkt.get("tag") == "final_answer" and not self.done_ts:
                self.done_ts = time.time() * 1000
                self.done_payload = pkt["payload"]

            # Broadcast to other agents
            for w in self.conns:
                if w.get_extra_info("peername")[1] == src_port:
                    continue
                w.write(size.to_bytes(4, "big") + data)
                await w.drain()
                self.bytes_tx += 4 + size

    async def _monitor_done(self):
        """Waits for the 'final_answer' tag and then triggers evaluation."""
        while self.done_ts is None:
            await asyncio.sleep(1)
        await self._evaluate()

    async def _evaluate(self):
        """Runs the final evaluation and archives artifacts."""
        quality = await eval_runner(self.done_payload, "ground_truth.json")
        report = dict(bytes_tx=self.bytes_tx,
                      bytes_rx=self.bytes_rx,
                      pkt_cnt=self.pkt_cnt,
                      header_overhead=self.header_overhead,
                      token_sum=self.token_sum,
                      elapsed_ms=self.done_ts - self.start_ts,
                      **quality)
        with open("metrics.json", "w") as f:
            json.dump(report, f, indent=2)

        # Archive workspaces
        with tarfile.open("run_artifacts.tar.gz", "w:gz") as tar:
            for d in Path("workspaces").iterdir():
                tar.add(d, arcname=d.name)
```

### **4.2. MeshAgent**

The `MeshAgent` is a generic agent template. Each instance is configured with a unique ID and a single tool. It listens for incoming packets, processes them if relevant, and emits its output back to the network.

```python
# Core logic for core/agent.py
class MeshAgent:
    def __init__(self, node_id: int, tool: str, adapter, port: int):
        self.id = node_id
        self.tool_name = tool
        self.adapter = adapter
        self.port = port
        self.ws = f"workspaces/{self.id}"
        Path(self.ws).mkdir(parents=True, exist_ok=True)
        self.engine = Engine(llm=LLM(model="gemma-7b-it"),
                             tools=[self._register_tool(tool)],
                             memory_path=f"{self.ws}/memory.json",
                             on_token=self._count_token)
        self.token_used = 0

    def _count_token(self, _): self.token_used += 1

    def _register_tool(self, name: str):
        # Tool registration logic (e.g., using @tool decorator)
        # ... (details omitted for brevity, matches prompt)
        pass

    async def serve(self):
        """Starts the agent's server to listen for messages."""
        server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        async with server:
            await server.serve_forever()

    async def _handle(self, r, w):
        """Callback to handle a new connection from the network."""
        while True:
            size = int.from_bytes(await r.readexactly(4), "big")
            packet = self.adapter.decode(await r.readexactly(size))
            asyncio.create_task(self._process(packet, w))

    async def _process(self, pkt: dict, writer):
        """The core agent logic loop: filter, process, emit."""
        self.token_used = 0
        log_path = f"{self.ws}/agent.log"
        # Type-based filtering and processing logic
        # ... (details omitted, matches the business logic in the prompt)
        # Example for the 'reason' agent:
        if pkt["type"] == "triples" and self.tool_name == "reason":
            ans = await self.engine.run("reason final", triples=pkt["triples"])
            await self._emit(writer, "data_event",
                             {"tag": "final_answer", "payload": ans,
                              "log_uri": f"{self.ws}/reason.log"})

    async def _emit(self, w, pkt_type: str, extra: dict):
        """Encodes and sends a packet to the network."""
        pkt = {"type": pkt_type, **extra, "token_used": self.token_used}
        blob = self.adapter.encode(pkt)
        w.write(len(blob).to_bytes(4, "big") + blob)
        await w.drain()
        # Log the emitted packet
        with open(f"{self.ws}/agent.log", "a") as f:
            f.write(json.dumps(pkt, ensure_ascii=False) + "\n")
```

---

## **5. Evaluation and Metrics**

Evaluation is triggered automatically by the `MeshNetwork` upon receiving a packet with `tag: "final_answer"`.

#### **Evaluation Runner**

The default evaluation script measures quality based on a combination of Exact Match (EM) and ROUGE-L score against a ground truth file. This function is designed to be modular and can be replaced with other scoring mechanisms (e.g., BLEU, GAIA's official scorer).

```python
# Logic for scripts/evaluate.py
async def eval_runner(pred: str, truth_path: str) -> dict:
    """Calculates quality scores for the final answer."""
    with open(truth_path) as f:
        truth = json.load(f)
    em = int(pred.strip() == truth["answer"].strip())
    rouge = rouge_l(pred, truth["answer"])
    return {
        "quality_score": (em + rouge) / 2,
        "exact_match": em,
        "rouge_l": rouge
    }
```

#### **Metrics Report**

The final output includes `metrics.json`, containing both performance and quality metrics:
*   **Performance**: `bytes_tx`, `bytes_rx`, `pkt_cnt`, `header_overhead`, `token_sum`, `elapsed_ms`.
*   **Quality**: `quality_score`, `exact_match`, `rouge_l`.

---

## **6. Startup & Usage**

The entire system is launched via a single script that initializes the adapter, agents, and the network.

```python
# Example from scripts/run_benchmark.py
# 1. Initialize Protocol Adapter
adapter = JsonAdapter()

# 2. Define Agent & Network Configuration
ports  = [9000, 9001, 9002, 9003]
tools  = ["search", "extract", "triple", "reason"]

# 3. Launch Agent Servers Asynchronously
for idx, (p, t) in enumerate(zip(ports, tools)):
    agent = MeshAgent(node_id=idx, tool=t, adapter=adapter, port=p)
    asyncio.create_task(agent.serve())

# 4. Launch and Run the Network Controller
net = MeshNetwork(ports, adapter)
await net.start()

# 5. Kick off the process and wait for completion
await net.broadcast_init(Path("gaia_doc.txt").read_text())
await net._monitor_done() # This will block until evaluation is complete

print("✅ DONE → See metrics.json and run_artifacts.tar.gz")
```

---

## **7. Extensibility Points**

The architecture is designed to be extensible in several key areas without requiring modification of the core logic.

| Direction | Hook / File | Description |
| --- | --- | --- |
| **New Protocol** | Implement a new `ProtocolAdapter` subclass. | Add header compression, CRC checksums, or signature verification. |
| **Scheduling Strategy** | Modify `_process()` logic in `MeshAgent`. | Implement more advanced topic subscription, parallel RPC calls, or a learned communication graph instead of broadcasting. |
| **LLM/Tool Runtime** | Replace the `Engine` or `LLM` in `MeshAgent`. | Swap `gemma-7b` for `llama-3-70b` or a different tool execution engine. The token counter provides a standard interface. |
| **Security** | Enhance `adapter.encode()`/`decode()`. | Add an `Ed25519` signature to packets and verify them at the network or agent level. |
| **Fault Tolerance** | Add `try...except` in `MeshNetwork._relay()`. | Handle `ConnectionResetError` to automatically deregister dead nodes and allow the system to continue functioning. |
| **New Tools** | Add definitions to `tools/registry.py`. | Define new functions that can be assigned to agents. |