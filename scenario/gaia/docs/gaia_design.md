# GAIA Design Overview

This document provides a polished and expanded description of the GAIA scenario design. The text below rewrites and refines the seven items provided by the author and then adds additional design items starting from item 8 where appropriate. The writing follows academic English conventions and emphasizes reproducibility, measurement, and modularity.

1. Planner module: programmatic generation of agent and network configurations

- The planner module uses large language models (LLMs) to synthesize an experimental plan that includes: agent configurations (role, toolset, and LLM prompt templates), tool-call metadata (interfaces, allowed arguments, and expected outputs), and the network configuration (topology and explicit workflow/message-flow definition). The planner encodes design decisions that are shared across all evaluated protocols to ensure fair comparisons.
- Agent count and scale are recommended explicitly via prompting schemas: e.g., discrete difficulty levels map to agent counts (level 1 -> 2 agents, level 2 -> 4 agents, level 3 -> 6 agents). The planner records the mapping and the prompting seed to enable reproducibility.
- The planner emits machine-readable manifests (JSON) that are consumed by the network bootstrapper and test harness; manifests include protocol identifiers, resource constraints, and runtime parameters.

2. Agent lifecycle and peer-to-peer serving model

- Each agent is launched as a network participant that exposes both client and server endpoints (i.e., it can receive requests and initiate outbound calls). Agents follow the message flows defined in the planner manifest.
- Upon receipt of a message, an agent performs message handling that typically consists of: parsing input, running an internal reasoning module (LLM or deterministic code), invoking its tools, and producing an assistant response. The network layer is responsible for reliable delivery to the next hop(s) as defined by the planned workflow.

3. Step-based network memory

- The network implements a step-based memory pool that records both user messages and assistant responses in structured JSON. Each step execution record contains metadata (step index, agent id, timestamps, execution status, and tool-call traces) together with the message payloads.
- Memory is append-only during execution and can be exported for offline analysis, replay, or summarization. The memory schema is versioned to ensure compatibility across experiments.

4. LLM-based summarization and evaluation pipeline

- At the completion of a workflow, the system uses an LLM-based summarizer to produce a concise representation of the execution and the outcome based on the stored memory pool. The summarization prompt is standardized across runs to ensure comparability.
- A separate LLM judge performs quality assessment on the summarizer output and/or the primary result. The judge evaluates whether the task was solved, assigns quality scores (accuracy, relevance, completeness), and annotates failure modes.
- The evaluation pipeline records additional resource metrics (time consumption, token usage) that are included in the final evaluation report.

5. Sandboxed tool execution and environment isolation

- The system provides a sandboxed execution environment for running code and third-party tools invoked by agents. Isolation ensures that individual agent tool-calls cannot interfere with the host environment or with other agents' workspaces.
- Sandboxing covers package dependency isolation (virtual environments or containers), filesystem and network access controls, and resource limits (CPU, memory, wall time). Logs and artifacts produced inside the sandbox are captured and associated with the corresponding step execution.

6. Fine-grained time accounting

- Time is recorded at multiple granularities: per-agent start/stop timestamps, per-step start/end timestamps, and end-to-end workflow duration. These measurements allow latency profiling, detection of stragglers, and correlation with resource consumption.
- Timestamps are captured in a consistent epoch (e.g., milliseconds since Unix epoch) and included in produced artifacts for downstream analysis.

7. LLM-driven adjudication

- The LLM judge ingests the full execution log and the predicted (or generated) answer to assess quality. The judge's evaluation is driven by structured prompts and rubric criteria (e.g., factual accuracy, task alignment, brevity, and justification of steps).
- Judgements are captured as structured metadata (pass/fail, quality scores) and appended to the experiment artifacts.


8.  Metrics, instrumentation, and reporting

- The evaluation report includes performance metrics (success rate, token consumption), quality metrics (scores from the judge), and operational metrics (agent uptime, step success/failure rates, latency percentiles).
- Reports are emitted in machine-readable form (JSON) and a human-readable summary (plain text or HTML) for quick inspection.

9.  Experimental protocol design and fairness

- To ensure fair comparisons between protocols, the same planner-generated manifest is used across all protocol variants under test. Only the protocol implementation differs (e.g., routing or agent orchestration strategies).
- The planner also produces a canonical seed and configuration that control sources of variability.

References and schema notes

- Memory pool and step-execution schemas are versioned; the design promotes forward compatibility for new analyzer and evaluator modules.

