"""
Complete prompt template for ProtoRouter - protocol selection for multi-agent systems.
"""

PROTOCOL_SELECTION_PROMPT = """You are "ProtoRouter", a deterministic and evaluation-friendly protocol selector for multi-agent systems.
Your job: For each agent in a scenario, pick exactly ONE protocol from {{A2A, ACP, Agora, ANP}} that best matches the agent's requirements.
You must justify choices with transparent, metric-level reasoning and produce machine-checkable JSON only.

--------------------------------------------
1) Canonical Feature Model (authoritative; use this only)
--------------------------------------------
A2A (Agent-to-Agent Protocol)
- Transport/Model: HTTP + JSON-RPC + SSE; first-class long-running tasks; task/artifact lifecycle.
- Capability/UX: Multimodal messages (text/audio/video) and explicit UI capability negotiation.
- Discovery: Agent Card (capability advertisement) with ability → endpoint linkage.
- Security/Trust: Enterprise-style authN/Z; NOT end-to-end encryption by default (E2E optional via outer layers).
- Integration: Complements MCP (tools/data); broad vendor ecosystem; high feature richness.
- Typical Strengths: enterprise integration, complex workflows, multimodal streaming, UI handshakes, long jobs.
- Typical Costs: spec breadth → higher learning/ops complexity; cross-org privacy needs extra layers.
- Primary orientation: sustained agent-to-agent interaction and lightweight turn-taking.
- Less suited: scenarios dominated by resource/state-machine style operations and bulk archival/ingestion pipelines.

ACP (Agent Communication Protocol)
- Transport/Model: REST-first over HTTP; MIME-based multimodality; async-first with streaming support.
- Discovery: Agent Manifest & offline discovery options; clear single/multi-server topologies.
- Security/Trust: Relies on web auth patterns; E2E not native.
- Integration: Minimal SDK expectations; straightforward REST exposure.
- Typical Strengths: simplicity, REST familiarity, deployment flexibility, easy wrapping of existing services.
- Typical Costs: less emphasis on UI capability negotiation; feature focus trending toward A2A in newer stacks.
- Primary orientation: structured, addressable operations with clear progress semantics and repeatable handling at scale.
- Less suited: ultra-light conversational micro-turns where resource/state semantics are explicitly avoided.

Agora (Meta-Protocol)
- Positioning: Minimal "meta" wrapper; sessions carry a protocolHash binding to a plain-text protocol doc.
- Discovery: /.wellknown returns supported protocol hashes; natural language is a fallback channel.
- Evolution: Encourages reusable "routines"; fast protocol evolution and heterogeneity tolerance.
- Security/Trust: No strong identity/E2E built-in; depends on deployment or upper layers.
- Typical Strengths: lightweight, negotiation-friendly, highly adaptable for research/decentralized experiments.
- Typical Costs: governance/audit features not built-in; production-grade security must be composed.
- Primary orientation: explicit procedure governance — selecting and following a concrete routine/version that must be auditable.
- Less suited: when no concrete procedure/version needs to be fixed or referenced.

ANP (Agent Network Protocol)
- Positioning: Network & trust substrate for agents; three layers: identity+E2E, meta-protocol, application protocols.
- Security/Trust: W3C DID-based identities; ECDHE-based end-to-end encryption; cross-org/verifiable comms.
- Discovery/Semantics: Descriptions for capabilities & protocols; supports multi-topology communications.
- Typical Strengths: strong identity, E2E privacy, cross-organization trust.
- Typical Costs: DID/keys lifecycle adds integration/ops complexity; ecosystem still maturing; UI/multimodal not first-class.
- Primary orientation: relationship assurance and information protection across boundaries (identity, confidentiality, non-repudiation).
- Less suited: purely local/benign traffic where verifiable identity and confidentiality are not primary concerns.


--------------------------------------------
3) Protocol Selection Task
--------------------------------------------

**Scenario Description:**
{scenario_description}

**Module Details:**
{module_details}

**Your Task:**
For each module in this scenario, you must select exactly ONE protocol from {{A2A, ACP, Agora, ANP}} that best matches the module's requirements.

You must respond using the protocol_selection function call with your analysis and selections."""


def format_scenario_prompt(scenario_data):
    """Format the scenario data into the prompt template"""
    
    scenario_description = scenario_data.get("description", "")
    modules = scenario_data.get("module", [])
    
    # Format module details
    module_details_parts = []
    for i, module in enumerate(modules, 1):
        module_name = module.get("name", f"Module-{i}")
        agents = module.get("agents", [])
        tasks = module.get("tasks", [])
        potential_issues = module.get("potential_issues", [])
        protocol_selection = module.get("protocol_selection", {})
        
        choices = protocol_selection.get('choices', ['A2A', 'ACP', 'Agora', 'ANP'])
        choices_str = ', '.join(choices) if isinstance(choices, list) else str(choices)
        
        module_detail = f"""
**Module {i}: {module_name}**
- Agents: {', '.join(agents)}
- Protocol Selection: Choose {protocol_selection.get('select_exactly', 1)} protocol(s) from {{{choices_str}}}

**Tasks:**
{chr(10).join(f"- {task}" for task in tasks)}

**Potential Issues:**
{chr(10).join(f"- {issue}" for issue in potential_issues)}
"""
        module_details_parts.append(module_detail.strip())
    
    module_details = "\n\n".join(module_details_parts)
    
    return PROTOCOL_SELECTION_PROMPT.format(
        scenario_description=scenario_description,
        module_details=module_details
    )


# Function definition for LLM tool calling
PROTOCOL_SELECTION_FUNCTION = {
    "name": "protocol_selection",
    "description": "Select protocols for each module in the scenario",
    "parameters": {
        "type": "object",
        "properties": {
            "module_selections": {
                "type": "array",
                "description": "Protocol selections for each module",
                "items": {
                    "type": "object",
                    "properties": {
                        "module_id": {
                            "type": "integer",
                            "description": "Module number (1, 2, 3, etc.)"
                        },
                        "selected_protocol": {
                            "type": "string",
                            "enum": ["A2A", "ACP", "Agora", "ANP"],
                            "description": "The selected protocol for this module"
                        },
                        "justification": {
                            "type": "string",
                            "description": "Detailed justification for the protocol selection"
                        }
                    },
                    "required": ["module_id", "selected_protocol", "justification"]
                },
                "maxItems": 5
            }
        },
        "required": ["module_selections"]
    }
}
