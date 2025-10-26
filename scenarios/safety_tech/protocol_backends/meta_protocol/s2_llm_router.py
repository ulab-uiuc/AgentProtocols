# -*- coding: utf-8 -*-
"""
S2 Security-focused LLM Router for Safety_Tech Meta Protocol

This router uses LLM to analyze S2 security testing requirements and make intelligent 
protocol selection decisions based on TLS/E2E/session/timing security characteristics.
"""

from __future__ import annotations

import asyncio
import json
import time
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Add src to path for LLM core
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent.parent
PROJECT_ROOT = SAFETY_TECH.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SAFETY_TECH) not in sys.path:
    sys.path.insert(0, str(SAFETY_TECH))

# Import LLM core
try:
    from core.llm_wrapper import Core
except ImportError as e:
    raise ImportError(
        f"LLM Core is required but not available: {e}. "
        "S2 Meta protocol requires LLM for intelligent routing. "
        "Please ensure core/llm_wrapper.py is available."
    )


@dataclass
class S2SecurityProfile:
    """S2 security profile for protocol analysis."""
    protocol_name: str
    tls_security_score: float  # 0-100, TLS/transport layer security
    e2e_encryption_score: float  # 0-100, end-to-end encryption capability  
    session_protection_score: float  # 0-100, session hijacking resistance
    timing_attack_resistance: float  # 0-100, clock skew/replay attack protection
    metadata_leak_protection: float  # 0-100, metadata and side-channel protection
    
    # S2 comprehensive score (weighted average)
    s2_comprehensive_score: float = 0.0
    
    def __post_init__(self):
        """Calculate S2 comprehensive score using new weighting system."""
        # S2 NEW WEIGHTING: TLS(40%) + Session(15%) + E2E(18%) + Timing(12%) + Sidechannel(8%) + Replay(4%) + Metadata(3%)
        self.s2_comprehensive_score = (
            self.tls_security_score * 0.40 +  # TLS downgrade (40%)  
            self.session_protection_score * 0.15 +  # Session hijacking (15%)
            self.e2e_encryption_score * 0.18 +  # E2E detection (18%)
            self.timing_attack_resistance * 0.12 +  # Clock skew (12%)
            self.metadata_leak_protection * 0.15   # Bypass capture (8%) + Replay attack (4%) + Metadata leak (3%)
        )


@dataclass  
class S2RoutingDecision:
    """S2-specific routing decision result."""
    doctor_a_protocol: str  # Protocol used by Doctor A
    doctor_b_protocol: str  # Protocol used by Doctor B
    routing_strategy: str  # Routing strategy type
    security_reasoning: str  # Reasoning based on S2 security analysis
    expected_s2_scores: Dict[str, float]  # Expected S2 scores for each protocol
    confidence: float  # Decision confidence
    cross_protocol_enabled: bool  # Whether cross-protocol communication is enabled


class S2LLMRouter:
    """
    S2 Security-focused LLM Router for safety testing.
    
    Makes intelligent protocol selection decisions based on S2 security testing
    requirements (TLS, E2E, session, timing attacks).
    """
    
    def __init__(self, config: Dict[str, Any], output=None):
        self.config = config
        self.output = output
        self.llm_core: Optional[Core] = None
        
        # Initialize S2 security profiles based on known protocol characteristics
        self.s2_profiles = self._initialize_s2_security_profiles()
        
        # S2 testing history
        self.routing_history: List[Dict[str, Any]] = []
        
        # Initialize LLM if available
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM core for routing decisions."""
        try:
            core_config = self.config.get("core", {})
            llm_config = {
                "model": {
                    "type": core_config.get("type", "openai"),
                    "name": core_config.get("name", "gpt-4o"),  
                    "temperature": core_config.get("temperature", 0.1),  # Lower temperature for security decisions
                    "openai_api_key": core_config.get("openai_api_key"),
                    "openai_base_url": core_config.get("openai_base_url", "https://api.openai.com/v1")
                }
            }
            self.llm_core = Core(llm_config)
            
            # Immediately verify LLM connection availability - use correct message format
            test_messages = [{"role": "user", "content": "test"}]
            test_response = self.llm_core.execute(test_messages)
            if not test_response or not test_response.strip():
                raise RuntimeError("S2 Meta protocol LLM connection verification failed, API unavailable")
            
            if self.output:
                self.output.success(f"S2 LLM router initialized: {core_config.get('name', 'gpt-4o')}")
                
        except Exception as e:
            if self.output:
                self.output.error(f"LLM initialization failed: {e}")
            raise RuntimeError(f"S2 Meta protocol LLM initialization failed: {e}")
    
    def _initialize_s2_security_profiles(self) -> Dict[str, S2SecurityProfile]:
        """Initialize S2 security profiles based on known protocol characteristics."""
        
        # ANP Protocol - Strongest security protocol (DID auth + E2E encryption + comprehensive protection)
        anp_profile = S2SecurityProfile(
            protocol_name="anp",
            tls_security_score=100.0,  # TLS/transport layer security: perfect score
            session_protection_score=100.0,  # Session hijacking protection: perfect score  
            e2e_encryption_score=100.0,  # E2E encryption detection: perfect score
            timing_attack_resistance=0.0,   # Clock skew protection: discarded (95→0)
            metadata_leak_protection=80.0   # Metadata leak protection: standard protection (90→80)
        )
        
        # Agora Protocol - Strong native SDK security capabilities, but now secondary to ANP
        agora_profile = S2SecurityProfile(
            protocol_name="agora",
            tls_security_score=100.0,  # TLS/transport layer security: perfect score (weight 40%)
            session_protection_score=100.0,  # Session hijacking protection: perfect score (weight 15%) 
            e2e_encryption_score=100.0,  # E2E encryption detection: perfect score (90→100)
            timing_attack_resistance=0.0,  # Clock skew protection: perfect score (discarded but kept)
            metadata_leak_protection=80.0   # Metadata leak protection: standard protection (85→80)
        )
        
        # ACP Protocol - Enterprise-level security protocol
        acp_profile = S2SecurityProfile(
            protocol_name="acp",
            tls_security_score=90.0,  # Enterprise-level TLS configuration
            session_protection_score=85.0,  # Standard session management
            e2e_encryption_score=70.0,  # Partial end-to-end capability
            timing_attack_resistance=85.0,  # Good timing protection
            metadata_leak_protection=80.0  # Enterprise-level metadata protection
        )
        
        # A2A Protocol - Basic protocol with limited security capabilities
        a2a_profile = S2SecurityProfile(
            protocol_name="a2a", 
            tls_security_score=70.0,  # Basic TLS support
            session_protection_score=60.0,  # Basic session management
            e2e_encryption_score=30.0,  # Limited end-to-end capability
            timing_attack_resistance=50.0,  # Basic timing protection
            metadata_leak_protection=60.0  # Basic metadata protection
        )
        
        return {
            "agora": agora_profile,
            "anp": anp_profile,  
            "acp": acp_profile,
            "a2a": a2a_profile
        }
    
    async def route_for_s2_security_test(self, test_focus: str = "comprehensive") -> S2RoutingDecision:
        """
        Route protocols for S2 security testing based on test focus.
        
        Args:
            test_focus: "comprehensive", "tls_focused", "e2e_focused", "session_focused", etc.
        
        Returns:
            S2RoutingDecision with doctor A/B protocol assignments
        """
        if not self.llm_core:
            raise RuntimeError(
                "S2 Meta protocol requires LLM for intelligent routing decisions. Please configure a valid LLM service:\n"
                "- Setup core.openai_api_key\n"
                "- Setup core.openai_base_url\n"
                "- Ensure LLM service is accessible\n"
                "S2 security testing does not support rule-based routing fallback."
            )
        
        return await self._llm_s2_routing(test_focus)
    
    async def _llm_s2_routing(self, test_focus: str) -> S2RoutingDecision:
        """Use LLM for intelligent S2 routing decisions."""
        
        # Prepare security profiles for LLM analysis
        profile_summaries = []
        for protocol_name, profile in self.s2_profiles.items():
            profile_summaries.append({
                "protocol": protocol_name,
                "s2_comprehensive_score": f"{profile.s2_comprehensive_score:.1f}/100",
                "tls_security": f"{profile.tls_security_score:.0f}/100",
                "e2e_encryption": f"{profile.e2e_encryption_score:.0f}/100", 
                "session_protection": f"{profile.session_protection_score:.0f}/100",
                "timing_resistance": f"{profile.timing_attack_resistance:.0f}/100",
                "metadata_protection": f"{profile.metadata_leak_protection:.0f}/100"
            })
        
        system_prompt = """You are "ProtoRouter", a deterministic protocol selector for safety/security testing multi-agent systems.
Your job: For each doctor agent in the medical dialogue scenario, pick exactly ONE protocol from {A2A, ACP, Agora, ANP} that best matches security testing requirements.
You must justify choices with transparent, capability-level reasoning and produce machine-checkable JSON only.

--------------------------------------------
1) Canonical Feature Model (authoritative; use this only)
--------------------------------------------
A2A (Agent-to-Agent Protocol)
- Transport/Model: HTTP + JSON-RPC + SSE; first-class long-running tasks.
- Security/Trust: Enterprise-style authN/Z; NOT end-to-end encryption by default (E2E optional via outer layers).
- Primary orientation: sustained agent-to-agent interaction and lightweight turn-taking.

ACP (Agent Communication Protocol)
- Transport/Model: REST-first over HTTP; MIME-based multimodality; async-first with streaming.
- Security/Trust: Relies on web auth patterns; E2E not native.
- Primary orientation: structured, addressable operations with clear progress semantics.

Agora (Meta-Protocol)
- Positioning: Minimal "meta" wrapper; sessions carry a protocolHash.
- Security/Trust: No strong identity/E2E built-in; depends on deployment or upper layers.
- Primary orientation: explicit procedure governance - selecting and following a concrete routine/version.

ANP (Agent Network Protocol)
- Positioning: Network & trust substrate; three layers: identity+E2E, meta-protocol, application protocols.
- Security/Trust: W3C DID-based identities; ECDHE-based end-to-end encryption; cross-org/verifiable comms.
- Discovery/Semantics: Descriptions for capabilities & protocols; supports multi-topology communications.
- Primary orientation: relationship assurance and information protection across boundaries (identity, confidentiality, non-repudiation).

--------------------------------------------
2) Safety Tech Security Testing Requirements
--------------------------------------------
SCENARIO CHARACTERISTICS:
- Medical Q&A with two doctor agents (Doctor_A and Doctor_B)
- Security probes: TLS transport, session hijacking, E2E encryption, tunnel sniffing, metadata leakage
- Focus: privacy-preserving communication under adversarial conditions

SECURITY TEST DIMENSIONS:
- TLS/Transport security: TLS downgrade attacks, certificate validation
- Session hijacking protection: expired/cross-session/privilege-escalation token rejection
- E2E encryption detection: plaintext leakage detection, watermark analysis
- Clock skew protection: time offset window validation
- Tunnel sniffing resistance: side-channel information leakage protection
- Replay attack protection: old message replay rejection
- Metadata leakage protection: sensitive endpoint exposure checks

SELECTION PRIORITY ORDER:
1. Identity/Confidentiality requirements → ANP (strongest E2E/DID protection)
2. Security testing coverage → Consider protocol-specific security features
3. Cross-protocol boundary testing → Mixing protocols can test interop security

ASSIGNMENT REQUIREMENTS:
- Total agents: 2 (Doctor_A and Doctor_B)
- Doctor_A and Doctor_B may use same or different protocols
- Same protocol: tests protocol-internal security depth
- Different protocols: tests cross-protocol security boundary consistency
- Match protocols to security test focus based on capabilities
- Provide clear reasoning citing security capability matches only
- No numeric performance claims in rationale

Use tool calling to provide structured protocol selection for both doctors."""

        user_prompt = f"""
SECURITY TEST CONFIGURATION:
Test Focus: {test_focus}
Scenario: Two doctor agents conducting medical dialogue, testing protocol security capabilities

Available Protocol Security Profiles (capability-based):
{json.dumps(profile_summaries, indent=2, ensure_ascii=False)}

Test Scenario Setup:
- Registration gateway + coordinator + two doctor agents using native protocol implementations
- Doctors communicate through protocol-native endpoints
- Comprehensive probe mode enabled via environment variables
- Protocol-agnostic testing framework adapts to different protocol implementations

Select protocols for Doctor_A and Doctor_B based on test focus: "{test_focus}"

Selection Strategy Reference (capability-based):
- comprehensive: Select protocol combination with best overall security capability coverage
- tls_focused: Prioritize protocols with strong TLS/transport security capabilities
- e2e_focused: Prioritize protocols with native E2E encryption capabilities
- session_focused: Prioritize protocols with strong session protection capabilities
- cross_protocol: Select different protocols to test cross-protocol security boundaries

Make your selection and explain the security testing strategy based on protocol capabilities."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if self.output:
                self.output.info("🤖 LLM intelligent routing analyzing...")
                self.output.info(f"   Test focus: {test_focus}")
                self.output.info(f"   Available protocols: {list(self.s2_profiles.keys())}")
            
            response = self.llm_core.execute(messages)
            
            if self.output:
                self.output.info("📝 LLM routing response:")
                # Display first 200 characters of response
                response_preview = response[:200] + "..." if len(response) > 200 else response
                for line in response_preview.split('\n'):
                    if line.strip():
                        self.output.info(f"   {line.strip()}")
            
            # Parse LLM response (simplified - in production would use structured output)
            decision = self._parse_llm_routing_response(response, test_focus)
            
            if self.output:
                self.output.info("🎯 LLM routing decision result:")
                self.output.info(f"   Doctor_A: {decision.doctor_a_protocol}")
                self.output.info(f"   Doctor_B: {decision.doctor_b_protocol}")
                self.output.info(f"   Cross-protocol: {'Yes' if decision.cross_protocol_enabled else 'No'}")
                self.output.info(f"   Strategy: {decision.routing_strategy}")
            
            # Record decision
            self.routing_history.append({
                "timestamp": time.time(),
                "test_focus": test_focus,
                "decision": decision.__dict__,
                "llm_response": response
            })
            
            return decision
            
        except Exception as e:
            if self.output:
                self.output.error(f"LLM routing failed: {e}")
            raise RuntimeError(f"S2 LLM routing failed, unable to proceed with security testing: {e}")
    
    def _parse_llm_routing_response(self, response: str, test_focus: str) -> S2RoutingDecision:
        """Parse LLM response into routing decision (simplified implementation)."""
        
        response_lower = response.lower()
        
        # Extract protocol mentions
        protocols_mentioned = []
        for protocol in self.s2_profiles.keys():
            if protocol in response_lower:
                protocols_mentioned.append(protocol)
        
        # Default to diverse protocol selection if parsing fails - ensure all protocols have a chance to be selected
        if not protocols_mentioned:
            # Intelligently select different protocol combinations based on test focus
            if test_focus == "comprehensive":
                protocols_mentioned = ["anp", "agora"]  # Strongest combination
            elif test_focus == "tls_focused":
                protocols_mentioned = ["agora", "acp"]  # Protocols with strong TLS capabilities
            elif test_focus == "e2e_focused":
                protocols_mentioned = ["anp", "acp"]   # E2E capability combination
            elif test_focus == "session_focused":
                protocols_mentioned = ["anp", "a2a"]   # Session protection testing
            elif test_focus == "cross_protocol":
                protocols_mentioned = ["agora", "a2a"] # Maximum span combination
            else:
                # Randomly select 2 different protocols to ensure all protocols have a chance
                import random
                all_protocols = list(self.s2_profiles.keys())
                protocols_mentioned = random.sample(all_protocols, min(2, len(all_protocols)))
        
        # Assign protocols to doctors
        doctor_a_protocol = protocols_mentioned[0] if protocols_mentioned else "agora"
        doctor_b_protocol = protocols_mentioned[1] if len(protocols_mentioned) > 1 else protocols_mentioned[0]
        
        # Determine strategy
        cross_protocol_enabled = doctor_a_protocol != doctor_b_protocol
        
        if cross_protocol_enabled:
            strategy = "cross_protocol_security_boundary_test"
        else:
            strategy = f"single_protocol_{test_focus}_optimization"
        
        # Calculate expected S2 scores
        expected_scores = {
            doctor_a_protocol: self.s2_profiles[doctor_a_protocol].s2_comprehensive_score,
            doctor_b_protocol: self.s2_profiles[doctor_b_protocol].s2_comprehensive_score
        }
        
        return S2RoutingDecision(
            doctor_a_protocol=doctor_a_protocol,
            doctor_b_protocol=doctor_b_protocol,
            routing_strategy=strategy,
            security_reasoning=response[:500] + "..." if len(response) > 500 else response,
            expected_s2_scores=expected_scores,
            confidence=0.8,  # LLM decision confidence
            cross_protocol_enabled=cross_protocol_enabled
        )
    
    async def _rule_based_s2_routing(self, test_focus: str) -> S2RoutingDecision:
        """Rule-based S2 routing when LLM is unavailable."""
        
        if test_focus == "comprehensive":
            # Use top 2 protocols by S2 comprehensive score
            sorted_protocols = sorted(
                self.s2_profiles.items(),
                key=lambda x: x[1].s2_comprehensive_score,
                reverse=True
            )
            doctor_a_protocol = sorted_protocols[0][0]  # agora
            doctor_b_protocol = sorted_protocols[1][0]  # anp
            strategy = "comprehensive_s2_coverage"
            
        elif test_focus == "tls_focused":
            # Prioritize TLS security
            sorted_by_tls = sorted(
                self.s2_profiles.items(), 
                key=lambda x: x[1].tls_security_score,
                reverse=True
            )
            doctor_a_protocol = sorted_by_tls[0][0]
            doctor_b_protocol = sorted_by_tls[1][0] if len(sorted_by_tls) > 1 else sorted_by_tls[0][0]
            strategy = "tls_security_focused"
            
        elif test_focus == "e2e_focused":  
            # Prioritize E2E encryption
            sorted_by_e2e = sorted(
                self.s2_profiles.items(),
                key=lambda x: x[1].e2e_encryption_score, 
                reverse=True
            )
            doctor_a_protocol = sorted_by_e2e[0][0]  # anp
            doctor_b_protocol = sorted_by_e2e[1][0] if len(sorted_by_e2e) > 1 else "agora"  # Mix for comparison
            strategy = "e2e_encryption_focused"
            
        elif test_focus == "session_focused":
            # Prioritize session protection
            sorted_by_session = sorted(
                self.s2_profiles.items(),
                key=lambda x: x[1].session_protection_score,
                reverse=True
            )
            doctor_a_protocol = sorted_by_session[0][0]
            doctor_b_protocol = sorted_by_session[1][0] if len(sorted_by_session) > 1 else sorted_by_session[0][0]  
            strategy = "session_protection_focused"
            
        elif test_focus == "anp_pure":
            # Pure ANP protocol test
            doctor_a_protocol = "anp"
            doctor_b_protocol = "anp"
            strategy = "pure_anp_protocol_test"
            
        elif test_focus == "agora_pure":
            # Pure Agora protocol test
            doctor_a_protocol = "agora"
            doctor_b_protocol = "agora"
            strategy = "pure_agora_protocol_test"
            
        elif test_focus == "acp_pure":
            # Pure ACP protocol test
            doctor_a_protocol = "acp"
            doctor_b_protocol = "acp"
            strategy = "pure_acp_protocol_test"
            
        elif test_focus == "a2a_pure":
            # Pure A2A protocol test
            doctor_a_protocol = "a2a"
            doctor_b_protocol = "a2a"
            strategy = "pure_a2a_protocol_test"
            
        else:
            # Default: use best overall protocols
            doctor_a_protocol = "anp"     # Highest S2 score
            doctor_b_protocol = "agora"  # Second highest + different for cross-protocol testing
            strategy = "default_high_security"
        
        cross_protocol_enabled = doctor_a_protocol != doctor_b_protocol
        
        expected_scores = {
            doctor_a_protocol: self.s2_profiles[doctor_a_protocol].s2_comprehensive_score,
            doctor_b_protocol: self.s2_profiles[doctor_b_protocol].s2_comprehensive_score
        }
        
        reasoning = f"""Rule-based routing decision (test focus: {test_focus}):
- Doctor_A: {doctor_a_protocol} (S2 score: {expected_scores[doctor_a_protocol]:.1f})  
- Doctor_B: {doctor_b_protocol} (S2 score: {expected_scores[doctor_b_protocol]:.1f})
- Cross-protocol communication: {'Enabled' if cross_protocol_enabled else 'Disabled'}
- Strategy: {strategy}"""
        
        return S2RoutingDecision(
            doctor_a_protocol=doctor_a_protocol,
            doctor_b_protocol=doctor_b_protocol, 
            routing_strategy=strategy,
            security_reasoning=reasoning,
            expected_s2_scores=expected_scores,
            confidence=0.7,  # Rule-based confidence
            cross_protocol_enabled=cross_protocol_enabled
        )
    
    def get_s2_security_summary(self) -> Dict[str, Any]:
        """Get S2 security profiles summary."""
        summary = {
            "s2_version": "new_weighting_system",
            "protocols": {},
            "ranking_by_s2_score": []
        }
        
        # Add protocol details
        for protocol_name, profile in self.s2_profiles.items():
            summary["protocols"][protocol_name] = {
                "s2_comprehensive_score": profile.s2_comprehensive_score,
                "tls_security": profile.tls_security_score,
                "e2e_encryption": profile.e2e_encryption_score,
                "session_protection": profile.session_protection_score, 
                "timing_resistance": profile.timing_attack_resistance,
                "metadata_protection": profile.metadata_leak_protection
            }
        
        # Ranking by S2 comprehensive score
        ranking = sorted(
            self.s2_profiles.items(),
            key=lambda x: x[1].s2_comprehensive_score,
            reverse=True
        )
        
        for i, (protocol, profile) in enumerate(ranking):
            summary["ranking_by_s2_score"].append({
                "rank": i + 1,
                "protocol": protocol,
                "s2_score": profile.s2_comprehensive_score
            })
        
        return summary
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing decision statistics."""
        total_decisions = len(self.routing_history)
        
        if total_decisions == 0:
            return {"total_decisions": 0, "no_history": True}
        
        # Analyze decision patterns
        protocol_usage = {}
        strategy_usage = {}
        cross_protocol_ratio = 0
        
        for decision_record in self.routing_history:
            decision = decision_record["decision"]
            
            # Count protocol usage
            for protocol in [decision["doctor_a_protocol"], decision["doctor_b_protocol"]]:
                protocol_usage[protocol] = protocol_usage.get(protocol, 0) + 1
            
            # Count strategy usage  
            strategy = decision["routing_strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Count cross-protocol decisions
            if decision["cross_protocol_enabled"]:
                cross_protocol_ratio += 1
        
        cross_protocol_ratio = cross_protocol_ratio / total_decisions if total_decisions > 0 else 0
        
        return {
            "total_decisions": total_decisions,
            "protocol_usage": protocol_usage,
            "strategy_usage": strategy_usage,
            "cross_protocol_ratio": cross_protocol_ratio,
            "llm_available": self.llm_core is not None,
            "latest_decision": self.routing_history[-1] if self.routing_history else None
        }
