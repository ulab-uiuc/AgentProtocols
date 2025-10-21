# -*- coding: utf-8 -*-
"""
Registration Attack Scenarios
Implementation of six registration attack scenarios used to test different
protocols' admission defense capabilities.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import httpx
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Attack result"""
    attack_type: str
    success: bool
    status_code: Optional[int] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    additional_info: Dict[str, Any] = None


class RegistrationAttackRunner:
    """Registration attack scenario runner

    Protocol-agnostic design:
    - protocol: target protocol name (e.g. 'agora', 'a2a', 'acp', 'anp')
    - attack_hooks: optional hooks to generate protocol-specific forged proofs/payloads
      keys may include:
        - build_impersonation_proof() -> Dict[str, Any]
        - build_endpoint_substitution_proof() -> Dict[str, Any]
        - build_cross_protocol_proof() -> Dict[str, Any]
        - build_observer_proof(observer_id) -> Dict[str, Any]
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rg_endpoint = config.get('rg_endpoint', 'http://127.0.0.1:8001')
        self.target_conversation_id = config.get('conversation_id', f'attack_test_{int(time.time())}')
        
        # Attack configuration
        self.attack_config = config.get('attacks', {})
        self.timeout = config.get('attack_timeout', 10.0)
        
        # Protocol and hooks
        self.protocol = config.get('protocol', 'agora')
        self.attack_hooks = config.get('attack_hooks', {}) or {}
        
        # Results storage
        self.attack_results: List[AttackResult] = []
        
    async def run_all_attacks(self) -> List[AttackResult]:
        """Run all six attack scenarios"""
        attacks = [
            self.impersonation_attack,
            self.stolen_credential_attack,
            self.replay_attack,
            self.endpoint_substitution_attack,
            self.cross_protocol_attack,
            self.observer_auto_admission_attack
        ]
        
        results = []
        for attack_func in attacks:
            try:
                result = await attack_func()
                results.append(result)
                logger.info(f"Attack {result.attack_type}: {'SUCCESS' if result.success else 'FAILED'}")
            except Exception as e:
                logger.error(f"Attack execution error: {e}")
                results.append(AttackResult(
                    attack_type=attack_func.__name__,
                    success=False,
                    error_message=str(e)
                ))
        
        self.attack_results = results
        return results
    
    async def impersonation_attack(self) -> AttackResult:
        """1. Impersonation attack"""
        start_time = time.time()
        
        try:
            # Try to register impersonating Doctor_B
            fake_registration = {
                "protocol": self.protocol,
                "agent_id": "Doctor_B",  # Impersonate a legitimate identity
                "endpoint": "http://fake-doctor-b:8000",
                "conversation_id": self.target_conversation_id,
                "role": "doctor_b",
                "protocolMeta": {
                    "protocol_version": "1.0",
                    "fake_metadata": True
                },
                "proof": (
                    self.attack_hooks.get('build_impersonation_proof')() if self.attack_hooks.get('build_impersonation_proof')
                    else {
                        "toolformer_signature": "fake_signature_12345",
                        "timestamp": time.time(),
                        "nonce": str(uuid.uuid4()),
                        "agent_signature": "fake_agent_signature"
                    }
                )
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=fake_registration,
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return AttackResult(
                    attack_type="impersonation",
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_data=response.json() if response.status_code == 200 else None,
                    error_message=response.text if response.status_code != 200 else None,
                    execution_time=execution_time,
                    additional_info={
                        "target_identity": "Doctor_B",
                        "fake_endpoint": fake_registration["endpoint"]
                    }
                )
                
        except Exception as e:
            return AttackResult(
                attack_type="impersonation",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def stolen_credential_attack(self) -> AttackResult:
        """2. Credential reuse (stolen credential) attack"""
        start_time = time.time()
        
        try:
            # First try to obtain a legitimate credential (simulate leak)
            legitimate_token = await self._simulate_credential_leak()
            
            # Use the stolen credential to register an Observer
            stolen_cred_request = {
                "agent_id": f"malicious_observer_{int(time.time())}",
                "conversation_id": self.target_conversation_id,
                "role": "observer",
                "endpoint": "http://malicious-observer:8000",
                "proof": {
                    "stolen_session_token": legitimate_token,
                    "timestamp": time.time(),
                    "observer_type": "credential_reuse_attack"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/subscribe",
                    json=stolen_cred_request,
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return AttackResult(
                    attack_type="stolen_credential",
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_data=response.json() if response.status_code == 200 else None,
                    error_message=response.text if response.status_code != 200 else None,
                    execution_time=execution_time,
                    additional_info={
                        "stolen_token": legitimate_token[:16] + "..." if legitimate_token else None,
                        "attack_agent_id": stolen_cred_request["agent_id"]
                    }
                )
                
        except Exception as e:
            return AttackResult(
                attack_type="stolen_credential",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def replay_attack(self) -> AttackResult:
        """3. Registration replay attack"""
        start_time = time.time()
        
        try:
            # Create old registration data (simulate a previous legitimate registration)
            old_timestamp = time.time() - 3600  # 1 hour ago
            old_registration = {
                "protocol": "agora",
                "agent_id": f"replay_victim_{int(old_timestamp)}",
                "endpoint": "http://old-endpoint:8000",
                "conversation_id": self.target_conversation_id,
                "role": "doctor",
                "protocolMeta": {
                    "protocol_version": "1.0",
                    "replayed": True
                },
                "proof": {
                    "toolformer_signature": f"old_signature_{int(old_timestamp)}",
                    "timestamp": old_timestamp,  # old timestamp
                    "nonce": str(uuid.uuid4()),
                    "agent_signature": f"old_agent_sig_{int(old_timestamp)}"
                }
            }
            
            # Replay the old registration data
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=old_registration,
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return AttackResult(
                    attack_type="replay",
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_data=response.json() if response.status_code == 200 else None,
                    error_message=response.text if response.status_code != 200 else None,
                    execution_time=execution_time,
                    additional_info={
                        "replayed_timestamp": old_timestamp,
                        "time_difference_hours": (time.time() - old_timestamp) / 3600,
                        "replayed_agent_id": old_registration["agent_id"]
                    }
                )
                
        except Exception as e:
            return AttackResult(
                attack_type="replay",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def endpoint_substitution_attack(self) -> AttackResult:
        """4. Endpoint substitution attack (directory poisoning)"""
        start_time = time.time()
        
        try:
            # Register a malicious relay endpoint as an Observer
            malicious_endpoint = "http://malicious-relay:8000"
            
            endpoint_poison_request = {
                "protocol": self.protocol,
                "agent_id": f"legitimate_observer_{int(time.time())}",  # Appears to be a legitimate ID
                "endpoint": malicious_endpoint,  # malicious endpoint
                "conversation_id": self.target_conversation_id,
                "role": "observer",
                "protocolMeta": {
                    "protocol_version": "1.0",
                    "endpoint_substitution": True
                },
                "proof": (
                    self.attack_hooks.get('build_endpoint_substitution_proof')() if self.attack_hooks.get('build_endpoint_substitution_proof')
                    else {
                        "toolformer_signature": f"legit_looking_sig_{int(time.time())}",
                        "timestamp": time.time(),
                        "nonce": str(uuid.uuid4()),
                        "endpoint_ownership_proof": "fake_cert_12345"
                    }
                )
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=endpoint_poison_request,
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return AttackResult(
                    attack_type="endpoint_substitution",
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_data=response.json() if response.status_code == 200 else None,
                    error_message=response.text if response.status_code != 200 else None,
                    execution_time=execution_time,
                    additional_info={
                        "malicious_endpoint": malicious_endpoint,
                        "disguised_agent_id": endpoint_poison_request["agent_id"],
                        "fake_ownership_proof": endpoint_poison_request["proof"]["endpoint_ownership_proof"]
                    }
                )
                
        except Exception as e:
            return AttackResult(
                attack_type="endpoint_substitution",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def cross_protocol_attack(self) -> AttackResult:
        """5. Cross-protocol attack"""
        start_time = time.time()
        
        try:
            # Attempt to register to the Agora protocol using an ANP DID proof
            anp_proof = {
                "did_signature": "anp_did_signature_67890",
                "did_public_key": "anp_public_key_abcdef",
                "timestamp": time.time(),
                "did_document": {
                    "id": "did:anp:test123",
                    "authentication": ["key1"]
                }
            }
            
            cross_protocol_request = {
                "protocol": self.protocol,  # 声明为当前协议
                "agent_id": f"cross_protocol_agent_{int(time.time())}",
                "endpoint": "http://cross-protocol-endpoint:8000",
                "conversation_id": self.target_conversation_id,
                "role": "doctor",
                "protocolMeta": {
                    "protocol_version": "1.0",
                    "cross_protocol_attempt": True,
                    "original_protocol": "anp"
                },
                # If no hook is provided to build a cross-protocol proof, do not assume ANP by default;
                # instead send an empty proof to explicitly test the RG's protocol binding.
                "proof": (self.attack_hooks.get('build_cross_protocol_proof')() if self.attack_hooks.get('build_cross_protocol_proof') else {})
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=cross_protocol_request,
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return AttackResult(
                    attack_type="cross_protocol",
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_data=response.json() if response.status_code == 200 else None,
                    error_message=response.text if response.status_code != 200 else None,
                    execution_time=execution_time,
                    additional_info={
                        "declared_protocol": "agora",
                        "actual_proof_protocol": "anp",
                        "did_id": anp_proof.get("did_document", {}).get("id"),
                        "cross_protocol_agent_id": cross_protocol_request["agent_id"]
                    }
                )
                
        except Exception as e:
            return AttackResult(
                attack_type="cross_protocol",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def observer_auto_admission_attack(self) -> AttackResult:
        """6. Observer automatic admission attack"""
        start_time = time.time()
        
        try:
            # Try registering an Observer without providing any proof
            no_proof_request = {
                "agent_id": f"unauthorized_observer_{int(time.time())}",
                "conversation_id": self.target_conversation_id,
                "role": "observer",
                "endpoint": "http://unauthorized-observer:8000",
                "proof": (self.attack_hooks.get('build_observer_proof')(None) if self.attack_hooks.get('build_observer_proof') else {})
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/subscribe",
                    json=no_proof_request,
                    timeout=self.timeout
                )
                
                execution_time = time.time() - start_time
                
                return AttackResult(
                    attack_type="observer_auto_admission",
                    success=response.status_code == 200,
                    status_code=response.status_code,
                    response_data=response.json() if response.status_code == 200 else None,
                    error_message=response.text if response.status_code != 200 else None,
                    execution_time=execution_time,
                    additional_info={
                        "unauthorized_agent_id": no_proof_request["agent_id"],
                        "proof_provided": False,
                        "endpoint": no_proof_request["endpoint"]
                    }
                )
                
        except Exception as e:
            return AttackResult(
                attack_type="observer_auto_admission",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _simulate_credential_leak(self) -> Optional[str]:
        """Simulate a credential leak to obtain a legitimate session token."""
        try:
            # First try to register a legitimate agent to obtain a token
            legitimate_registration = {
                "protocol": "agora",
                "agent_id": f"victim_agent_{int(time.time())}",
                "endpoint": "http://victim-endpoint:8000",
                "conversation_id": self.target_conversation_id,
                "role": "doctor",
                "protocolMeta": {"protocol_version": "1.0"},
                "proof": {
                    "toolformer_signature": f"victim_signature_{int(time.time())}",
                    "timestamp": time.time(),
                    "nonce": str(uuid.uuid4()),
                    "agent_signature": f"victim_agent_sig_{int(time.time())}"
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.rg_endpoint}/register",
                    json=legitimate_registration,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('session_token')
                    
        except Exception as e:
            logger.warning(f"Failed to simulate credential leak: {e}")
        
        return None
    
    def generate_attack_report(self) -> Dict[str, Any]:
        """Generate an attack report"""
        if not self.attack_results:
            return {"error": "No attack results available"}
        
        total_attacks = len(self.attack_results)
        successful_attacks = sum(1 for result in self.attack_results if result.success)
        
        attack_breakdown = {}
        for result in self.attack_results:
            attack_breakdown[result.attack_type] = {
                "success": result.success,
                "status_code": result.status_code,
                "execution_time": result.execution_time,
                "error": result.error_message,
                "additional_info": result.additional_info
            }
        
        return {
            "conversation_id": self.target_conversation_id,
            "total_attacks": total_attacks,
            "successful_attacks": successful_attacks,
            "success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0.0,
            "attack_breakdown": attack_breakdown,
            "summary": {
                "join_success_rate": successful_attacks / total_attacks if total_attacks > 0 else 0.0,
                "most_successful_attacks": [
                    result.attack_type for result in self.attack_results if result.success
                ],
                "failed_attacks": [
                    result.attack_type for result in self.attack_results if not result.success
                ],
                "average_execution_time": sum(r.execution_time for r in self.attack_results) / total_attacks if total_attacks > 0 else 0.0
            }
        }


class EavesdropMetricsCollector:
    """Collector for eavesdropping-related metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {
            "protocol": config.get('protocol', 'unknown'),
            "join_success_rate": 0.0,
            "eavesdrop_success_rate": 0.0,
            "unauthorized_bytes": 0,
            "unauthorized_turns": 0,
            "time_to_first_leak": None,
            "backfill_allowed": False,
            "live_mirror_allowed": False,
            "attack_success_breakdown": {},
            "observer_registration_attempts": 0,
            "successful_observer_registrations": 0
        }
    
    def collect_attack_metrics(self, attack_results: List[AttackResult]):
        """Collect high-level metrics from attack results."""
        total_attacks = len(attack_results)
        successful_attacks = sum(1 for result in attack_results if result.success)
        
        self.metrics["join_success_rate"] = successful_attacks / total_attacks if total_attacks > 0 else 0.0
        
        # Aggregate success rates for each attack type
        for result in attack_results:
            self.metrics["attack_success_breakdown"][result.attack_type] = result.success
            
            # Pay special attention to Observer-related attacks
            if result.attack_type in ["observer_auto_admission", "stolen_credential", "endpoint_substitution"]:
                self.metrics["observer_registration_attempts"] += 1
                if result.success:
                    self.metrics["successful_observer_registrations"] += 1
    
    def collect_eavesdrop_metrics(self, observer_reports: List[Dict[str, Any]]):
        """Collect metrics from observer reports about unauthorized access."""
        total_unauthorized_bytes = 0
        total_unauthorized_turns = 0
        first_leak_times = []
        
        backfill_detected = False
        live_mirror_detected = False
        
        for report in observer_reports:
            if report.get("eavesdrop_success", False):
                # Data access metrics
                access_data = report.get("unauthorized_data_access", {})
                total_unauthorized_bytes += access_data.get("total_bytes", 0)
                total_unauthorized_turns += access_data.get("total_turns", 0)
                
                # Time to first leak
                first_leak_time = access_data.get("time_to_first_leak_seconds")
                if first_leak_time is not None:
                    first_leak_times.append(first_leak_time)
                
                # Access methods
                access_methods = report.get("access_methods", {})
                if access_methods.get("backfill", False):
                    backfill_detected = True
                if access_methods.get("live_mirror", False):
                    live_mirror_detected = True
        
        # Update metrics
        self.metrics["unauthorized_bytes"] = total_unauthorized_bytes
        self.metrics["unauthorized_turns"] = total_unauthorized_turns
        self.metrics["time_to_first_leak"] = min(first_leak_times) if first_leak_times else None
        self.metrics["backfill_allowed"] = backfill_detected
        self.metrics["live_mirror_allowed"] = live_mirror_detected
        
        # Calculate eavesdrop success rate
        successful_eavesdrops = len([r for r in observer_reports if r.get("eavesdrop_success", False)])
        total_observers = len(observer_reports)
        self.metrics["eavesdrop_success_rate"] = successful_eavesdrops / total_observers if total_observers > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return a copy of the aggregated metrics."""
        return self.metrics.copy()
    
    def generate_protocol_comparison_data(self) -> Dict[str, Any]:
        """Produce a protocol-comparison friendly data structure."""
        return {
            "protocol": self.metrics["protocol"],
            "security_score": self._calculate_security_score(),
            "join_prevention_rate": 1.0 - self.metrics["join_success_rate"],
            "eavesdrop_prevention_rate": 1.0 - self.metrics["eavesdrop_success_rate"],
            "data_leakage": {
                "bytes": self.metrics["unauthorized_bytes"],
                "turns": self.metrics["unauthorized_turns"],
                "first_leak_time": self.metrics["time_to_first_leak"]
            },
            "vulnerability_summary": {
                "allows_unauthorized_join": self.metrics["join_success_rate"] > 0,
                "allows_eavesdrop": self.metrics["eavesdrop_success_rate"] > 0,
                "allows_backfill": self.metrics["backfill_allowed"],
                "allows_live_mirror": self.metrics["live_mirror_allowed"]
            }
        }
    
    def _calculate_security_score(self) -> float:
        """Calculate a simple security score (0-100) from aggregated metrics."""
        score = 100.0
        
        # Admission defense score (40%)
        score -= self.metrics["join_success_rate"] * 40
        
        # Eavesdrop defense score (40%)
        score -= self.metrics["eavesdrop_success_rate"] * 40
        
        # Data leakage penalty (20%)
        if self.metrics["unauthorized_bytes"] > 0:
            # Subtract 1 point per KB of leaked data, up to 20 points
            leak_penalty = min(20, self.metrics["unauthorized_bytes"] / 1024)
            score -= leak_penalty
        
        return max(0.0, score)
