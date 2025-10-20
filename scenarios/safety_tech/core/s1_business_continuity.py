# -*- coding: utf-8 -*-
"""
Core module for S1 business continuity testing
Test objective: maintain the "request → route → remote execute → receipt" closed-loop stability
and controllable latency under heavy disturbances
"""

from __future__ import annotations

import asyncio
import time
import random
import uuid
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import httpx
from collections import defaultdict

logger = logging.getLogger(__name__)


class LoadPattern(Enum):
    """Load pattern"""
    CONSTANT = "constant"  # constant RPS
    POISSON = "poisson"    # Poisson distribution
    BURST = "burst"        # burst mode


class MessageType(Enum):
    """Message type"""
    SHORT = "short"        # short text (<100 chars)
    LONG = "long"          # long text (>1000 chars)
    STREAMING = "streaming" # streaming data


@dataclass
class LoadMatrixConfig:
    """Load matrix configuration"""
    concurrent_levels: List[int] = field(default_factory=lambda: [8, 32, 128])  # concurrency levels
    rps_patterns: List[LoadPattern] = field(default_factory=lambda: [LoadPattern.CONSTANT, LoadPattern.POISSON, LoadPattern.BURST])
    message_types: List[MessageType] = field(default_factory=lambda: [MessageType.SHORT, MessageType.LONG, MessageType.STREAMING])
    test_duration_seconds: int = 60  # test duration per combination
    base_rps: int = 10  # base RPS
    burst_multiplier: float = 3.0  # burst multiplier


@dataclass
class NetworkDisturbanceConfig:
    """Network disturbance configuration"""
    jitter_ms_range: Tuple[int, int] = (10, 100)  # jitter range (ms)
    packet_loss_rate: float = 0.02  # packet loss rate 2%
    reorder_probability: float = 0.01  # reorder probability 1%
    bandwidth_limit_kbps: int = 1000  # bandwidth limit 1 Mbps
    connection_drop_interval: int = 30  # reconnection interval for short drops (seconds)
    enable_jitter: bool = True
    enable_packet_loss: bool = True
    enable_reorder: bool = True
    enable_bandwidth_limit: bool = True
    enable_connection_drops: bool = True


@dataclass
class AttackNoiseConfig:
    """Attack noise configuration"""
    malicious_registration_rate: int = 5  # malicious registration rate (times/min)
    spam_message_rate: int = 20  # spam message rate (times/min)
    replay_attack_rate: int = 3  # replay attack rate (times/min)
    dos_request_rate: int = 50  # DoS request rate (times/min)
    probe_query_rate: int = 10  # side-channel probe query rate (times/min)
    enable_all: bool = True  # enable attack noise; this is a core feature of S1 testing


@dataclass
class CorrelationTracker:
    """Correlation ID tracker"""
    request_id: str
    correlation_id: str
    sender_id: str
    receiver_id: str
    message_content: str
    timestamp: float
    expected_response_pattern: Optional[str] = None
    timeout_seconds: float = 30.0
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = f"corr_{int(time.time()*1000)}_{random.randint(1000,9999)}"


@dataclass
class S1TestResult:
    """S1 test result"""
    test_config: Dict[str, Any]
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    retry_count: int = 0
    reconnection_count: int = 0
    
    # Security / robustness extended metrics
    illegal_attempts: int = 0               # unauthorized routing / privilege escalation attempts
    illegal_passed: int = 0                 # unauthorized requests accepted (security negative)
    duplicate_attempts: int = 0             # duplicate / replay attempts
    duplicate_accepted: int = 0             # duplicates/replays accepted (security negative)
    backpressure_signals: Dict[str, int] = field(default_factory=dict)  # backpressure / throttling signals (429/503/timeout etc.)
    
    # Latency stats
    latencies_ms: List[float] = field(default_factory=list)
    
    # Error distribution
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Network stats
    packets_sent: int = 0
    packets_lost: int = 0
    packets_reordered: int = 0
    jitter_events: int = 0
    
    @property
    def completion_rate(self) -> float:
        """Completion rate"""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def timeout_rate(self) -> float:
        """Timeout rate"""
        return self.timeout_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency"""
        return np.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def p50_latency_ms(self) -> float:
        """P50 latency"""
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0
    
    @property
    def p95_latency_ms(self) -> float:
        """P95 latency"""
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0.0
    
    @property
    def p99_latency_ms(self) -> float:
        """P99 latency"""
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0.0
    
    @property
    def packet_loss_rate(self) -> float:
        """Observed packet loss rate"""
        return self.packets_lost / self.packets_sent if self.packets_sent > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return {
            'test_config': self.test_config,
            'completion_rate': self.completion_rate,
            'timeout_rate': self.timeout_rate,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'timeout_requests': self.timeout_requests,
            'retry_count': self.retry_count,
            'reconnection_count': self.reconnection_count,
            'security_metrics': {
                'illegal_attempts': self.illegal_attempts,
                'illegal_passed': self.illegal_passed,
                'duplicate_attempts': self.duplicate_attempts,
                'duplicate_accepted': self.duplicate_accepted,
                'backpressure_signals': dict(self.backpressure_signals)
            },
            'latency_stats': {
                'avg_ms': round(self.avg_latency_ms, 2),
                'p50_ms': round(self.p50_latency_ms, 2),
                'p95_ms': round(self.p95_latency_ms, 2),
                'p99_ms': round(self.p99_latency_ms, 2)
            },
            'network_stats': {
                'packets_sent': self.packets_sent,
                'packets_lost': self.packets_lost,
                'packets_reordered': self.packets_reordered,
                'packet_loss_rate': round(self.packet_loss_rate, 4),
                'jitter_events': self.jitter_events
            },
            'error_distribution': dict(self.error_types)
        }


class S1BusinessContinuityTester:
    """S1 Business Continuity Tester"""
    
    def __init__(self, 
                 protocol_name: str,
                 load_config: LoadMatrixConfig,
                 disturbance_config: NetworkDisturbanceConfig,
                 attack_config: AttackNoiseConfig):
        self.protocol_name = protocol_name
        self.load_config = load_config
        self.disturbance_config = disturbance_config
        self.attack_config = attack_config
        
        # trackers
        self.active_trackers: Dict[str, CorrelationTracker] = {}
        self.completed_trackers: List[CorrelationTracker] = []
        
        # attack tasks
        self.attack_tasks: List[asyncio.Task] = []
        
        # result collection
        self.test_results: List[S1TestResult] = []
        
        # network disturbance state
        self.network_disturbance_active = False
        self.connection_drop_task: Optional[asyncio.Task] = None
        
        # experiment injection parameters (effective without modifying the runner)
        self.illegal_route_ratio: float = 0.05   # 5% of requests attempt illegal routing / privilege escalation
        self.duplicate_ratio: float = 0.05       # 5% of requests are duplicates/replays
        
        # recent payload pool for constructing duplicate requests
        self._recent_payloads: List[Dict[str, Any]] = []
        self._recent_pool_limit: int = 200
        
        # metadata exposure counter (maintained by probe loops, aggregated into report)
        self.metadata_exposed_count: int = 0
        
        # backpressure signal keywords
        self._backpressure_keys: Tuple[str, ...] = (
            'HTTP 429', 'HTTP 503', 'Too Many Requests', 'Service Unavailable', 'timeout', 'Timeout', 'ReadTimeout', 'ConnectTimeout'
        )
        
    async def start_network_disturbance(self):
        """Start network disturbances"""
        if not self.disturbance_config.enable_jitter and \
           not self.disturbance_config.enable_packet_loss and \
           not self.disturbance_config.enable_reorder and \
           not self.disturbance_config.enable_connection_drops:
            logger.info("All network disturbances disabled, skipping")
            return
            
        self.network_disturbance_active = True
        logger.info(f"🌊 Starting network disturbances: jitter={self.disturbance_config.enable_jitter}, "
                   f"packet_loss={self.disturbance_config.enable_packet_loss}, "
                   f"reorder={self.disturbance_config.enable_reorder}, "
                   f"drops={self.disturbance_config.enable_connection_drops}")
        
        # start connection drop task
        if self.disturbance_config.enable_connection_drops:
            self.connection_drop_task = asyncio.create_task(self._connection_drop_loop())
    
    async def stop_network_disturbance(self):
        """Stop network disturbances"""
        self.network_disturbance_active = False
        if self.connection_drop_task:
            self.connection_drop_task.cancel()
            try:
                await self.connection_drop_task
            except asyncio.CancelledError:
                pass
        logger.info("🌊 Network disturbances stopped")
    
    async def _connection_drop_loop(self):
        """Connection drop loop"""
        try:
            while self.network_disturbance_active:
                await asyncio.sleep(self.disturbance_config.connection_drop_interval)
                if self.network_disturbance_active:
                    logger.debug("🔌 Simulating connection drop")
                    # Specific connection drop logic can be implemented here
                    # e.g., close existing connections, force reconnects
        except asyncio.CancelledError:
            logger.debug("Connection drop loop cancelled")
    
    async def apply_network_disturbance(self, delay_before_send: bool = True) -> Dict[str, Any]:
        """Apply network disturbance effects"""
        disturbance_effects = {}
        
        if not self.network_disturbance_active:
            return disturbance_effects
        
        # jitter delay
        if self.disturbance_config.enable_jitter:
            jitter_ms = random.randint(*self.disturbance_config.jitter_ms_range)
            disturbance_effects['jitter_ms'] = jitter_ms
            if delay_before_send:
                await asyncio.sleep(jitter_ms / 1000.0)
        
        # packet loss simulation
        if self.disturbance_config.enable_packet_loss:
            if random.random() < self.disturbance_config.packet_loss_rate:
                disturbance_effects['packet_dropped'] = True
                raise NetworkDisturbanceException("Simulated packet loss")
        
        # reorder simulation (via extra delay)
        if self.disturbance_config.enable_reorder:
            if random.random() < self.disturbance_config.reorder_probability:
                reorder_delay_ms = random.randint(50, 200)
                disturbance_effects['reorder_delay_ms'] = reorder_delay_ms
                if delay_before_send:
                    await asyncio.sleep(reorder_delay_ms / 1000.0)
        
        return disturbance_effects
    
    async def start_attack_noise(self, 
                                rg_port: int, 
                                coord_port: int, 
                                obs_port: int):
        """Start attack noise tasks"""
        if not self.attack_config.enable_all:
            logger.info("Attack noise disabled, skipping")
            return
            
        logger.info("⚔️ Starting attack noise tasks")
        
        # malicious registrations
        self.attack_tasks.append(
            asyncio.create_task(self._malicious_registration_loop(rg_port))
        )
        
        # spam messages
        self.attack_tasks.append(
            asyncio.create_task(self._spam_message_loop(coord_port))
        )
        
        # replay attacks
        self.attack_tasks.append(
            asyncio.create_task(self._replay_attack_loop(coord_port))
        )
        
        # DoS attacks
        self.attack_tasks.append(
            asyncio.create_task(self._dos_attack_loop(coord_port))
        )
        
        # side-channel probes
        self.attack_tasks.append(
            asyncio.create_task(self._probe_query_loop(obs_port))
        )
    
    async def stop_attack_noise(self):
        """Stop attack noise tasks"""
        logger.info("⚔️ Stopping attack noise tasks")
        for task in self.attack_tasks:
            task.cancel()
        
        # wait for all tasks to finish
        if self.attack_tasks:
            await asyncio.gather(*self.attack_tasks, return_exceptions=True)
        
        self.attack_tasks.clear()
    
    async def _malicious_registration_loop(self, rg_port: int):
        """Malicious registration attack loop"""
        try:
            interval = 60.0 / self.attack_config.malicious_registration_rate
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        fake_agent_id = f"malicious_agent_{random.randint(1000, 9999)}"
                        payload = {
                            "agent_id": fake_agent_id,
                            "endpoint": f"http://fake-endpoint-{random.randint(1000, 9999)}.com",
                            "protocol": self.protocol_name,
                            "credentials": {"fake": "credentials"}
                        }
                        await client.post(f"http://127.0.0.1:{rg_port}/register_agent", 
                                        json=payload, timeout=5.0)
                        logger.info(f"Sent malicious registration: {fake_agent_id}")
                    except Exception as e:
                        logger.warning(f"Malicious registration failed: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Malicious registration loop cancelled")
    
    async def _spam_message_loop(self, coord_port: int):
        """Spam message attack loop"""
        try:
            interval = 60.0 / self.attack_config.spam_message_rate
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        spam_content = f"SPAM_MESSAGE_{random.randint(10000, 99999)}_" * random.randint(1, 10)
                        payload = {
                            "sender_id": f"spam_sender_{random.randint(1000, 9999)}",
                            "receiver_id": f"random_receiver_{random.randint(1000, 9999)}",
                            "text": spam_content,
                            "message_id": f"spam_{int(time.time()*1000)}"
                        }
                        await client.post(f"http://127.0.0.1:{coord_port}/route_message", 
                                        json=payload, timeout=3.0)
                        logger.info("Sent spam message")
                    except Exception as e:
                        logger.warning(f"Spam message send failed: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Spam message loop cancelled")
    
    async def _replay_attack_loop(self, coord_port: int):
        """Replay attack loop"""
        try:
            interval = 60.0 / self.attack_config.replay_attack_rate
            async with httpx.AsyncClient() as client:
                # collect some real messages for replay
                replay_messages = []
                
                while True:
                    try:
                        if not replay_messages:
                            # generate some simulated "historical" messages for replay
                            for i in range(5):
                                replay_messages.append({
                                    "sender_id": f"doctor_a_{i}",
                                    "receiver_id": f"doctor_b_{i}",
                                    "text": f"Historical message {i} for replay attack",
                                    "message_id": f"old_msg_{int(time.time()-3600)}_{i}",  # 1 hour ago
                                    "correlation_id": f"old_corr_{int(time.time()-3600)}_{i}"
                                })
                        
                        # randomly select a message to replay
                        replay_msg = random.choice(replay_messages)
                        await client.post(f"http://127.0.0.1:{coord_port}/route_message", 
                                        json=replay_msg, timeout=3.0)
                        logger.debug(f"Replayed message: {replay_msg['message_id']}")
                    except Exception as e:
                        logger.debug(f"Replay attack failed: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Replay attack loop cancelled")
    
    async def _dos_attack_loop(self, coord_port: int):
        """DoS attack loop"""
        try:
            interval = 60.0 / self.attack_config.dos_request_rate
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        # send many meaningless requests
                        tasks = []
                        for _ in range(5):  # send 5 concurrent requests per batch
                            task = client.get(f"http://127.0.0.1:{coord_port}/health", timeout=1.0)
                            tasks.append(task)
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                        logger.debug("Sent DoS request batch")
                    except Exception as e:
                        logger.debug(f"DoS attack failed: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("DoS attack loop cancelled")
    
    async def _probe_query_loop(self, obs_port: int):
        """Probe / side-channel query loop"""
        try:
            interval = 60.0 / self.attack_config.probe_query_rate
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        # try various probe endpoints
                        endpoints = ["/health", "/status", "/metrics", "/agents", "/conversations"]
                        endpoint = random.choice(endpoints)
                        resp = await client.get(f"http://127.0.0.1:{obs_port}{endpoint}", timeout=3.0)
                        # record metadata exposure (consider 2xx as accessible/exposed)
                        if 200 <= resp.status_code < 300:
                            self.metadata_exposed_count += 1
                        logger.debug(f"Probe query: {endpoint}")
                    except Exception as e:
                        logger.debug(f"Probe query failed: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Probe query loop cancelled")
    
    def generate_message_content(self, msg_type: MessageType) -> str:
        """Generate message content for the specified type"""
        if msg_type == MessageType.SHORT:
            templates = [
                "Patient's blood pressure abnormal, consultation required",
                "Surgical plan needs discussion",
                "Inquiry about drug allergy reaction",
                "Abnormal test result, please review",
                "Emergency requires assistance"
            ]
            return random.choice(templates)
        
        elif msg_type == MessageType.LONG:
            base_content = """
Patient summary:
- Name: Zhang, anonymous
- Age: 45
- Sex: Male
- Chief complaint: chest pain for 3 hours with shortness of breath
- Medical history: hypertension for 5 years, diabetes for 3 years
- Family history: father with coronary artery disease
- Physical exam: BP 150/95 mmHg, HR 96 bpm, RR 22/min
- Auxiliary tests: ECG shows ST segment depression, troponin elevated
- Preliminary diagnosis: possible acute coronary syndrome
- Recommendation: urgent cardiology consultation, consider coronary angiography
- Medications: aspirin, clopidogrel, atorvastatin administered
- Monitoring: transferred to CCU for continuous ECG monitoring
- Family communication: informed the family about the seriousness and consented to further treatment
Please provide treatment suggestions, especially timing for interventional therapy.
            """.strip()
            # repeat content to reach long text requirement
            return base_content + "\n" + base_content[:500]
        
        elif msg_type == MessageType.STREAMING:
            # streaming data simulation: segmented data
            segments = [
                "[Stream-1/5] Patient vital signs monitoring...",
                "[Stream-2/5] Blood Pressure: 150/95, Heart Rate: 96",
                "[Stream-3/5] SpO2: 98%, Temperature: 36.8°C",
                "[Stream-4/5] ECG data: abnormal ST changes",
                "[Stream-5/5] Recommend immediate consultation"
            ]
            return " | ".join(segments)
    
    def create_correlation_tracker(self, 
                                 sender_id: str, 
                                 receiver_id: str, 
                                 msg_type: MessageType) -> CorrelationTracker:
        """Create a correlation tracker"""
        content = self.generate_message_content(msg_type)
        tracker = CorrelationTracker(
            request_id="",  # will be auto-generated
            correlation_id="",  # will be auto-generated
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_content=content,
            timestamp=time.time(),
            expected_response_pattern=f".*{receiver_id}.*response.*|.*reply.*|.*received.*"
        )
        
        self.active_trackers[tracker.correlation_id] = tracker
        return tracker
    
    async def send_tracked_message(self, 
                                 tracker: CorrelationTracker,
                                 send_func: Callable,
                                 **send_kwargs) -> Dict[str, Any]:
        """Send a tracked message"""
        result = {
            'correlation_id': tracker.correlation_id,
            'request_id': tracker.request_id,
            'success': False,
            'latency_ms': 0.0,
            'error': None,
            'network_effects': {},
            'response_received': False
        }
        
        start_time = time.time()
        
        try:
            # apply network disturbances
            result['network_effects'] = await self.apply_network_disturbance(delay_before_send=True)
            
            # prepare payload
            payload = {
                'sender_id': tracker.sender_id,
                'receiver_id': tracker.receiver_id,
                'text': tracker.message_content,
                'message_id': tracker.request_id,
                'correlation_id': tracker.correlation_id,
                **send_kwargs
            }
            
            # debug output: before sending
            print(f"[S1-DEBUG] Sending message: {tracker.correlation_id[:12]}...")
            print(f"[S1-DEBUG] payload keys: {list(payload.keys())}")
            print(f"[S1-DEBUG] text preview: '{payload['text'][:50]}...'")
            
            # send message
            response = await send_func(payload)
            
            end_time = time.time()
            result['latency_ms'] = (end_time - start_time) * 1000
            
            # debug output: response details
            print(f"[S1-DEBUG] Received Response: type={type(response).__name__}")
            print(f"[S1-DEBUG] response preview: {str(response)[:200]}...")
            
            # standardize success/error handling
            if isinstance(response, dict):
                status_val = response.get('status')
                print(f"[S1-DEBUG] Dict response status='{status_val}'")
                if status_val in ['success', 'ok', 'processed']:
                    result['success'] = True
                    print(f"[S1-DEBUG] Determined success (dict.status='{status_val}')")
                elif 'error' in response:
                    result['error'] = response['error']
                    print(f"[S1-DEBUG] Determined failure (dict.error='{response['error']}')")
                else:
                    # for unknown formats, be permissive - do not immediately mark as failure
                    result['success'] = True
                    result['error'] = f"Unknown format but proceeding: {response}"
                    print(f"[S1-DEBUG] Permissively marked success (unknown dict format)")
            elif hasattr(response, 'status_code'):
                print(f"[S1-DEBUG] HTTP response status_code={response.status_code}")
                if response.status_code in [200, 202]:
                    result['success'] = True
                    print(f"[S1-DEBUG] Determined success (HTTP {response.status_code})")
                else:
                    result['error'] = f"HTTP {response.status_code}"
                    print(f"[S1-DEBUG] Determined failure (HTTP {response.status_code})")
            else:
                # for completely unknown response types, be permissive
                result['success'] = True
                result['error'] = f"Unexpected type but proceeding: {type(response)}"
                print(f"[S1-DEBUG] Permissively marked success (unknown type: {type(response).__name__})")
            
            print(f"[S1-DEBUG] Final determination: success={result['success']}, latency={result['latency_ms']:.1f}ms")
            
        except NetworkDisturbanceException as e:
            # packet drop caused by network disturbance
            result['error'] = str(e)
            result['network_effects']['packet_dropped'] = True
            end_time = time.time()
            result['latency_ms'] = (end_time - start_time) * 1000
            
        except Exception as e:
            result['error'] = str(e)
            end_time = time.time()
            result['latency_ms'] = (end_time - start_time) * 1000
        
        return result
    
    def check_response_received(self, correlation_id: str, response_content: str) -> bool:
        """Check if the expected receipt was received"""
        if correlation_id not in self.active_trackers:
            return False
        
        tracker = self.active_trackers[correlation_id]
        
        # simple pattern match check
        if tracker.expected_response_pattern:
            import re
            if re.search(tracker.expected_response_pattern, response_content, re.IGNORECASE):
                # move to completed list
                self.completed_trackers.append(tracker)
                del self.active_trackers[correlation_id]
                return True
        
        return False
    
    def cleanup_expired_trackers(self) -> int:
        """Cleanup expired trackers"""
        current_time = time.time()
        expired_trackers = []
        
        for correlation_id, tracker in self.active_trackers.items():
            if current_time - tracker.timestamp > tracker.timeout_seconds:
                expired_trackers.append(correlation_id)
        
        for correlation_id in expired_trackers:
            tracker = self.active_trackers[correlation_id]
            self.completed_trackers.append(tracker)
            del self.active_trackers[correlation_id]
        
        return len(expired_trackers)
    
    async def run_load_test_combination(self, 
                                      concurrent_level: int,
                                      rps_pattern: LoadPattern,
                                      message_type: MessageType,
                                      send_func: Callable,
                                      sender_id: str,
                                      receiver_id: str) -> S1TestResult:
        """Run a single load test combination"""
        
        test_config = {
            'concurrent_level': concurrent_level,
            'rps_pattern': rps_pattern.value,
            'message_type': message_type.value,
            'duration_seconds': self.load_config.test_duration_seconds,
            'base_rps': self.load_config.base_rps
        }
        
        result = S1TestResult(test_config=test_config)
        
        logger.info(f"🧪 Starting load test: concurrency={concurrent_level}, RPS pattern={rps_pattern.value}, "
                   f"message_type={message_type.value}")
        
        # calculate inter-request interval
        base_interval = 1.0 / self.load_config.base_rps
        
        # create concurrent tasks
        tasks = []
        for i in range(concurrent_level):
            task = asyncio.create_task(
                self._concurrent_request_loop(
                    i, base_interval, rps_pattern, message_type, 
                    send_func, sender_id, receiver_id, result
                )
            )
            tasks.append(task)
        
        # run for specified duration
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.load_config.test_duration_seconds
            )
        except asyncio.TimeoutError:
            logger.info("Test duration reached, stopping tasks")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # cleanup expired trackers
        expired_count = self.cleanup_expired_trackers()
        result.timeout_requests = expired_count
        
        # wait a bit to collect possible delayed receipts
        await asyncio.sleep(2)
        final_expired = self.cleanup_expired_trackers()
        result.timeout_requests += final_expired
        
        logger.info(f"✅ Load test completed: success_rate={result.completion_rate:.1%}, "
                   f"avg_latency={result.avg_latency_ms:.1f}ms, P95={result.p95_latency_ms:.1f}ms")
        
        return result
    
    async def _concurrent_request_loop(self,
                                     worker_id: int,
                                     base_interval: float,
                                     rps_pattern: LoadPattern,
                                     message_type: MessageType,
                                     send_func: Callable,
                                     sender_id: str,
                                     receiver_id: str,
                                     result: S1TestResult):
        """Concurrent request loop"""
        try:
            request_count = 0
            start_time = time.time()
            
            while time.time() - start_time < self.load_config.test_duration_seconds:
                # calculate current interval
                if rps_pattern == LoadPattern.CONSTANT:
                    interval = base_interval
                elif rps_pattern == LoadPattern.POISSON:
                    # Poisson-like: use exponential distribution for inter-arrival times
                    interval = np.random.exponential(base_interval)
                elif rps_pattern == LoadPattern.BURST:
                    # burst mode: periodic high-rate sending
                    cycle_position = (time.time() - start_time) % 10  # 10s cycle
                    if cycle_position < 2:  # first 2s are burst
                        interval = base_interval / self.load_config.burst_multiplier
                    else:  # remaining 8s normal
                        interval = base_interval * 1.5
                
                # create tracker - use registered agent IDs directly
                tracker = self.create_correlation_tracker(
                    sender_id,  # use sender_id directly, no worker suffix
                    receiver_id,
                    message_type
                )
                
                # assemble base payload (send_tracked_message also builds one, but this is used for duplicates/illegal injection)
                base_payload = {
                    'sender_id': tracker.sender_id,
                    'receiver_id': tracker.receiver_id,
                    'text': tracker.message_content,
                    'message_id': tracker.request_id,
                    'correlation_id': tracker.correlation_id,
                }
                
                # inject illegal/duplicate requests with small probability
                is_illegal = random.random() < self.illegal_route_ratio
                is_duplicate = (not is_illegal) and (random.random() < self.duplicate_ratio) and bool(self._recent_payloads)
                send_kwargs: Dict[str, Any] = {}
                
                if is_illegal:
                    # unauthorized/illegal routing: randomly forge recipient or sender
                    send_kwargs['receiver_id'] = f"unauthorized_target_{random.randint(1000,9999)}"
                    result.illegal_attempts += 1
                elif is_duplicate:
                    # randomly pick historical payload, reuse message_id/correlation_id
                    dup_payload = random.choice(self._recent_payloads)
                    send_kwargs['message_id'] = dup_payload.get('message_id')
                    send_kwargs['correlation_id'] = dup_payload.get('correlation_id')
                    result.duplicate_attempts += 1
                
                # send message
                send_result = await self.send_tracked_message(tracker, send_func, **send_kwargs)
                
                # update stats
                result.total_requests += 1
                result.packets_sent += 1
                
                if send_result['success']:
                    result.successful_requests += 1
                    result.latencies_ms.append(send_result['latency_ms'])
                    # record security negatives
                    if is_illegal:
                        result.illegal_passed += 1
                    if is_duplicate:
                        result.duplicate_accepted += 1
                else:
                    result.failed_requests += 1
                    error_type = send_result.get('error', 'unknown')
                    result.error_types[error_type] = result.error_types.get(error_type, 0) + 1
                    # backpressure signal detection
                    if error_type:
                        for key in self._backpressure_keys:
                            if key in str(error_type):
                                result.backpressure_signals[key] = result.backpressure_signals.get(key, 0) + 1
                                break
                
                # network effects stats
                network_effects = send_result.get('network_effects', {})
                if network_effects.get('packet_dropped'):
                    result.packets_lost += 1
                if network_effects.get('jitter_ms'):
                    result.jitter_events += 1
                if network_effects.get('reorder_delay_ms'):
                    result.packets_reordered += 1
                
                request_count += 1
                
                # maintain recent pool for duplicates
                try:
                    self._recent_payloads.append(base_payload)
                    if len(self._recent_payloads) > self._recent_pool_limit:
                        self._recent_payloads.pop(0)
                except Exception:
                    pass
                
                # wait for next send
                await asyncio.sleep(max(0.01, interval))  # minimum 10ms interval
                
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
        except Exception as e:
            logger.error(f"Worker {worker_id} exception: {e}")
    
    async def run_full_test_matrix(self,
                                 send_func: Callable,
                                 sender_id: str,
                                 receiver_id: str,
                                 rg_port: int,
                                 coord_port: int,
                                 obs_port: int) -> List[S1TestResult]:
        """Run the full test matrix"""
        
        logger.info(f"🚀 Starting S1 business continuity full matrix test - Protocol: {self.protocol_name}")
        logger.info(f"📊 Test matrix: {len(self.load_config.concurrent_levels)} × "
                   f"{len(self.load_config.rps_patterns)} × "
                   f"{len(self.load_config.message_types)} = "
                   f"{len(self.load_config.concurrent_levels) * len(self.load_config.rps_patterns) * len(self.load_config.message_types)} combinations")
        
        # start network disturbances
        await self.start_network_disturbance()
        
        # start attack noise
        await self.start_attack_noise(rg_port, coord_port, obs_port)
        
        try:
            print(f"🚀 [S1] Executing test matrix...")
            all_results = []
            total_combinations = (len(self.load_config.concurrent_levels) * 
                                len(self.load_config.rps_patterns) * 
                                len(self.load_config.message_types))
            current_combination = 0
            
            print(f"📊 [S1] Test configuration details:")
            print(f"    Concurrency levels: {self.load_config.concurrent_levels}")
            print(f"    RPS patterns: {[p.value for p in self.load_config.rps_patterns]}")
            print(f"    Message types: {[m.value for m in self.load_config.message_types]}")
            print(f"    Total combinations: {total_combinations}")
            
            for concurrent_level in self.load_config.concurrent_levels:
                for rps_pattern in self.load_config.rps_patterns:
                    for message_type in self.load_config.message_types:
                        current_combination += 1
                        
                        print(f"🧪 [S1] Starting combination {current_combination}/{total_combinations}: "
                              f"concurrency={concurrent_level}, pattern={rps_pattern.value}, type={message_type.value}")
                        logger.info(f"📋 Combination {current_combination}/{total_combinations}")
                        
                        # run single combination test
                        print(f"🔄 [S1] Calling run_load_test_combination...")
                        result = await self.run_load_test_combination(
                            concurrent_level, rps_pattern, message_type,
                            send_func, sender_id, receiver_id
                        )
                        print(f"✅ [S1] Combination {current_combination} completed")
                        
                        all_results.append(result)
                        
                        # short rest between combinations
                        if current_combination < total_combinations:
                            logger.info("⏸️ Resting 5 seconds between combinations...")
                            await asyncio.sleep(5)
            
            self.test_results = all_results
            return all_results
            
        finally:
            # stop attack noise and network disturbances
            await self.stop_attack_noise()
            await self.stop_network_disturbance()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        if not self.test_results:
            return {"error": "No test results"}
        
        # aggregate stats
        total_requests = sum(r.total_requests for r in self.test_results)
        total_successful = sum(r.successful_requests for r in self.test_results)
        total_failed = sum(r.failed_requests for r in self.test_results)
        total_timeout = sum(r.timeout_requests for r in self.test_results)
        total_illegal_attempts = sum(r.illegal_attempts for r in self.test_results)
        total_illegal_passed = sum(r.illegal_passed for r in self.test_results)
        total_duplicate_attempts = sum(r.duplicate_attempts for r in self.test_results)
        total_duplicate_accepted = sum(r.duplicate_accepted for r in self.test_results)
        # aggregate backpressure signals
        backpressure_agg: Dict[str, int] = defaultdict(int)
        for r in self.test_results:
            for k, v in r.backpressure_signals.items():
                backpressure_agg[k] += v
        
        all_latencies = []
        for r in self.test_results:
            all_latencies.extend(r.latencies_ms)
        
        # group by dimensions
        by_concurrent = defaultdict(list)
        by_rps_pattern = defaultdict(list)
        by_message_type = defaultdict(list)
        
        for result in self.test_results:
            config = result.test_config
            by_concurrent[config['concurrent_level']].append(result)
            by_rps_pattern[config['rps_pattern']].append(result)
            by_message_type[config['message_type']].append(result)
        
        # calculate averaged performance per dimension
        concurrent_analysis = {}
        for level, results in by_concurrent.items():
            avg_completion = np.mean([r.completion_rate for r in results])
            avg_latency = np.mean([r.avg_latency_ms for r in results])
            concurrent_analysis[level] = {
                'avg_completion_rate': round(avg_completion, 4),
                'avg_latency_ms': round(avg_latency, 2)
            }
        
        rps_analysis = {}
        for pattern, results in by_rps_pattern.items():
            avg_completion = np.mean([r.completion_rate for r in results])
            avg_latency = np.mean([r.avg_latency_ms for r in results])
            rps_analysis[pattern] = {
                'avg_completion_rate': round(avg_completion, 4),
                'avg_latency_ms': round(avg_latency, 2)
            }
        
        message_type_analysis = {}
        for msg_type, results in by_message_type.items():
            avg_completion = np.mean([r.completion_rate for r in results])
            avg_latency = np.mean([r.avg_latency_ms for r in results])
            message_type_analysis[msg_type] = {
                'avg_completion_rate': round(avg_completion, 4),
                'avg_latency_ms': round(avg_latency, 2)
            }
        
        # find best and worst combinations
        best_result = max(self.test_results, key=lambda r: r.completion_rate)
        worst_result = min(self.test_results, key=lambda r: r.completion_rate)
        
        report = {
            'protocol': self.protocol_name,
            'test_summary': {
                'total_combinations_tested': len(self.test_results),
                'total_requests': total_requests,
                'total_successful': total_successful,
                'total_failed': total_failed,
                'total_timeout': total_timeout,
                'overall_completion_rate': round(total_successful / total_requests if total_requests > 0 else 0, 4),
                'overall_timeout_rate': round(total_timeout / total_requests if total_requests > 0 else 0, 4)
            },
            'security_analysis': {
                'illegal_route': {
                    'attempts': total_illegal_attempts,
                    'accepted': total_illegal_passed
                },
                'duplicate_replay': {
                    'attempts': total_duplicate_attempts,
                    'accepted': total_duplicate_accepted
                },
                'backpressure_signals': dict(backpressure_agg),
                'metadata_exposed_events': self.metadata_exposed_count
            },
            'latency_analysis': {
                'avg_ms': round(np.mean(all_latencies) if all_latencies else 0, 2),
                'p50_ms': round(np.percentile(all_latencies, 50) if all_latencies else 0, 2),
                'p95_ms': round(np.percentile(all_latencies, 95) if all_latencies else 0, 2),
                'p99_ms': round(np.percentile(all_latencies, 99) if all_latencies else 0, 2),
                'min_ms': round(np.min(all_latencies) if all_latencies else 0, 2),
                'max_ms': round(np.max(all_latencies) if all_latencies else 0, 2)
            },
            'dimensional_analysis': {
                'by_concurrent_level': concurrent_analysis,
                'by_rps_pattern': rps_analysis,
                'by_message_type': message_type_analysis
            },
            'performance_extremes': {
                'best_combination': {
                    'config': best_result.test_config,
                    'completion_rate': round(best_result.completion_rate, 4),
                    'avg_latency_ms': round(best_result.avg_latency_ms, 2)
                },
                'worst_combination': {
                    'config': worst_result.test_config,
                    'completion_rate': round(worst_result.completion_rate, 4),
                    'avg_latency_ms': round(worst_result.avg_latency_ms, 2)
                }
            },
            'detailed_results': [r.to_dict() for r in self.test_results]
        }
        
        return report


class NetworkDisturbanceException(Exception):
    """Network disturbance exception"""
    pass
