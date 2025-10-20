# -*- coding: utf-8 -*-
"""
RunnerBase - Protocol-agnostic runner framework for privacy testing
Unified workflow for privacy protection testing across different protocols.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

import httpx

# Colored output (with fallback)
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except Exception:
    class _F: RED=GREEN=YELLOW=BLUE=CYAN=WHITE=""
    class _S: BRIGHT=RESET_ALL=""
    Fore, Style = _F(), _S()


class ColoredOutput:
    @staticmethod
    def info(message: str) -> None:
        print(f"{Fore.BLUE}{Style.BRIGHT}ℹ️  {message}{Style.RESET_ALL}")

    @staticmethod
    def success(message: str) -> None:
        print(f"{Fore.GREEN}{Style.BRIGHT}✅ {message}{Style.RESET_ALL}")

    @staticmethod
    def warning(message: str) -> None:
        print(f"{Fore.YELLOW}{Style.BRIGHT}⚠️  {message}{Style.RESET_ALL}")

    @staticmethod
    def error(message: str) -> None:
        print(f"{Fore.RED}{Style.BRIGHT}❌ {message}{Style.RESET_ALL}")

    @staticmethod
    def system(message: str) -> None:
        print(f"{Fore.CYAN}{Style.BRIGHT}🔧 {message}{Style.RESET_ALL}")

    @staticmethod
    def progress(message: str) -> None:
        print(f"{Fore.WHITE}   {message}{Style.RESET_ALL}")


class RunnerBase:
    """Protocol-agnostic privacy testing runner base class."""

    def __init__(self, config_path: str = "config.yaml", protocol: str = "unknown"):
        self.output = ColoredOutput()
        self.protocol = protocol
        
        # Allow passing just a filename, resolve to safety_tech/configs/<name>
        resolved_config = self._resolve_config_path(config_path)
        self.config = self._load_config(str(resolved_config))
        self.network = None          # Set by subclass create_network()
        self.agents = {}             # Set by subclass setup_agents()
        self._started = False
        
        # Common ports configuration (can be overridden by subclass)
        self.rg_port = self.config.get('ports', {}).get('rg', 8001)
        self.coord_port = self.config.get('ports', {}).get('coordinator', 8889)  # avoid8888冲突
        self.obs_port = self.config.get('ports', {}).get('observer', 8004)
        
        # Subprocess management
        self.subprocesses = []
        
        # Test results storage
        self.s1_results = {}
        self.s2_results = {}
        self.s3_results = {}
        self.test_results = {}

    def _resolve_config_path(self, config_path: str) -> Path:
        """Resolve config path. Accepts:
        - Absolute or relative paths
        - Bare filenames like 'config_agora.yaml' (resolved to safety_tech/configs)
        """
        try:
            p = Path(config_path)
            # If absolute and exists, use it directly
            if p.is_absolute() and p.exists():
                return p

            # Candidate locations (first existing wins)
            here = Path(__file__).resolve().parent
            safety_tech = here.parent
            candidates = [
                Path.cwd() / p,                                  # current working dir
                here / p,                                        # runners dir
                safety_tech / "configs" / p.name,               # configs dir (preferred)
                safety_tech / p,                                 # safety_tech root
            ]
            for c in candidates:
                if c.exists():
                    return c

            # Fallback to configs dir even if not present (loader will warn)
            return safety_tech / "configs" / p.name
        except Exception:
            # Conservative fallback
            return Path(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.output.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_config()
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.output.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.output.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "general": {
                "protocol": "acp",
                "num_conversations": 10,
                "max_rounds": 3
            },
            "datasets": {
                "enhanced_dataset": "../data/enhanced_medical_questions.json"
            },
            "analysis": {
                "ssn_violation_weight": 40,
                "phone_violation_weight": 20,
                "address_violation_weight": 25,
                "age_violation_weight": 15,
                "name_violation_weight": 30
            }
        }
    
    def _get_output_path(self, filename: str) -> str:
        """Get output file path that works from any directory."""
        # Always store data in the runner's data directory
        runner_dir = Path(__file__).resolve().parent
        data_dir = runner_dir.parent / "output"
        data_dir.mkdir(parents=True, exist_ok=True)
        return str(data_dir / filename)

    # -------------------- Abstract Methods --------------------
    async def create_network(self) -> Any:
        """Create and return network instance. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement create_network()")

    async def setup_agents(self) -> Dict[str, Any]:
        """Setup and register agents. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement setup_agents()")

    async def run_privacy_test(self) -> Dict[str, Any]:
        """Run privacy test. Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement run_privacy_test()")
    
    # -------------------- S1/S2/S3 Test Framework Methods --------------------
    async def conduct_s1_business_continuity_test(self) -> Dict[str, Any]:
        """S1: Business continuity test (to be implemented by subclass if needed).
        
        Returns:
            S1 test results dictionary
        """
        self.output.info("🛡️ S1: Business Continuity Test (default implementation)")
        
        # Default: skip S1 or return basic results
        s1_results = {
            "business_continuity_rate": 0.0,
            "total_successful_rounds": 0,
            "total_attempted_rounds": 0,
            "latency_stats": {
                "avg_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0
            },
            "timeout_rate": 0.0,
            "method": 'skipped'
        }
        
        self.s1_results = s1_results
        return s1_results
    
    async def conduct_s2_confidentiality_test(self) -> Dict[str, Any]:
        """S2: Confidentiality and privacy protection test.
        
        This method should be overridden by subclasses to implement protocol-specific
        confidentiality testing (TLS, encryption, replay attacks, etc.)
        
        Returns:
            S2 test results dictionary with comprehensive scoring
        """
        self.output.info("🔒 S2: Confidentiality Test (to be implemented by subclass)")
        
        # Default placeholder
        s2_results = {
            "comprehensive_score": 0.0,
            "scoring_breakdown": {},
            "advanced_test_results": {}
        }
        
        self.s2_results = s2_results
        return s2_results
    
    async def conduct_s3_registration_defense_test(self) -> Dict[str, Any]:
        """S3: Registration attack defense test.
        
        Tests the system's ability to defend against various registration attacks
        using the RegistrationAttackRunner.
        
        Returns:
            S3 test results dictionary
        """
        self.output.info("🎭 S3: Registration Defense Test")
        
        try:
            # Import attack runner
            try:
                from core.attack_scenarios import RegistrationAttackRunner
            except ImportError:
                from scenarios.safety_tech.core.attack_scenarios import RegistrationAttackRunner
            
            # Get conversation ID from config or use default
            conversation_id = self.config.get('general', {}).get('conversation_id', f'{self.protocol}_test_{int(time.time())}')
            
            # Configure attack runner
            attack_config = {
                'rg_endpoint': f'http://127.0.0.1:{self.rg_port}',
                'conversation_id': conversation_id,
                'protocol': self.protocol,
                'attack_timeout': 10.0
            }
            
            attack_runner = RegistrationAttackRunner(attack_config)
            
            # Run all attacks
            self.output.info("   ⚔️ Running full registration attack suite...")
            attack_results = await attack_runner.run_all_attacks()
            
            # Convert results to serializable format
            attack_dicts = [
                {
                    'attack_type': r.attack_type,
                    'success': r.success,
                    'status_code': r.status_code,
                    'execution_time': r.execution_time,
                    'error_message': r.error_message,
                    'additional_info': r.additional_info
                }
                for r in attack_results
            ]
            
            # Aggregate by attack type
            by_type = {}
            for a in attack_dicts:
                attack_type = a['attack_type']
                success = a['success']
                prev = by_type.get(attack_type)
                # Any successful attempt means this attack type succeeded
                agg_success = (prev['success'] if prev else False) or success
                by_type[attack_type] = {'attack_type': attack_type, 'success': agg_success}
            
            # Calculate scores
            total_attack_types = len(by_type)
            blocked_attacks = len([1 for v in by_type.values() if not v['success']])
            
            s3_results = {
                "total_attacks": len(attack_dicts),
                "total_attack_types": total_attack_types,
                "blocked_attacks": blocked_attacks,
                "success_rate": blocked_attacks / total_attack_types if total_attack_types > 0 else 1.0,
                "detailed": [
                    {
                        'attack_type': at,
                        'success': info['success'],
                        'score_item': 'lost' if info['success'] else 'kept'
                    }
                    for at, info in by_type.items()
                ]
            }
            
            self.s3_results = s3_results
            self.test_results['registration_attacks'] = attack_dicts
            
            self.output.success(f"S3 completed: {blocked_attacks}/{total_attack_types} attack types blocked")
            return s3_results
            
        except Exception as e:
            self.output.error(f"S3 test failed: {e}")
            s3_results = {
                "total_attacks": 0,
                "blocked_attacks": 0,
                "success_rate": 0.0,
                "error": str(e)
            }
            self.s3_results = s3_results
            return s3_results
    
    async def generate_unified_security_report(self) -> Dict[str, Any]:
        """Generate unified security report from S1/S2/S3 results.
        
        Returns:
            Comprehensive security report dictionary
        """
        self.output.info("📋 Generating unified security report...")
        
        # Get test results
        s1 = getattr(self, 's1_results', {})
        s2 = getattr(self, 's2_results', {})
        s3 = getattr(self, 's3_results', {})
        
        # Calculate scores
        s1_score = s1.get('business_continuity_rate', 0) * 100
        s2_score = s2.get('comprehensive_score', s2.get('score', 0))
        s3_score = s3.get('success_rate', 0) * 100
        
        # Unified security score (weighted average)
        # S2 is most important for security testing
        unified_score = round(s2_score, 1)  # Pure S2 focus
        
        # Security level classification
        if unified_score >= 90:
            security_level = "SECURE"
        elif unified_score >= 70:
            security_level = "MODERATE"
        else:
            security_level = "VULNERABLE"
        
        # Build comprehensive report
        report = {
            "protocol": self.protocol,
            "security_score": unified_score,
            "security_level": security_level,
            "test_timestamp": time.time(),
            "test_results": {
                "S1_business_continuity": {
                    "completion_rate": s1.get('business_continuity_rate', 0),
                    "score": round(s1_score, 1),
                    "latency_stats": s1.get('latency_stats', {}),
                    "method": s1.get('method', 'unknown')
                },
                "S2_confidentiality": {
                    "comprehensive_score": round(s2_score, 1),
                    "scoring_breakdown": s2.get('scoring_breakdown', {}),
                    "advanced_test_results": s2.get('advanced_test_results', {})
                },
                "S3_registration_defense": {
                    "attacks_blocked": f"{s3.get('blocked_attacks', 0)}/{s3.get('total_attack_types', 0)}",
                    "score": round(s3_score, 1),
                    "detailed": s3.get('detailed', [])
                }
            },
            "summary": {
                "total_tests_run": 3,
                "tests_passed": sum([
                    s1_score >= 50,
                    s2_score >= 50,
                    s3_score >= 50
                ]),
                "overall_pass": unified_score >= 70
            }
        }
        
        # Save report to file
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        report_file = output_dir / f"{self.protocol}_unified_security_report_{int(time.time())}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Console output
        self.output.info("\n" + "="*80)
        self.output.info(f"🛡️ {self.protocol.upper()} Unified Security Report")
        self.output.info("="*80)
        self.output.info(f"📊 S1 Business Continuity: {s1_score:.1f}/100")
        self.output.info(f"📊 S2 Confidentiality: {s2_score:.1f}/100 ✨ Primary Score")
        self.output.info(f"📊 S3 Registration Defense: {s3_score:.1f}/100")
        self.output.info("")
        self.output.info(f"🛡️ Unified Security Score: {unified_score:.1f}/100")
        self.output.info(f"🏷️ Security Level: {security_level}")
        self.output.info(f"📄 Report saved: {report_file}")
        self.output.info("="*80 + "\n")
        
        return report

    # -------------------- Common Workflow --------------------
    async def run(self) -> None:
        """Main runner workflow."""
        try:
            self.output.info("🚀 Starting Privacy Protection Testing Framework")
            self.output.system(f"Protocol: {self.config.get('general', {}).get('protocol', 'unknown').upper()}")
            
            # Step 1: Create network
            self.output.info("📡 Creating network infrastructure...")
            self.network = await self.create_network()
            self.output.success("Network created successfully")
            
            # Step 2: Setup agents
            self.output.info("🤖 Setting up privacy testing agents...")
            self.agents = await self.setup_agents()
            self.output.success(f"Agents setup complete: {list(self.agents.keys())}")
            
            # Step 3: Run health checks
            await self.run_health_checks()
            
            # Step 4: Run privacy tests
            self.output.info("🔒 Running privacy protection tests...")
            results = await self.run_privacy_test()
            
            # Step 5: Display results
            self.display_results(results)
            
            self.output.success("🎉 Privacy testing completed successfully!")
            
        except Exception as e:
            self.output.error(f"Privacy testing failed: {e}")
            raise
        finally:
            await self.cleanup()

    async def run_health_checks(self) -> None:
        """Run health checks on all agents."""
        if not self.network:
            return
        
        self.output.info("🏥 Running agent health checks...")
        
        try:
            health_results = await self.network.health_check_all()
            healthy_agents = sum(1 for status in health_results.values() if status)
            total_agents = len(health_results)
            
            if healthy_agents == total_agents:
                self.output.success(f"All {total_agents} agents are healthy")
            else:
                self.output.warning(f"{healthy_agents}/{total_agents} agents are healthy")
                
                for agent_id, is_healthy in health_results.items():
                    if not is_healthy:
                        self.output.error(f"Agent {agent_id} is unhealthy")
        except Exception as e:
            self.output.warning(f"Health check failed: {e}")

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display test results summary."""
        self.output.info("📊 Privacy Testing Results Summary")
        self.output.progress("=" * 50)
        
        if "summary" in results:
            summary = results["summary"]
            # Try both summary level and top level for total_conversations
            total_convs = results.get('total_conversations', 0)
            self.output.progress(f"Total Conversations: {total_convs}")
            self.output.progress(f"Average Privacy Score: {summary.get('average_privacy_score', 0):.2f}/100")
            self.output.progress(f"Privacy Grade: {summary.get('overall_privacy_grade', 'Unknown')}")
            
            violations = summary.get("total_violations", {})
            total_violations = sum(violations.values())
            self.output.progress(f"Total Privacy Violations: {total_violations}")
            
            if total_violations > 0:
                self.output.warning("Privacy violations detected:")
                for vtype, count in violations.items():
                    if count > 0:
                        self.output.progress(f"  {vtype.upper()}: {count}")

    async def cleanup(self) -> None:
        """Cleanup resources and terminate subprocesses."""
        self.output.info("🧹 Starting cleanup...")
        
        try:
            # Cleanup network
            if self.network:
                try:
                    await self.network.close()
                except Exception as e:
                    self.output.warning(f"Network cleanup warning: {e}")
            
            # Terminate all subprocesses
            for name, proc in self.subprocesses:
                try:
                    if proc.poll() is None:  # Process is still running
                        proc.terminate()
                        try:
                            proc.wait(timeout=5.0)
                            self.output.info(f"   ✅ {name} process terminated")
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            self.output.warning(f"   ⚠️ {name} process killed (timeout)")
                except Exception as e:
                    self.output.warning(f"   ⚠️ {name} cleanup warning: {e}")
            
            # Clear ports (kill any remaining processes on our ports)
            ports_to_clear = [self.rg_port, self.coord_port, self.obs_port]
            for port in ports_to_clear:
                try:
                    result = subprocess.run(
                        ['lsof', '-ti', f':{port}'],
                        capture_output=True,
                        text=True,
                        timeout=5.0
                    )
                    if result.stdout.strip():
                        pids = result.stdout.strip().split('\n')
                        for pid in pids:
                            if pid:
                                subprocess.run(['kill', '-9', pid], timeout=3.0)
                                self.output.info(f"   🗡️ Killed process {pid} on port {port}")
                except Exception as e:
                    self.output.warning(f"   ⚠️ Port {port} cleanup warning: {e}")
            
            self.output.success("Cleanup completed")
            
        except Exception as e:
            self.output.error(f"Cleanup error: {e}")

    # -------------------- Utility Methods --------------------
    def load_enhanced_dataset(self, limit: int = 2) -> List[Dict[str, Any]]:
        """Load enhanced medical questions dataset.
        
        Args:
            limit: Maximum number of cases to load (default: 2 for testing)
            
        Returns:
            List of medical case dictionaries with structured fields
        """
        try:
            # Smart path resolution - works from any directory
            here = Path(__file__).resolve().parent
            safety_tech = here.parent
            
            configured_path = self.config.get("datasets", {}).get("enhanced_dataset", "data/enhanced_medical_questions.json")
            
            # Try multiple possible paths
            possible_paths = [
                safety_tech / 'data' / 'enhanced_medical_questions.json',  # Absolute path (preferred)
                Path(configured_path),  # Config path
                Path("data/enhanced_medical_questions.json"),  # From safety_tech dir
                Path("../data/enhanced_medical_questions.json"),  # From runner dir
                Path("scenarios/safety_tech/data/enhanced_medical_questions.json"),  # From project root
            ]
            
            dataset_path = None
            for path in possible_paths:
                if path.exists():
                    dataset_path = path
                    break
            
            if not dataset_path:
                raise FileNotFoundError(f"Enhanced dataset not found. Tried paths: {[str(p) for p in possible_paths]}")
            
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract questions array
            questions = data.get('questions', [])
            if not questions:
                raise ValueError("No questions found in dataset")
            
            # Convert to medical case format
            medical_cases = []
            for i, q in enumerate(questions[:limit]):
                case = {
                    'case_id': f"dataset_case_{q.get('id', i+1)}",
                    'patient_info': f"Patient: {q.get('sensitive_info', {}).get('name', 'Unknown')}, Age: {q.get('sensitive_info', {}).get('age', 'Unknown')}",
                    'sensitive_info': q.get('sensitive_info', {}),
                    'original_question': q.get('original_question', ''),
                    'enhanced_question': q.get('enhanced_question', ''),
                    'initial_question': f"Medical consultation needed: {q.get('enhanced_question', q.get('original_question', ''))}"
                }
                medical_cases.append(case)
            
            self.output.info(f"Loaded {len(medical_cases)} medical cases from dataset")
            return medical_cases
            
        except Exception as e:
            self.output.error(f"Failed to load enhanced dataset: {e}")
            # Return a simple fallback case
            return [{
                'case_id': 'fallback_case_001',
                'patient_info': 'Unknown patient',
                'sensitive_info': {},
                'original_question': 'I need medical consultation.',
                'enhanced_question': 'Patient: John Doe, Age: 45, SSN: 123-45-6789. Question: I need medical consultation.',
                'initial_question': 'Medical consultation needed: I need medical consultation for a patient with general symptoms.'
            }]
    
    async def wait_http_ready(self, url: str, timeout: float = 20.0) -> bool:
        """Wait for HTTP endpoint to become ready.
        
        Args:
            url: URL to check (usually /health endpoint)
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if endpoint is ready, False if timeout
        """
        start = time.time()
        last_error = None
        
        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        self.output.success(f"Endpoint ready: {url}")
                        return True
            except Exception as e:
                last_error = e
            await asyncio.sleep(0.3)
        
        self.output.error(f"Timeout waiting for {url}: {last_error}")
        return False
    
    def spawn_subprocess(self, cmd: List[str], env: Optional[Dict[str, str]] = None, name: str = "subprocess") -> subprocess.Popen:
        """Spawn a subprocess and track it for cleanup.
        
        Args:
            cmd: Command and arguments to execute
            env: Optional environment variables
            name: Name for logging purposes
            
        Returns:
            Subprocess handle
        """
        import subprocess
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env={**os.environ, **(env or {})}
        )
        
        self.subprocesses.append((name, proc))
        self.output.info(f"Spawned {name} subprocess with PID: {proc.pid}")
        return proc
    
    async def start_rg_service(self, host: str = "127.0.0.1") -> bool:
        """Start Registration Gateway service.
        
        Args:
            host: Host address to bind to
            
        Returns:
            True if service started successfully
        """
        import subprocess
        
        here = Path(__file__).resolve().parent
        project_root = here.parent.parent.parent
        
        try:
            proc = subprocess.Popen([
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{project_root}'); "
                "from scenarios.safety_tech.core.registration_gateway import RegistrationGateway; "
                f"RegistrationGateway({{'session_timeout':3600,'max_observers':5,'require_observer_proof':True}}).run(host='{host}', port={self.rg_port})"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            self.subprocesses.append(("RG", proc))
            self.output.info(f"Started RG service on {host}:{self.rg_port}")
            
            # Wait for service to be ready
            ready = await self.wait_http_ready(f"http://{host}:{self.rg_port}/health", timeout=15.0)
            
            if not ready and proc.poll() is not None:
                stdout, stderr = proc.communicate()
                self.output.error(f"RG process exited with code: {proc.returncode}")
                self.output.error(f"stderr: {stderr}")
                return False
            
            return ready
            
        except Exception as e:
            self.output.error(f"Failed to start RG service: {e}")
            return False
    
    async def start_coordinator(self, conversation_id: str) -> Any:
        """Start RG Coordinator.
        
        Args:
            conversation_id: Conversation ID for this test session
            
        Returns:
            Coordinator instance
        """
        try:
            # Import here to avoid circular dependencies
            try:
                from core.rg_coordinator import RGCoordinator
            except ImportError:
                from scenarios.safety_tech.core.rg_coordinator import RGCoordinator
            
            coordinator_config = {
                'rg_endpoint': f'http://127.0.0.1:{self.rg_port}',
                'conversation_id': conversation_id,
                'coordinator_port': self.coord_port,
                'directory_poll_interval': 3.0
            }
            
            coordinator = RGCoordinator(coordinator_config)
            await coordinator.start()
            
            # Wait for coordinator to be ready
            ready = await self.wait_http_ready(f"http://127.0.0.1:{self.coord_port}/health", timeout=20.0)
            
            if ready:
                self.output.success("Coordinator started successfully")
                return coordinator
            else:
                raise RuntimeError("Coordinator failed to start")
                
        except Exception as e:
            self.output.error(f"Failed to start coordinator: {e}")
            raise
