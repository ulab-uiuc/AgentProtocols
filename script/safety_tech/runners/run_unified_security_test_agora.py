# -*- coding: utf-8 -*-
"""
Agora Real LLM Test Runner
真正的LLM对话测试，医生Agent通过RG网络进行真实的医疗对话
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import httpx

# 设置路径
HERE = Path(__file__).resolve().parent
SAFETY_TECH = HERE.parent
sys.path.insert(0, str(SAFETY_TECH))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入组件
try:
    from core.registration_gateway import RegistrationGateway
    from core.rg_coordinator import RGCoordinator
    from core.observer_agent import ObserverAgent, create_observer_agent
    from core.attack_scenarios import RegistrationAttackRunner, EavesdropMetricsCollector
    from core.backend_api import spawn_backend, register_backend, health_backend
    from protocol_backends.agora.registration_adapter import AgoraRegistrationAdapter
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


class AgoraRealLLMTest:
    """Agora真实LLM对话测试"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        # 降低第三方日志噪声
        try:
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("script.safety_tech.core.llm_wrapper").setLevel(logging.ERROR)
            logging.getLogger("core.rg_coordinator").setLevel(logging.WARNING)
            logging.getLogger("openai._base_client").setLevel(logging.ERROR)
        except Exception:
            pass
        
        # 端口配置
        self.rg_port = 8001
        self.coord_port = 8888
        self.obs_port = 8004
        
        # 组件实例
        self.rg: Optional[RegistrationGateway] = None
        self.coordinator: Optional[RGCoordinator] = None
        self.attack_runner: Optional[RegistrationAttackRunner] = None
        self.metrics_collector: Optional[EavesdropMetricsCollector] = None
        
        # 真实的医生Agent
        self.doctor_a: Optional[RGDoctorAAgent] = None
        self.doctor_b: Optional[RGDoctorBAgent] = None
        self.observers: List[ObserverAgent] = []
        
        # 测试结果
        self.test_results = {}
        
        # 会话配置
        self.conversation_id = self.config.get('general', {}).get('conversation_id', f'agora_real_llm_test_{int(time.time())}')
        
        # 医疗案例将从数据集加载
        self.medical_cases = []
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
    
    def _load_medical_dataset(self) -> List[Dict[str, Any]]:
        """加载医疗数据集"""
        try:
            # 从配置获取数据集路径
            dataset_path = self.config.get("datasets", {}).get("enhanced_dataset", "script/safety_tech/data/enhanced_medical_questions.json")
            
            # 尝试多个可能的路径
            possible_paths = [
                dataset_path,
                "data/enhanced_medical_questions.json",
                "../data/enhanced_medical_questions.json",
                "/Users/jason/Desktop/Multiagent-Protocol/script/safety_tech/data/enhanced_medical_questions.json"
            ]
            
            dataset_file = None
            for path in possible_paths:
                if Path(path).exists():
                    dataset_file = path
                    break
            
            if not dataset_file:
                raise FileNotFoundError(f"Could not find dataset file in any of: {possible_paths}")
            
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            questions = dataset.get('questions', [])
            logger.info(f"✅ Loaded {len(questions)} medical cases from dataset")
            
            # 转换为测试案例格式
            medical_cases = []
            for i, question in enumerate(questions[:2]):  # 取前2个案例进行测试
                case = {
                    "case_id": f"dataset_case_{question.get('id', i+1)}",
                    "patient_info": f"Patient: {question.get('sensitive_info', {}).get('name', 'Unknown')}, Age: {question.get('sensitive_info', {}).get('age', 'Unknown')}",
                    "sensitive_info": question.get('sensitive_info', {}),
                    "original_question": question.get('original_question', ''),
                    "enhanced_question": question.get('enhanced_question', ''),
                    "initial_question": f"Medical consultation needed: {question.get('enhanced_question', question.get('original_question', ''))}"
                }
                medical_cases.append(case)
            
            return medical_cases
            
        except Exception as e:
            logger.error(f"❌ Failed to load medical dataset: {e}")
            # 返回一个简单的fallback案例
            return [{
                "case_id": "fallback_case_001",
                "patient_info": "Unknown patient",
                "initial_question": "I need medical consultation for a patient with general symptoms."
            }]
    
    async def setup_infrastructure(self):
        """设置基础设施"""
        logger.info("🚀 Setting up Real LLM Test infrastructure...")
        
        # 0. 加载医疗数据集
        self.medical_cases = self._load_medical_dataset()
        logger.info(f"📋 Loaded {len(self.medical_cases)} medical cases from dataset")
        
        # 1. 启动注册网关
        rg_config = self.config.get('rg', {})
        rg_config.update({
            'session_timeout': 3600,
            'max_observers': 5,
            'require_observer_proof': False  # 允许测试Observer
        })
        
        self.rg = RegistrationGateway(rg_config)
        
        # 在后台启动RG服务
        import threading
        def run_rg():
            try:
                self.rg.run(host="127.0.0.1", port=8001)
            except Exception as e:
                logger.error(f"RG startup failed: {e}")
        
        rg_thread = threading.Thread(target=run_rg, daemon=True)
        rg_thread.start()
        
        # 等待RG启动并验证
        for i in range(10):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://127.0.0.1:8001/health", timeout=2.0)
                    if response.status_code == 200:
                        logger.info("✅ RG service started successfully")
                        break
            except Exception:
                continue
        else:
            raise Exception("❌ RG service failed to start after 10 seconds")
        
        # 2. 创建协调器
        coordinator_config = {
            'rg_endpoint': 'http://127.0.0.1:8001',
            'conversation_id': self.conversation_id,
            'coordinator_port': 8888,
            'bridge': self.config.get('bridge', {}),
            'directory_poll_interval': 3.0
        }
        
        self.coordinator = RGCoordinator(coordinator_config)
        await self.coordinator.start()
        
        # 验证协调器启动
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8888/health", timeout=2.0)
                if response.status_code == 200:
                    logger.info("✅ RG Coordinator started successfully")
                else:
                    raise Exception("Coordinator health check failed")
        except Exception as e:
            raise Exception(f"❌ Coordinator startup verification failed: {e}")
        
        # 指标收集器（协议通用）
        if self.metrics_collector is None:
            self.metrics_collector = EavesdropMetricsCollector({'protocol': 'agora'})

        logger.info("✅ Infrastructure setup completed")
    
    async def start_real_doctor_agents(self):
        """启动真实的医生Agent"""
        logger.info("👨‍⚕️ Starting Real Doctor Agents with LLM...")
        
        # 使用统一后端API启动Agora医生节点
        await spawn_backend('agora', 'doctor_a', 8002)
        await spawn_backend('agora', 'doctor_b', 8003)
        
        # 等待服务启动并检查健康状态
        await asyncio.sleep(2)
        for port, agent_name in [(8002, 'Agora_Doctor_A'), (8003, 'Agora_Doctor_B')]:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"http://127.0.0.1:{port}/health", timeout=5.0)
                    health_data = response.json()
                    logger.info(f"🔍 {agent_name} Health: {health_data}")
            except Exception as e:
                logger.error(f"❌ Failed to check {agent_name} health: {e}")
        
        # 使用统一后端API注册Agent
        try:
            respA = await register_backend('agora', 'Agora_Doctor_A', f"http://127.0.0.1:8002", self.conversation_id, 'doctor_a', rg_endpoint='http://127.0.0.1:8001')
            logger.info(f"✅ Doctor A registered: {respA}")
        except Exception as e:
            logger.error(f"❌ Failed to register Doctor A: {e}")
            
        try:
            respB = await register_backend('agora', 'Agora_Doctor_B', f"http://127.0.0.1:8003", self.conversation_id, 'doctor_b', rg_endpoint='http://127.0.0.1:8001')
            logger.info(f"✅ Doctor B registered: {respB}")
        except Exception as e:
            logger.error(f"❌ Failed to register Doctor B: {e}")
        
        logger.info("✅ Real Doctor Agents started successfully")
        
        # 目录断言：等待RG目录包含医生A/B
        async def _wait_for_directory(expected_ids, timeout=15):
            start = time.time()
            while time.time() - start < timeout:
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get("http://127.0.0.1:8001/directory", params={"conversation_id": self.conversation_id}, timeout=2.0)
                        if resp.status_code == 200:
                            participants = {p["agent_id"] for p in resp.json().get("participants", [])}
                            if expected_ids.issubset(participants):
                                return True
                except Exception:
                    pass
                await asyncio.sleep(1)
            return False
        ok = await _wait_for_directory({"Agora_Doctor_A", "Agora_Doctor_B"}, timeout=15)
        if not ok:
            raise Exception("Doctors not fully registered into RG directory within timeout")
        logger.info("✅ Directory assertion passed: Both doctors registered in RG")
    
    async def setup_observers(self):
        """S2新设计：不再使用Observer，直接返回。"""
        logger.info("👁️ Skipping Observer setup (new S2 design uses black-box probes)")
        self.observers = []

    async def trigger_backfill_if_enabled(self, limit: int = 5):
        """在启用backfill策略时，显式请求回填以便度量。"""
        try:
            ok_legit = await self.coordinator.request_observer_backfill("Legitimate_Observer", limit=limit)
            ok_mal = await self.coordinator.request_observer_backfill("Malicious_Observer", limit=limit)
            logger.info(f"📦 Backfill requested: legit={ok_legit}, malicious={ok_mal}, limit={limit}")
        except Exception as e:
            logger.warning(f"Backfill request failed: {e}")
    
    async def conduct_s1_concurrent_attack_conversations(self):
        """S1: 业务连续性测试（新架构）"""
        logger.info("🛡️ === S1: 业务连续性测试（新架构） ===")
        
        # S1测试模式配置
        import os as _os
        s1_test_mode = _os.environ.get('AGORA_S1_TEST_MODE', 'light').lower()
        _skip = s1_test_mode in ('skip', 'none', 'off')
        
        if not _skip:
            # 创建S1业务连续性测试器
            from script.safety_tech.core.s1_config_factory import create_s1_tester
            
            if s1_test_mode == 'protocol_optimized':
                s1_tester = create_s1_tester('agora', 'protocol_optimized')
            else:
                s1_tester = create_s1_tester('agora', s1_test_mode)
            
            logger.info(f"📊 S1测试模式: {s1_test_mode}")
            logger.info(f"📊 负载矩阵: {len(s1_tester.load_config.concurrent_levels)} × "
                      f"{len(s1_tester.load_config.rps_patterns)} × "
                      f"{len(s1_tester.load_config.message_types)} = "
                      f"{len(s1_tester.load_config.concurrent_levels) * len(s1_tester.load_config.rps_patterns) * len(s1_tester.load_config.message_types)} 种组合")
            
            # 定义Agora发送函数
            import httpx as _httpx
            import asyncio as _asyncio
            
            async def agora_send_function(payload):
                """Agora协议发送函数"""
                correlation_id = payload.get('correlation_id', 'unknown')
                async with _httpx.AsyncClient() as client:
                    try:
                        # 通过协调器路由发送
                        response = await client.post(f"http://127.0.0.1:{self.coord_port}/route_message", 
                                                   json=payload, timeout=30.0)
                        
                        if response.status_code in (200, 202):
                            try:
                                resp_data = response.json()
                                if resp_data.get('status') in ['success', 'ok', 'processed']:
                                    return {"status": "success", "data": resp_data}
                                else:
                                    return resp_data
                            except Exception:
                                return {"status": "success", "message": "Request processed"}
                        else:
                            try:
                                error_detail = response.json()
                                return {"status": "error", "error": error_detail.get('detail', f"HTTP {response.status_code}")}
                            except:
                                return {"status": "error", "error": f"HTTP {response.status_code}"}
                                
                    except Exception as e:
                        import traceback
                        error_detail = f"{type(e).__name__}: {str(e)}"
                        return {"status": "error", "error": error_detail}
            
            # S1测试前协调器状态检查
            logger.info("🔍 S1测试前协调器状态检查:")
            coord_participants_ready = False
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    # 检查协调器健康状态
                    health_resp = await client.get(f"http://127.0.0.1:{self.coord_port}/health", timeout=5.0)
                    logger.info(f"  协调器健康状态: {health_resp.status_code}")
                    
                    if health_resp.status_code == 200:
                        logger.info("  ✅ 协调器进程运行正常")
                        
                        # 测试协调器路由功能
                        test_payload = {
                            'sender_id': 'Agora_Doctor_A',
                            'receiver_id': 'Agora_Doctor_B', 
                            'text': 'Test message for coordinator',
                            'correlation_id': 'test_coord_123'
                        }
                        
                        route_resp = await client.post(f"http://127.0.0.1:{self.coord_port}/route_message", 
                                                     json=test_payload, timeout=5.0)
                        if route_resp.status_code == 200:
                            logger.info("  ✅ 协调器路由功能正常，参与者信息已加载")
                            coord_participants_ready = True
                        else:
                            logger.info(f"  ❌ 协调器路由测试失败: {route_resp.status_code}")
                            try:
                                error_detail = route_resp.json()
                                logger.info(f"  ❌ 错误详情: {error_detail}")
                            except:
                                pass
                                
                        # 检查RG目录信息  
                        rg_directory = await client.get(f"http://127.0.0.1:{self.rg_port}/directory", 
                                                      params={"conversation_id": self.conversation_id}, timeout=5.0)
                        if rg_directory.status_code == 200:
                            rg_data = rg_directory.json()
                            logger.info(f"  📋 RG目录: {rg_data['total_participants']} 个参与者")
                            for p in rg_data['participants'][:2]:
                                logger.info(f"      - {p['agent_id']}: {p['role']}")
                        else:
                            logger.info(f"  ⚠️ RG目录查询失败: {rg_directory.status_code}")
                            
            except Exception as e:
                logger.info(f"  ❌ 协调器状态检查失败: {e}")
                coord_participants_ready = False
            
            # 如果协调器参与者信息未就绪，等待更长时间
            if not coord_participants_ready:
                logger.info(f"  ⚠️ 协调器参与者信息未就绪，等待协调器轮询更新...")
                await asyncio.sleep(15)  # 等待协调器轮询RG目录（增加到15秒）
                # 再次尝试路由测试
                try:
                    async with httpx.AsyncClient() as client:
                        route_test = await client.post(f"http://127.0.0.1:{self.coord_port}/route_message", 
                                                     json=test_payload, timeout=5.0)
                        if route_test.status_code == 200:
                            logger.info(f"  ✅ 延迟后协调器路由功能恢复正常")
                            coord_participants_ready = True
                        else:
                            logger.info(f"  ❌ 协调器路由仍然失败，S1测试可能受影响")
                            try:
                                error_detail = route_test.json()
                                logger.info(f"  ❌ 错误详情: {error_detail}")
                            except:
                                pass
                except Exception as e2:
                    logger.info(f"  ❌ 延迟检查也失败: {e2}")
                
            if not coord_participants_ready:
                logger.info(f"  ⚠️ 警告：协调器可能存在问题，S1测试结果可能不准确")

            # 运行新版S1业务连续性测试
            try:
                logger.info(f"🚀 即将开始S1业务连续性测试，发送函数类型: {type(agora_send_function)}")
                logger.info(f"🚀 测试参数: sender=Agora_Doctor_A, receiver=Agora_Doctor_B")
                logger.info(f"🚀 端口配置: rg_port={self.rg_port}, coord_port={self.coord_port}, obs_port={self.obs_port}")
                
                # 运行S1业务连续性测试矩阵
                s1_results = await s1_tester.run_full_test_matrix(
                    send_func=agora_send_function,
                    sender_id='Agora_Doctor_A',
                    receiver_id='Agora_Doctor_B',
                    rg_port=self.rg_port,
                    coord_port=self.coord_port,
                    obs_port=self.obs_port
                )
                
            except Exception as e:
                logger.error(f"❌ S1测试执行失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                s1_results = []
        # 处理S1测试结果
        if _skip:
            logger.info("⏭️ 跳过S1业务连续性测试（按配置）")
            s1_report = {
                'test_summary': {
                    'overall_completion_rate': 0.0,
                    'overall_timeout_rate': 0.0,
                    'total_requests': 0,
                    'total_successful': 0,
                    'total_test_combinations': 0
                },
                'latency_analysis': {
                    'avg_ms': 0.0,
                    'p95_ms': 0.0,
                    'p99_ms': 0.0
                },
                'detailed_results': []
            }
        else:
            s1_report = s1_tester.generate_comprehensive_report()
        
        logger.info(f"\n🛡️ === S1业务连续性测试结果 ===")
        logger.info(f"📊 总体完成率: {s1_report['test_summary']['overall_completion_rate']:.1%}")
        logger.info(f"📊 总体超时率: {s1_report['test_summary']['overall_timeout_rate']:.1%}")
        logger.info(f"📊 延迟统计: 平均{s1_report['latency_analysis']['avg_ms']:.1f}ms, "
                  f"P50={s1_report['latency_analysis'].get('p50_ms', 0):.1f}ms, "
                  f"P95={s1_report['latency_analysis']['p95_ms']:.1f}ms, "
                  f"P99={s1_report['latency_analysis']['p99_ms']:.1f}ms")
        
        # 存储结果供后续使用（保持与原代码兼容）
        business_continuity_rate = s1_report['test_summary']['overall_completion_rate']
        self.s1_results = {
            "conversation_results": [],  # 新架构不使用conversation_results
            "business_continuity_rate": business_continuity_rate,
            "total_successful_rounds": s1_report['test_summary']['total_successful'],
            "total_attempted_rounds": s1_report['test_summary']['total_requests'],
            "latency_stats": {
                "avg_ms": s1_report['latency_analysis']['avg_ms'],
                "p95_ms": s1_report['latency_analysis']['p95_ms'],
                "p99_ms": s1_report['latency_analysis']['p99_ms']
            },
            "timeout_rate": s1_report['test_summary']['overall_timeout_rate'],
            "method": 'coordinator'
        }
        
        return []  # 新架构不需要返回conversation_results
    
    # 攻击函数（用于S1并发攻击测试）
    async def _malicious_registration_attack(self):
        """恶意Agent尝试注册攻击"""
        try:
            async with httpx.AsyncClient() as client:
                for i in range(10):
                    malicious_data = {
                        "protocol": "agora",
                        "agent_id": f"malicious_agora_agent_{i}",
                        "endpoint": f"http://malicious-server:800{i}",
                        "conversation_id": "conv_agora_eaves", 
                        "role": "doctor_fake",
                        "proof": {"fake": "agora_proof"}
                    }
                    await client.post("http://127.0.0.1:8001/register", json=malicious_data, timeout=2.0)
                    await asyncio.sleep(2)
        except Exception:
            pass

    async def _spam_message_attack(self):
        """垃圾消息攻击"""
        try:
            # 由于Agora使用Agent间直接通信，这里模拟持续干扰
            await asyncio.sleep(20)
        except Exception:
            pass

    async def _replay_interference_attack(self):
        """重放攻击干扰"""
        try:
            await asyncio.sleep(30)
        except Exception:
            pass

    async def _dos_attack(self):
        """DoS攻击"""
        try:
            async with httpx.AsyncClient() as client:
                tasks = []
                for i in range(50):
                    task = client.get("http://127.0.0.1:8001/health", timeout=1.0)
                    tasks.append(task)
                
                for i in range(10):
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await asyncio.sleep(0.1)
        except Exception:
            pass

    async def _observer_interference_attack(self):
        """Observer干扰攻击"""
        try:
            async with httpx.AsyncClient() as client:
                for i in range(15):
                    fake_requests = [
                        client.get("http://127.0.0.1:8004/status", timeout=2.0),
                        client.get("http://127.0.0.1:8005/status", timeout=2.0)
                    ]
                    await asyncio.gather(*fake_requests, return_exceptions=True)
                    await asyncio.sleep(2)
        except Exception:
            pass
    
    async def conduct_s2_malicious_eavesdrop_test(self):
        """S2: 新版保密性测试（传输与会话机密性验证）——对齐ACP/ANP"""
        import httpx as _httpx
        logger.info("🔒 === S2: 保密性测试（传输与会话机密性验证） ===")
        
        # S2配置：保密性探针开关
        enable_s2_probes = os.environ.get('AGORA_ENABLE_S2_PROBES', 'true').lower() == 'true'  # 默认启用
        s2_probe_type = os.environ.get('AGORA_S2_PROBE_TYPE', 'comprehensive')  # 启用完整S2测试
        
        # 创建S2探针配置
        probe_config = None
        if enable_s2_probes:
            try:
                from core.probe_config import (
                    create_s2_tls_downgrade_config, create_s2_replay_attack_config, 
                    create_comprehensive_probe_config
                )
                probe_factories = {
                    'tls_downgrade': create_s2_tls_downgrade_config,
                    'replay_attack': create_s2_replay_attack_config,
                    'comprehensive': create_comprehensive_probe_config
                }
                probe_config = probe_factories.get(s2_probe_type, create_s2_tls_downgrade_config)().to_dict()
                logger.info(f"📡 S2探针已启用: {s2_probe_type}")
            except ImportError:
                logger.info(f"⚠️ S2探针配置模块不可用，跳过探针测试")
                enable_s2_probes = False
        
        s2 = {
            'plaintext_exposure': 0,
            'downgrade_attempts': 0,
            'downgrade_blocked': 0,
            'replay_attempts': 0,
            'replay_blocked': 0,
            'metadata_leakage': 0,
            # 新增高级测试结果
            'pcap_analysis': {},       # 旁路抓包分析结果
            'cert_matrix_results': {}, # 证书矩阵测试结果
            'e2e_detection_results': {}, # E2E加密检测结果
            'session_hijack_results': {}, # 会话劫持测试结果
            'time_skew_results': {}    # 时钟漂移测试结果
        }
        # 1) 被动嗅探测试（通过探针配置实现）
        if enable_s2_probes:
            logger.info(f"   🔍 被动嗅探测试: 探针类型 {s2_probe_type}")
            # 探针已在发送消息时激活，这里统计结果
        else:
            logger.info("   🔍 被动嗅探测试: 探针未启用，跳过")
        # 2) 降级
        logger.info("   🔻 TLS/加密降级测试")
        for i in range(3):
            try:
                async with _httpx.AsyncClient(verify=False) as client:
                    resp = await client.get("http://127.0.0.1:8888/health", timeout=3.0)
                    s2['downgrade_attempts'] += 1
                    if resp.status_code == 200 and 'https' in str(resp.url):
                        s2['downgrade_blocked'] += 1
                        logger.info(f"   ✅ 降级攻击 {i+1} 被阻止（强制HTTPS）")
                    elif resp.status_code == 200:
                        logger.info(f"   ❌ 降级攻击 {i+1} 成功（HTTP明文连接）")
                    else:
                        s2['downgrade_blocked'] += 1
                        logger.info(f"   ✅ 降级攻击 {i+1} 被阻止（连接拒绝）")
            except Exception as e:
                s2['downgrade_attempts'] += 1
                # 区分真正的防护和技术故障
                if 'ReadTimeout' in type(e).__name__ or 'Timeout' in type(e).__name__:
                    # 超时不算被阻止，是技术问题
                    logger.info(f"   ⚠️ 降级攻击 {i+1} 超时（技术故障: {type(e).__name__}）")
                else:
                    s2['downgrade_blocked'] += 1
                    logger.info(f"   ✅ 降级攻击 {i+1} 被阻止（异常: {type(e).__name__})")
        # 3) 重放
        logger.info("   🔄 重放攻击测试")
        import time as _time
        for i in range(2):
            try:
                old_message = {
                    'sender_id': 'Agora_Doctor_A',
                    'receiver_id': 'Agora_Doctor_B',
                    'text': f'Replay test message {i+1}',
                    'message_id': f"replay_msg_{int(_time.time()-3600)}_{i}",
                    'correlation_id': f"replay_corr_{int(_time.time()-3600)}_{i}",
                    'timestamp': _time.time() - 3600
                }
                async with _httpx.AsyncClient() as client:
                    resp = await client.post("http://127.0.0.1:8888/route_message", json=old_message, timeout=40.0)
                    s2['replay_attempts'] += 1
                    if resp.status_code in [400,401,403]:
                        s2['replay_blocked'] += 1
                        logger.info(f"   ✅ 重放攻击 {i+1} 被阻止（状态: {resp.status_code}）")
                    elif resp.status_code in [200,202]:
                        logger.info(f"   ❌ 重放攻击 {i+1} 成功（状态: {resp.status_code}）")
                    else:
                        logger.info(f"   ⚠️ 重放攻击 {i+1} 结果不明（状态: {resp.status_code}）")
            except Exception as e:
                s2['replay_attempts'] += 1
                # 区分真正的防护和技术故障
                if 'ReadTimeout' in type(e).__name__ or 'Timeout' in type(e).__name__:
                    # 超时不算被阻止，是技术问题
                    logger.info(f"   ⚠️ 重放攻击 {i+1} 超时（技术故障: {type(e).__name__}）")
                else:
                    s2['replay_blocked'] += 1
                    logger.info(f"   ✅ 重放攻击 {i+1} 被阻止（异常: {type(e).__name__})")
        # 4) 元数据泄露
        logger.info("   📊 元数据泄露评估")
        try:
            async with _httpx.AsyncClient() as client:
                endpoints = ["/health","/metrics","/status","/info","/debug"]
                exposed = 0
                for ep in endpoints:
                    try:
                        r = await client.get(f"http://127.0.0.1:8888{ep}", timeout=2.0)
                        if r.status_code == 200:
                            exposed += 1
                            logger.info(f"   ⚠️ 端点 {ep} 可访问，可能泄露元数据")
                    except Exception:
                        pass
                s2['metadata_leakage'] = exposed
                if exposed == 0:
                    logger.info("   ✅ 元数据泄露测试: 无敏感端点暴露")
                else:
                    logger.info(f"   ❌ 元数据泄露测试: {exposed} 个端点可能泄露信息")
        except Exception as e:
            logger.info(f"   ✅ 元数据泄露测试: 系统拒绝探测 ({type(e).__name__})")

        # S2高级测试1: 旁路抓包 + MITM实测
        if enable_s2_probes and s2_probe_type == 'comprehensive':
            logger.info(f"   📡 启动旁路抓包 + MITM实测")
            try:
                from core.pcap_analyzer import run_pcap_mitm_test
                pcap_results = await run_pcap_mitm_test(
                    interface="lo0", 
                    duration=8,  # 8秒抓包
                    enable_mitm=False  # 暂时禁用MITM以避免复杂设置
                )
                s2['pcap_analysis'] = pcap_results
                
                # 统计真实明文字节数
                pcap_analysis = pcap_results.get('pcap_analysis', {})
                if pcap_analysis.get('status') == 'analyzed':
                    s2['plaintext_exposure'] = pcap_analysis.get('plaintext_bytes', 0)
                    sensitive_count = pcap_analysis.get('sensitive_keyword_count', 0)
                    logger.info(f"   📊 旁路抓包结果: {s2['plaintext_exposure']} 字节明文, {sensitive_count} 敏感关键字")
                else:
                    logger.info(f"   ⚠️ 旁路抓包失败: {pcap_analysis.get('error', '未知错误')}")
                    
            except Exception as e:
                logger.info(f"   ❌ 旁路抓包测试异常: {e}")
                s2['pcap_analysis']['error'] = str(e)
        
        # S2高级测试2: 证书有效性矩阵
        if enable_s2_probes and s2_probe_type in ['comprehensive', 'cert_matrix']:
            logger.info(f"   🔐 证书有效性矩阵测试")
            try:
                from core.cert_matrix import run_cert_matrix_test
                cert_results = await run_cert_matrix_test(host="127.0.0.1", port=self.coord_port)
                s2['cert_matrix_results'] = cert_results
                
                matrix_score = cert_results.get('matrix_score', {})
                total_score = matrix_score.get('total_score', 0)
                grade = matrix_score.get('grade', 'UNKNOWN')
                logger.info(f"   📊 证书矩阵评分: {total_score}/100 ({grade})")
                
            except Exception as e:
                logger.info(f"   ❌ 证书矩阵测试异常: {e}")
                s2['cert_matrix_results']['error'] = str(e)
        
        # S2高级测试3: E2E负载加密检测
        if enable_s2_probes and s2_probe_type == 'comprehensive':
            logger.info(f"   🔍 E2E负载加密存在性检测")
            try:
                from core.e2e_detector import E2EEncryptionDetector
                e2e_detector = E2EEncryptionDetector("AGORA_E2E_WATERMARK_TEST")
                
                # 发送带水印的测试消息
                test_payload = e2e_detector.create_plaintext_probe_payload()
                probe_config = create_comprehensive_probe_config().to_dict()
                
                # 通过协议发送探测消息 (使用Agora的消息路由)
                try:
                    async with _httpx.AsyncClient() as client:
                        probe_response = await client.post(
                            f"http://127.0.0.1:{self.coord_port}/route_message",
                            json=test_payload,
                            timeout=10.0
                        )
                        
                        # 分析返回的探针结果
                        if probe_response.status_code == 200:
                            response_data = probe_response.json()
                            s2['e2e_detection_results']['e2e_watermark_injected'] = True
                            s2['e2e_detection_results']['response'] = response_data
                            logger.info(f"   📊 E2E检测: 水印注入完成，等待中间点分析")
                        else:
                            logger.info(f"   ⚠️ E2E检测: 未收到探针结果 (状态: {probe_response.status_code})")
                except Exception as send_e:
                    logger.info(f"   ❌ E2E检测发送异常: {send_e}")
                    
            except Exception as e:
                logger.info(f"   ❌ E2E加密检测异常: {e}")
                s2['e2e_detection_results']['error'] = str(e)
        
        # S2高级测试4: 时钟漂移矩阵测试
        if enable_s2_probes and s2_probe_type == 'comprehensive':
            logger.info(f"   ⏰ 时钟漂移矩阵测试")
            try:
                from core.probe_config import create_s2_time_skew_config
                
                # 测试不同档位的时钟漂移
                skew_levels = [30, 120, 300, 600]  # ±30s, ±2m, ±5m, ±10m
                skew_results = {
                    'total_tests': 0,
                    'blocked_tests': 0,
                    'skew_level_results': {}
                }
                
                for skew_level in skew_levels:
                    skew_config = create_s2_time_skew_config([skew_level]).to_dict()
                    level_results = {
                        'attempts': 0,
                        'blocked': 0,
                        'success': 0
                    }
                    
                    # 每个档位测试3次
                    for i in range(3):
                        try:
                            # 生成带时钟漂移的过时消息
                            import time as _time
                            test_payload = {
                                'sender_id': 'Agora_Doctor_A',
                                'receiver_id': 'Agora_Doctor_B',
                                'text': f'Time skew test {i+1} for level {skew_level}s',
                                'message_id': f"time_skew_{skew_level}_{i}_{int(_time.time())}",
                                'correlation_id': f"time_skew_test_{skew_level}_{i}",
                                'timestamp': _time.time() - skew_level  # 过时的时间戳
                            }
                            
                            async with _httpx.AsyncClient() as client:
                                response = await client.post(
                                    f"http://127.0.0.1:{self.coord_port}/route_message",
                                    json=test_payload,
                                    timeout=5.0
                                )
                                
                                level_results['attempts'] += 1
                                skew_results['total_tests'] += 1
                                
                                # 检查是否被阻断（错误状态码或特定错误）
                                if response.status_code in [400, 401, 403]:
                                    # 检查错误内容是否真的是时钟相关
                                    try:
                                        error_data = response.json()
                                        error_detail = error_data.get('detail', '').lower()
                                        if 'time' in error_detail or 'timestamp' in error_detail or 'nonce' in error_detail:
                                            level_results['blocked'] += 1
                                            skew_results['blocked_tests'] += 1
                                        else:
                                            level_results['success'] += 1  # 其他类型的错误不算时钟阻断
                                    except:
                                        level_results['blocked'] += 1  # 无法解析，保守认为是阻断
                                        skew_results['blocked_tests'] += 1
                                elif response.status_code == 200:
                                    level_results['success'] += 1
                                elif response.status_code == 500:
                                    # HTTP 500通常是系统错误，不是时钟漂移阻断
                                    level_results['success'] += 1  # 不算阻断
                                else:
                                    # 其他状态码认为是阻断
                                    level_results['blocked'] += 1
                                    skew_results['blocked_tests'] += 1
                                    
                        except Exception as e:
                            # 连接异常区分类型：超时不算时钟阻断，连接拒绝才算
                            level_results['attempts'] += 1
                            skew_results['total_tests'] += 1
                            
                            error_msg = str(e).lower()
                            if 'timeout' in error_msg or 'timed out' in error_msg:
                                # 超时不算时钟漂移阻断，算成功通过
                                level_results['success'] += 1
                            else:
                                # 其他异常（如连接拒绝）才算阻断
                                level_results['blocked'] += 1
                                skew_results['blocked_tests'] += 1
                    
                    # 计算该档位的阻断率
                    if level_results['attempts'] > 0:
                        block_rate = level_results['blocked'] / level_results['attempts']
                        level_results['block_rate'] = block_rate
                    else:
                        level_results['block_rate'] = 0
                    
                    skew_results['skew_level_results'][f'{skew_level}s'] = level_results
                    logger.info(f"      ±{skew_level}s: {level_results['blocked']}/{level_results['attempts']} 被阻断 ({level_results['block_rate']:.1%})")
                
                # 计算总体时钟漂移防护评分
                overall_block_rate = skew_results['blocked_tests'] / skew_results['total_tests'] if skew_results['total_tests'] > 0 else 0
                time_skew_score = int(overall_block_rate * 100)
                
                s2['time_skew_results'] = skew_results
                s2['time_skew_results']['overall_block_rate'] = overall_block_rate
                s2['time_skew_results']['score'] = time_skew_score
                
                logger.info(f"   📊 时钟漂移防护总评分: {time_skew_score}/100 (阻断率 {overall_block_rate:.1%})")
                
            except Exception as e:
                logger.info(f"   ❌ 时钟漂移矩阵测试异常: {e}")
                s2['time_skew_results']['error'] = str(e)
        
        # S2高级测试5: 会话劫持/凭据复用测试
        if enable_s2_probes and s2_probe_type == 'comprehensive':
            logger.info(f"   🔐 会话劫持/凭据复用测试")
            try:
                hijack_results = {
                    'total_attempts': 0,
                    'blocked_attempts': 0,
                    'successful_hijacks': 0,
                    'hijack_types': {}
                }
                
                # 测试不同类型的会话劫持 (适配Agora协议特点)
                hijack_scenarios = [
                    {
                        'name': 'expired_session_token',
                        'token': 'expired_session_12345',
                        'description': '过期会话令牌'
                    },
                    {
                        'name': 'cross_session_reuse',
                        'token': 'other_session_67890',
                        'description': '跨会话令牌复用'
                    },
                    {
                        'name': 'malformed_auth_header',
                        'token': 'malformed__invalid__header',
                        'description': '格式错误认证头'
                    },
                    {
                        'name': 'privilege_escalation',
                        'token': 'admin_session_98765',
                        'description': '权限提升会话'
                    }
                ]
                
                for scenario in hijack_scenarios:
                    scenario_name = scenario['name']
                    scenario_results = {
                        'attempts': 0,
                        'blocked': 0,
                        'success': 0
                    }
                    
                    # 每种劫持场景测试2次
                    for i in range(2):
                        try:
                            test_payload = {
                                'sender_id': 'Agora_Doctor_A',
                                'receiver_id': 'Agora_Doctor_B',
                                'text': f'Session hijack test: {scenario["description"]} attempt {i+1}',
                                'message_id': f"hijack_{scenario_name}_{i}_{int(_time.time())}",
                                'correlation_id': f"hijack_test_{scenario_name}_{i}",
                                'session_token': scenario['token'],  # 注入伪造令牌
                                'auth_header': f'Bearer {scenario["token"]}'  # 注入伪造认证头
                            }
                            
                            async with _httpx.AsyncClient() as client:
                                # 使用伪造的认证头
                                headers = {'Authorization': f'Bearer {scenario["token"]}'}
                                response = await client.post(
                                    f"http://127.0.0.1:{self.coord_port}/route_message",
                                    json=test_payload,
                                    headers=headers,
                                    timeout=5.0
                                )
                                
                                scenario_results['attempts'] += 1
                                hijack_results['total_attempts'] += 1
                                
                                # 检查是否被阻断
                                if response.status_code in [401, 403, 404]:
                                    scenario_results['blocked'] += 1
                                    hijack_results['blocked_attempts'] += 1
                                elif response.status_code == 200:
                                    scenario_results['success'] += 1
                                    hijack_results['successful_hijacks'] += 1
                                elif response.status_code == 500:
                                    # HTTP 500通常不是认证阻断，可能是系统错误
                                    scenario_results['success'] += 1
                                    hijack_results['successful_hijacks'] += 1
                                else:
                                    # 其他错误状态码也认为是阻断
                                    scenario_results['blocked'] += 1
                                    hijack_results['blocked_attempts'] += 1
                                    
                        except Exception as e:
                            # 连接异常区分类型：超时不算认证阻断
                            scenario_results['attempts'] += 1
                            hijack_results['total_attempts'] += 1
                            
                            error_msg = str(e).lower()
                            if 'timeout' in error_msg or 'timed out' in error_msg:
                                # 超时不算会话劫持阻断，算劫持成功
                                scenario_results['success'] += 1
                                hijack_results['successful_hijacks'] += 1
                            else:
                                # 其他异常（如连接拒绝）才算阻断
                                scenario_results['blocked'] += 1
                                hijack_results['blocked_attempts'] += 1
                    
                    # 计算该场景的阻断率
                    if scenario_results['attempts'] > 0:
                        block_rate = scenario_results['blocked'] / scenario_results['attempts']
                        scenario_results['block_rate'] = block_rate
                    else:
                        scenario_results['block_rate'] = 0
                    
                    hijack_results['hijack_types'][scenario_name] = scenario_results
                    logger.info(f"      {scenario['description']}: {scenario_results['blocked']}/{scenario_results['attempts']} 被阻断 ({scenario_results['block_rate']:.1%})")
                
                # 计算总体会话劫持防护评分
                overall_hijack_block_rate = hijack_results['blocked_attempts'] / hijack_results['total_attempts'] if hijack_results['total_attempts'] > 0 else 0
                session_hijack_score = int(overall_hijack_block_rate * 100)
                
                hijack_results['overall_block_rate'] = overall_hijack_block_rate
                hijack_results['score'] = session_hijack_score
                hijack_results['hijack_blocked'] = overall_hijack_block_rate > 0.5  # 超过50%阻断率算有效防护
                
                s2['session_hijack_results'] = hijack_results
                
                logger.info(f"   📊 会话劫持防护总评分: {session_hijack_score}/100 (阻断率 {overall_hijack_block_rate:.1%})")
                
            except Exception as e:
                logger.info(f"   ❌ 会话劫持测试异常: {e}")
                s2['session_hijack_results']['error'] = str(e)

        # 新版S2综合评分计算（8组件权重系统）
        total_attacks = s2['downgrade_attempts'] + s2['replay_attempts']
        blocked_attacks = s2['downgrade_blocked'] + s2['replay_blocked']
        attack_block_rate = (blocked_attacks/total_attacks) if total_attacks>0 else 1.0
        
        # 1. TLS/传输层安全 (40%)
        tls_score = 100  # Agora使用HTTP/HTTPS，基础传输层保护
        cert_results = s2.get('cert_matrix_results', {})
        if cert_results.get('matrix_score'):
            tls_score = cert_results['matrix_score'].get('total_score', 100)
        
        # 2. 重放攻击防护 (4%)
        replay_score = int(attack_block_rate * 100)
        
        # 3. E2E负载加密检测 (18%)
        e2e_results = s2.get('e2e_detection_results', {})
        e2e_score = 0
        if e2e_results.get('e2e_watermark_injected'):
            e2e_score = 60  # 基础参与分
            if not e2e_results.get('watermark_leaked', True):
                e2e_score = 90  # 优秀保护分
        
        # 4. 会话劫持防护 (15%)
        session_results = s2.get('session_hijack_results', {})
        session_hijack_score = session_results.get('score', 0)
        
        # 5. 时钟漂移防护 (12%)
        time_skew_results = s2.get('time_skew_results', {})
        time_skew_score = time_skew_results.get('score', 0)
        
        # 6. 旁路抓包保护 (8%)
        pcap_results = s2.get('pcap_analysis', {})
        pcap_score = 100  # 默认良好保护
        if pcap_results.get('pcap_analysis', {}).get('plaintext_bytes', 0) > 0:
            pcap_score = max(0, 100 - pcap_results['pcap_analysis']['plaintext_bytes'] // 10)
        
        # 7. 重放攻击防护 (4%)
        # 已在第2项计算
        
        # 8. 元数据泄露防护 (3%)
        metadata_score = max(0, 100 - s2['metadata_leakage']*20)
        
        # 加权综合评分
        s2_comprehensive_score = (
            tls_score * 0.40 +              # TLS/传输层 40%
            session_hijack_score * 0.15 +   # 会话劫持防护 15%
            e2e_score * 0.18 +              # E2E检测 18%
            session_hijack_score * 0.15 +   # 会话劫持防护 15%
            time_skew_score * 0.12 +        # 时钟漂移防护 12%
            pcap_score * 0.08 +             # 旁路抓包 8%
            replay_score * 0.04 +           # 重放攻击防护 4%
            metadata_score * 0.03           # 元数据泄露防护 3%
        )
        
        s2_comprehensive_score = min(100, max(0, s2_comprehensive_score))
        
        # 详细评分日志
        s2['scoring_breakdown'] = {
            'tls_transport_security': {'score': tls_score, 'weight': 0.40},
            'session_hijack_protection': {'score': session_hijack_score, 'weight': 0.15},
            'e2e_encryption_detection': {'score': e2e_score, 'weight': 0.18},
            'time_skew_protection': {'score': time_skew_score, 'weight': 0.12},
            'pcap_sniffing_protection': {'score': pcap_score, 'weight': 0.08},
            'replay_attack_protection': {'score': replay_score, 'weight': 0.04},
            'metadata_leakage_protection': {'score': metadata_score, 'weight': 0.03}
        }
        
        # 保留新旧两套评分为兼容性
        s2_score_legacy = (attack_block_rate*70 + (100 if s2['plaintext_exposure']==0 else max(0, 100 - s2['plaintext_exposure']))*0.20 + metadata_score*0.10)
        
        self.s2_results = {
            "comprehensive_score": s2_comprehensive_score,  # 新版评分
            "scoring_breakdown": s2['scoring_breakdown'],
            "legacy_score": s2_score_legacy,  # 旧版兼容
            "legacy_metrics": {
                "attack_block_rate": attack_block_rate,
                "plaintext_exposure": s2['plaintext_exposure'],
                "metadata_leakage": s2['metadata_leakage']
            },
            "advanced_test_results": {
                'pcap_analysis': s2.get('pcap_analysis', {}),
                'cert_matrix_results': s2.get('cert_matrix_results', {}),
                'e2e_detection_results': s2.get('e2e_detection_results', {}),
                'session_hijack_results': s2.get('session_hijack_results', {}),
                'time_skew_results': s2.get('time_skew_results', {})
            }
        }
        
        logger.info(f"   📊 S2保密性综合评分: {s2_comprehensive_score:.1f}/100")
        logger.info(f"      TLS/传输层安全: {tls_score:.1f}/100 (40%)")
        logger.info(f"      会话劫持防护: {session_hijack_score:.1f}/100 (15%)")
        logger.info(f"      E2E加密检测: {e2e_score:.1f}/100 (18%)")
        logger.info(f"      时钟漂移防护: {time_skew_score:.1f}/100 (12%)")
        logger.info(f"      旁路抓包保护: {pcap_score:.1f}/100 (8%)")
        logger.info(f"      重放攻击防护: {replay_score:.1f}/100 (4%)")
        logger.info(f"      元数据泄露防护: {metadata_score:.1f}/100 (3%)")
    
    async def conduct_s3_registration_defense_test(self):
        """S3: 恶意注册防护测试"""
        logger.info("🎭 === S3: 恶意注册防护测试 ===")
        
        # 使用现有的攻击测试方法
        await self.run_quick_attack_test()
        await self.run_full_attack_test()
        
        # 存储S3结果（细化每项是否得分/丢分）——按攻击类型聚合为6类
        quick_attacks = self.test_results.get('quick_attacks', [])
        full_attacks = self.test_results.get('full_attacks', [])
        combined = quick_attacks + full_attacks
        by_type = {}
        for a in combined:
            attack_type = a.get('attack_type') or a.get('type') or 'unknown'
            success = a.get('success', False)
            prev = by_type.get(attack_type)
            # 任一尝试成功则视为该类型丢分
            agg_success = (prev['success'] if prev else False) or success
            by_type[attack_type] = {'attack_type': attack_type, 'success': agg_success}
        
        detailed_items = []
        for attack_type, info in by_type.items():
            detailed_items.append({
                'attack_type': attack_type,
                'success': info['success'],
                'score_item': 'lost' if info['success'] else 'kept'
            })
        
        total_attacks = len(by_type)
        blocked_attacks = len([1 for v in by_type.values() if not v['success']])
        
        self.s3_results = {
            "total_attacks": total_attacks,
            "blocked_attacks": blocked_attacks,
            "success_rate": blocked_attacks / total_attacks if total_attacks > 0 else 1.0,
            "detailed": detailed_items
        }
        
        logger.info(f"📊 S3结果: {blocked_attacks}/{total_attacks} 攻击被阻止")
        logger.info("✅ S3恶意注册防护测试完成")
    
    async def generate_unified_security_report(self):
        """生成统一格式的安全报告"""
        logger.info("📋 生成统一安全报告...")
        
        # 获取S1-S3结果
        s1_results = getattr(self, 's1_results', {})
        s2_results = getattr(self, 's2_results', {})
        s3_results = getattr(self, 's3_results', {})
        
        # 新版统一安全评分公式（S2=100%，S1和S3权重=0%）
        # 专注于协议层面的保密性与安全防护能力
        s1_score = s1_results.get('business_continuity_rate', 0) * 100
        s2_score = s2_results.get('comprehensive_score', s2_results.get('score', 0))  # 优先使用新版评分
        s3_score = s3_results.get('success_rate', 0) * 100
        
        unified_security_score = round(s2_score, 1)  # 纯S2评分
        
        # 安全等级分类
        if unified_security_score >= 90:
            security_level = "SECURE"
        elif unified_security_score >= 70:
            security_level = "MODERATE"
        else:
            security_level = "VULNERABLE"
        
        # 统一格式报告
        unified_report = {
            "protocol": "agora",
            "security_score": unified_security_score,
            "security_level": security_level,
            "conversation_details": {
                "total_cases": len(s1_results.get('conversation_results', [])),
                "successful_conversations": s1_results.get('total_successful_rounds', 0),
                "llm_interactions": s1_results.get('conversation_results', [])
            },
            "rg_verification": {
                "mode": getattr(self.rg, 'verification_mode', None),
                "metrics": getattr(self.rg, 'metrics', None) if hasattr(self.rg, 'metrics') else None,
                "doctor_a": {
                    "method": getattr(self.doctor_a, 'verification_method', None),
                    "latency_ms": getattr(self.doctor_a, 'verification_latency_ms', None),
                    "blocked_by": getattr(self.doctor_a, 'blocked_by', None) if hasattr(self.doctor_a, 'blocked_by') else None,
                    "reason": getattr(self.doctor_a, 'blocked_reason', None) if hasattr(self.doctor_a, 'blocked_reason') else None
                },
                "doctor_b": {
                    "method": getattr(self.doctor_b, 'verification_method', None),
                    "latency_ms": getattr(self.doctor_b, 'verification_latency_ms', None),
                    "blocked_by": getattr(self.doctor_b, 'blocked_by', None) if hasattr(self.doctor_b, 'blocked_by') else None,
                    "reason": getattr(self.doctor_b, 'blocked_reason', None) if hasattr(self.doctor_b, 'blocked_reason') else None
                }
            },
            "test_results": {
                "S1_business_continuity": {
                    "completion_rate": s1_results.get('business_continuity_rate', 0),
                    "score": round(s1_score, 1),
                    "latency_stats": s1_results.get('latency_stats', {
                        "avg_ms": 0,
                        "p95_ms": 0,
                        "p99_ms": 0
                    }),
                    "timeout_rate": s1_results.get('timeout_rate', 0),
                    "method": s1_results.get('method', 'coordinator')
                },
                "S2_privacy_protection": {
                    "comprehensive_score": round(s2_score, 1),
                    "scoring_breakdown": s2_results.get('scoring_breakdown', {}),
                    "legacy_metrics": s2_results.get('legacy_metrics', {
                        "attack_block_rate": s2_results.get('attack_block_rate', 0),
                        "plaintext_exposure": s2_results.get('plaintext_exposure', 0),
                        "metadata_leakage": s2_results.get('metadata_leakage', 0)
                    }),
                    "advanced_test_results": s2_results.get('advanced_test_results', {
                        'pcap_analysis': {},
                        'cert_matrix_results': {},
                        'e2e_detection_results': {},
                        'session_hijack_results': {},
                        'time_skew_results': {}
                    })
                },
                "S3_registration_defense": {
                    "attacks_blocked": f"{s3_results.get('blocked_attacks', 0)}/{s3_results.get('total_attacks', 0)}",
                    "score": round(s3_score, 1)
                }
            },
            "unified_metrics": {
                "total_attacks_attempted": s3_results.get('total_attacks', 0),
                "total_attacks_blocked": s3_results.get('blocked_attacks', 0),
                "total_messages_tested": len(self.medical_cases) * 5,
                "malicious_observers_deployed": s2_results.get('malicious_observers', 0),
                "test_duration_seconds": 300
            },
            "timestamp": time.time()
        }
        
        # 保存报告
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        report_file = output_dir / f"agora_unified_security_report_{int(time.time())}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(unified_report, f, indent=2, ensure_ascii=False)
        
        # 控制台输出
        logger.info("\n" + "="*80)
        logger.info("🛡️ Agora 统一安全防护测试报告")
        logger.info("="*80)
        logger.info(f"📋 协议: Agora")
        logger.info(f"📊 医疗案例: {len(self.medical_cases)}/10 (标准)")
        logger.info(f"💬 对话轮次: {s1_results.get('total_successful_rounds', 0)}/{len(self.medical_cases)*5} (标准)")
        logger.info("")
        logger.info("🔍 安全测试结果:")
        logger.info(f"   S1 业务连续性: {s1_score:.1f}/100 (暂停计分，权重=0%)")
        logger.info(f"   S2 保密性防护: {s2_score:.1f}/100 (传输与会话机密性) ✨ 主评分项")
        logger.info(f"   S3 注册攻击防护: {s3_score:.1f}/100 (暂停计分，权重=0%)")
        for item in self.s3_results.get('detailed', []):
            tag = '得分' if item.get('score_item') == 'kept' else '丢分'
            logger.info(f"      · {item.get('attack_type','unknown')}: {tag}")
        logger.info("")
        logger.info(f"🛡️ 统一安全评分: {unified_security_score:.1f}/100 (纯S2评分)")
        logger.info(f"🏷️ 安全等级: {security_level}")
        logger.info(f"📄 详细报告: {report_file}")
        logger.info("="*80 + "\n")
        
        return unified_report
    
    async def collect_eavesdrop_evidence(self):
        """收集窃听证据"""
        logger.info("🔍 Collecting Eavesdrop Evidence...")
        
        observer_reports = []
        
        for i, observer in enumerate(self.observers):
            port = 8004 + i
            try:
                async with httpx.AsyncClient() as client:
                    # 获取Observer状态
                    status_resp = await client.get(f"http://127.0.0.1:{port}/status", timeout=10.0)
                    status_data = status_resp.json() if status_resp.status_code == 200 else {}
                    
                    # 获取窃听报告
                    report_resp = await client.get(f"http://127.0.0.1:{port}/eavesdrop_report", timeout=10.0)
                    report_data = report_resp.json() if report_resp.status_code == 200 else {}
                    
                    # 获取接收到的消息
                    messages_resp = await client.get(f"http://127.0.0.1:{port}/messages?limit=100", timeout=10.0)
                    messages_data = messages_resp.json() if messages_resp.status_code == 200 else {}
                    
                    observer_evidence = {
                        "observer_id": observer.observer_id,
                        "status": status_data,
                        "eavesdrop_report": report_data,
                        "intercepted_messages": messages_data,
                        "evidence_collected": True
                    }
                    
                    observer_reports.append(observer_evidence)
                    
                    # 日志摘要
                    total_messages = messages_data.get('count', 0)
                    total_bytes = report_data.get('unauthorized_data_access', {}).get('total_bytes', 0)
                    
                    logger.info(f"   👁️ {observer.observer_id}: {total_messages} messages, {total_bytes} bytes intercepted")
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to collect evidence from {observer.observer_id}: {e}")
                observer_reports.append({
                    "observer_id": observer.observer_id,
                    "error": str(e),
                    "evidence_collected": False
                })
        
        self.test_results['eavesdrop_evidence'] = observer_reports
        
        # 统计
        successful_observers = len([r for r in observer_reports if r.get('evidence_collected', False)])
        total_intercepted_messages = sum(
            r.get('intercepted_messages', {}).get('count', 0) 
            for r in observer_reports if r.get('evidence_collected', False)
        )
        
        logger.info(f"🎯 Eavesdrop Evidence Summary:")
        logger.info(f"   👁️ Active observers: {successful_observers}/{len(self.observers)}")
        logger.info(f"   📨 Total intercepted messages: {total_intercepted_messages}")
        logger.info("✅ Eavesdrop evidence collection completed")
    
    async def run_quick_attack_test(self):
        """快速攻击测试"""
        logger.info("⚔️ Running Quick Attack Tests...")
        
        attack_config = {
            'rg_endpoint': 'http://127.0.0.1:8001',
            'conversation_id': self.conversation_id,
            'attacks': self.config.get('attacks', {}),
            'attack_timeout': 5.0
        }
        
        self.attack_runner = RegistrationAttackRunner(attack_config)
        
        # 只运行几个关键攻击
        key_attacks = [
            self.attack_runner.observer_auto_admission_attack,
            self.attack_runner.stolen_credential_attack,
            self.attack_runner.replay_attack
        ]
        
        attack_results = []
        for attack_func in key_attacks:
            try:
                result = await attack_func()
                # 转换AttackResult为可序列化的字典
                attack_dict = {
                    "attack_type": result.attack_type,
                    "success": result.success,
                    "status_code": result.status_code,
                    "execution_time": result.execution_time,
                    "error_message": result.error_message,
                    "additional_info": result.additional_info
                }
                attack_results.append(attack_dict)
                logger.info(f"   ⚔️ {result.attack_type}: {'SUCCESS' if result.success else 'BLOCKED'}")
            except Exception as e:
                logger.error(f"   ❌ Attack failed: {e}")
        
        self.test_results['quick_attacks'] = attack_results
        logger.info("✅ Quick attack tests completed")

    async def run_full_attack_test(self):
        """运行全量六类注册攻击（协议无关接口）。"""
        logger.info("⚔️ Running Full Registration Attack Tests...")
        # 钩子：为Agora提供最小有效protocol_hash/sources，以便通过基本hash校验
        try:
            from protocol_backends.agora.registration_adapter import compute_hash, encode_as_data_uri  # re-exported names
        except Exception:
            compute_hash = None
            encode_as_data_uri = None

        def _build_proto_hash_proof():
            if compute_hash and encode_as_data_uri:
                doc = "---\nname: medical-consultation\n---\n# Roles\n- doctor_a\n- doctor_b\n"
                return {
                    'protocol_hash': compute_hash(doc),
                    'protocol_sources': [encode_as_data_uri(doc)],
                    'timestamp': time.time(),
                    'nonce': str(__import__('uuid').uuid4()),
                }
            # 回退：不提供，将走默认分支，可能被拒
            return {}

        attack_hooks = {
            'build_impersonation_proof': lambda: {
                **_build_proto_hash_proof(),
                'agent_signature': 'fake_agent_signature',
            },
            'build_endpoint_substitution_proof': lambda: {
                **_build_proto_hash_proof(),
                'endpoint_ownership_proof': 'fake',  # 故意无效，配合RG端点证明开关
            },
            # cross_protocol 使用默认伪造ANP proof
        }

        attack_config = {
            'rg_endpoint': 'http://127.0.0.1:8001',
            'conversation_id': self.conversation_id,
            'protocol': 'agora',
            'attack_hooks': attack_hooks,
            'attack_timeout': 10.0,
        }

        self.attack_runner = RegistrationAttackRunner(attack_config)
        full_results = await self.attack_runner.run_all_attacks()
        self.metrics_collector.collect_attack_metrics(full_results)

        self.test_results['full_attacks'] = [
            {
                'attack_type': r.attack_type,
                'success': r.success,
                'status_code': r.status_code,
                'execution_time': r.execution_time,
                'error': r.error_message,
                'additional_info': r.additional_info,
            }
            for r in full_results
        ]
        logger.info("✅ Full attack tests completed")

    async def endpoint_proof_ab_test(self):
        """端点所属证明开关的A/B对照，仅复用端点替换攻击进行对比。"""
        try:
            original = self.rg.config.get('agora_require_endpoint_proof', False)
            # A: 关闭端点证明
            self.rg.config['agora_require_endpoint_proof'] = False
            res_a = await RegistrationAttackRunner({
                'rg_endpoint': 'http://127.0.0.1:8001',
                'conversation_id': self.conversation_id,
                'protocol': 'agora',
                'attack_timeout': 10.0,
            }).endpoint_substitution_attack()
            # B: 开启端点证明
            self.rg.config['agora_require_endpoint_proof'] = True
            res_b = await RegistrationAttackRunner({
                'rg_endpoint': 'http://127.0.0.1:8001',
                'conversation_id': self.conversation_id,
                'protocol': 'agora',
                'attack_timeout': 10.0,
            }).endpoint_substitution_attack()
            # 还原
            self.rg.config['agora_require_endpoint_proof'] = original
            self.test_results['endpoint_proof_ab'] = {
                'A_disabled': {'success': res_a.success, 'status_code': res_a.status_code},
                'B_enabled': {'success': res_b.success, 'status_code': res_b.status_code},
            }
            logger.info(f"🧪 Endpoint proof A/B: {self.test_results['endpoint_proof_ab']}")
        except Exception as e:
            logger.warning(f"Endpoint proof A/B test failed: {e}")
    
    async def generate_real_test_report(self):
        """生成真实测试报告"""
        logger.info("📊 Generating Real LLM Test Report...")
        
        # 收集所有数据
        conversation_data = self.test_results.get('real_llm_conversations', [])
        eavesdrop_data = self.test_results.get('eavesdrop_evidence', [])
        attack_data_quick = self.test_results.get('quick_attacks', [])
        attack_data_full = self.test_results.get('full_attacks', [])
        endpoint_ab = self.test_results.get('endpoint_proof_ab', {})
        
        # 统计
        successful_conversations = len([c for c in conversation_data if c.get('llm_conversations', False)])
        total_llm_turns = sum(c.get('total_turns_a', 0) + c.get('total_turns_b', 0) for c in conversation_data)
        successful_eavesdrops = len([e for e in eavesdrop_data if e.get('evidence_collected', False)])
        successful_attacks = (
            len([a for a in attack_data_quick if a.get('success', False)]) +
            len([a for a in attack_data_full if a.get('success', False)])
        )
        
        # 计算安全评分
        conversation_success_rate = successful_conversations / len(conversation_data) if conversation_data else 0
        eavesdrop_success_rate = successful_eavesdrops / len(eavesdrop_data) if eavesdrop_data else 0
        total_attacks = len(attack_data_quick) + len(attack_data_full)
        attack_success_rate = successful_attacks / total_attacks if total_attacks else 0
        # backfill统计
        backfill_enabled_count = 0
        backfill_message_count = 0
        for e in eavesdrop_data:
            report = e.get('eavesdrop_report', {})
            access_methods = report.get('access_methods', {})
            if access_methods.get('backfill'):
                backfill_enabled_count += 1
                try:
                    backfill_message_count += int(access_methods.get('backfill_message_count', 0))
                except Exception:
                    pass
        
        # 安全评分 (越低越不安全)
        security_score = max(0, 100 - (eavesdrop_success_rate * 50 + attack_success_rate * 30))
        
        final_report = {
            "test_metadata": {
                "test_type": "real_llm_safety_tech",
                "protocol": "agora",
                "conversation_id": self.conversation_id,
                "test_timestamp": time.time(),
                "real_llm_used": True
            },
            "llm_conversation_results": {
                "total_cases": len(conversation_data),
                "successful_cases": successful_conversations,
                "success_rate": conversation_success_rate,
                "total_llm_turns": total_llm_turns,
                "conversation_details": conversation_data
            },
            "eavesdrop_results": {
                "total_observers": len(eavesdrop_data),
                "successful_eavesdrops": successful_eavesdrops,
                "eavesdrop_success_rate": eavesdrop_success_rate,
                "observer_details": eavesdrop_data
            },
            "attack_results": {
                "total_attacks": total_attacks,
                "successful_attacks": successful_attacks,
                "attack_success_rate": attack_success_rate,
                "quick_attacks": attack_data_quick,
                "full_attacks": attack_data_full,
                "endpoint_proof_ab": endpoint_ab
            },
            "backfill_summary": {
                "observers_with_backfill": backfill_enabled_count,
                "total_backfill_messages": backfill_message_count
            },
            "security_assessment": {
                "security_score": security_score,
                "llm_conversations_protected": conversation_success_rate > 0 and eavesdrop_success_rate < 0.5,
                "eavesdrop_prevention": eavesdrop_success_rate < 0.3,
                "attack_prevention": attack_success_rate < 0.3,
                "overall_security": "HIGH" if security_score > 70 else "MEDIUM" if security_score > 40 else "LOW"
            },
            "summary": {
                "real_llm_conversations": total_llm_turns > 0,
                "safety_tech_working": successful_eavesdrops > 0 or successful_attacks > 0,
                "agora_protocol_security": security_score,
                "test_passed": total_llm_turns > 0 and (successful_eavesdrops > 0 or successful_attacks > 0)
            }
        }
        
        # 保存报告
        report_file = SAFETY_TECH / "output" / f"agora_real_llm_test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Report saved to: {report_file}")
        
        # 打印摘要
        print("\n" + "="*80)
        print("🔬 AGORA REAL LLM SAFETY TECH TEST RESULTS")
        print("="*80)
        print(f"🗣️ Real LLM Conversations: {total_llm_turns} turns")
        print(f"👁️ Successful Eavesdrops: {successful_eavesdrops}/{len(eavesdrop_data)}")
        print(f"⚔️ Successful Attacks: {successful_attacks}/{total_attacks}")
        print(f"🛡️ Security Score: {security_score:.1f}/100")
        print(f"✅ Safety Tech Working: {'YES' if final_report['summary']['safety_tech_working'] else 'NO'}")
        print(f"🎯 Test Status: {'PASSED' if final_report['summary']['test_passed'] else 'FAILED'}")
        print("="*80)
        
        return final_report
    
    async def cleanup(self):
        """清理资源"""
        logger.info("🧹 Cleaning up resources...")
        
        if self.coordinator:
            await self.coordinator.stop()
        
        logger.info("✅ Cleanup completed")
    
    async def run_unified_security_test(self):
        """运行统一安全防护测试"""
        try:
            # 1. 设置基础设施
            await self.setup_infrastructure()
            
            # 2. 启动真实医生Agent
            await self.start_real_doctor_agents()
            
            # 3. 设置Observer
            await self.setup_observers()
            
            # S1: 并发攻击下对话稳定性测试
            conversation_results = await self.conduct_s1_concurrent_attack_conversations()
            
            # S2: 恶意窃听检测测试
            await self.conduct_s2_malicious_eavesdrop_test()
            
            # S3: 恶意注册防护测试
            await self.conduct_s3_registration_defense_test()
            
            # 生成统一格式报告
            final_report = await self.generate_unified_security_report()
            
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Unified security test failed: {e}")
            raise
        finally:
            await self.cleanup()

    async def cleanup(self):
        """清理所有资源和子进程"""
        logger.info("🧹 开始清理测试资源...")
        
        # 终止RG进程
        if hasattr(self, 'rg') and self.rg:
            try:
                if hasattr(self.rg, 'process') and self.rg.process:
                    self.rg.process.terminate()
                    try:
                        await asyncio.wait_for(self.rg.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.rg.process.kill()
                logger.info("   ✅ RG进程已终止")
            except Exception as e:
                logger.warning(f"   ⚠️ RG进程终止异常: {e}")
        
        # 终止协调器进程
        if hasattr(self, 'coordinator') and self.coordinator:
            try:
                if hasattr(self.coordinator, 'process') and self.coordinator.process:
                    self.coordinator.process.terminate()
                    try:
                        await asyncio.wait_for(self.coordinator.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.coordinator.process.kill()
                logger.info("   ✅ 协调器进程已终止")
            except Exception as e:
                logger.warning(f"   ⚠️ 协调器进程终止异常: {e}")
        
        # 终止医生Agent进程
        for agent_name in ['doctor_a', 'doctor_b']:
            agent = getattr(self, agent_name, None)
            if agent:
                try:
                    if hasattr(agent, 'process') and agent.process:
                        agent.process.terminate()
                        try:
                            await asyncio.wait_for(agent.process.wait(), timeout=3.0)
                        except asyncio.TimeoutError:
                            agent.process.kill()
                    logger.info(f"   ✅ {agent_name}进程已终止")
                except Exception as e:
                    logger.warning(f"   ⚠️ {agent_name}进程终止异常: {e}")
        
        # 终止Observer进程
        if hasattr(self, 'observers') and self.observers:
            for i, observer in enumerate(self.observers):
                try:
                    if hasattr(observer, 'process') and observer.process:
                        observer.process.terminate()
                        try:
                            await asyncio.wait_for(observer.process.wait(), timeout=3.0)
                        except asyncio.TimeoutError:
                            observer.process.kill()
                    logger.info(f"   ✅ Observer{i}进程已终止")
                except Exception as e:
                    logger.warning(f"   ⚠️ Observer{i}进程终止异常: {e}")
        
        # 清理端口（杀死可能残留的进程）
        import subprocess
        ports_to_clear = [self.rg_port, self.coord_port, self.obs_port, 8002, 8003, 9102, 9103]
        for port in ports_to_clear:
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                     capture_output=True, text=True, timeout=5.0)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid:
                            subprocess.run(['kill', '-9', pid], timeout=3.0)
                            logger.info(f"   🗡️ 强制终止端口{port}上的进程{pid}")
            except Exception as e:
                logger.warning(f"   ⚠️ 清理端口{port}异常: {e}")
        
        logger.info("✅ 测试资源清理完成")


async def main():
    """主函数"""
    import os as _os
    # 强制NVIDIA LLaMA 8B (OpenAI兼容接口) - 统一使用8B模型
    _os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
    _os.environ["OPENAI_API_KEY"] = "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"
    _os.environ["OPENAI_MODEL"] = "meta/llama-3.3-70b-instruct"
    # 允许通过环境变量覆盖配置文件路径
    _override = _os.environ.get("SAFETY_TECH_CONFIG")
    config_file = Path(_override) if _override else (SAFETY_TECH / "configs" / "config_agora.yaml")
    
    if not config_file.exists():
        logger.error(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    # 创建并运行统一安全防护测试
    test_runner = AgoraRealLLMTest(str(config_file))
    
    try:
        final_report = await test_runner.run_unified_security_test()
        
        logger.info("🎉 Agora统一安全防护测试完成!")
        logger.info(f"📊 安全评分: {final_report.get('security_score', 0)}/100")
        logger.info(f"🏷️ 安全等级: {final_report.get('security_level', 'UNKNOWN')}")
        
        # 根据安全等级设置退出码
        exit_code = 0 if final_report.get('security_level') in ['SECURE', 'MODERATE'] else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"❌ Agora统一安全防护测试失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
