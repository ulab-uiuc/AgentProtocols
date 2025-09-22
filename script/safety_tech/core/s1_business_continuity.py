# -*- coding: utf-8 -*-
"""
S1业务连续性测试核心模块
测试目标：在强干扰下保持"请求→路由→对端执行→回执"的闭环稳定与时延可控
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
    """负载模式"""
    CONSTANT = "constant"  # 恒定RPS
    POISSON = "poisson"    # 泊松分布
    BURST = "burst"        # 突发模式


class MessageType(Enum):
    """报文类型"""
    SHORT = "short"        # 短文本 (<100字符)
    LONG = "long"          # 长文本 (>1000字符)
    STREAMING = "streaming" # 流式数据


@dataclass
class LoadMatrixConfig:
    """负载矩阵配置"""
    concurrent_levels: List[int] = field(default_factory=lambda: [8, 32, 128])  # 并发数
    rps_patterns: List[LoadPattern] = field(default_factory=lambda: [LoadPattern.CONSTANT, LoadPattern.POISSON, LoadPattern.BURST])
    message_types: List[MessageType] = field(default_factory=lambda: [MessageType.SHORT, MessageType.LONG, MessageType.STREAMING])
    test_duration_seconds: int = 60  # 每个组合的测试时长
    base_rps: int = 10  # 基础RPS
    burst_multiplier: float = 3.0  # 突发倍数


@dataclass
class NetworkDisturbanceConfig:
    """网络扰动配置"""
    jitter_ms_range: Tuple[int, int] = (10, 100)  # 抖动范围
    packet_loss_rate: float = 0.02  # 丢包率 2%
    reorder_probability: float = 0.01  # 乱序概率 1%
    bandwidth_limit_kbps: int = 1000  # 带宽限制 1Mbps
    connection_drop_interval: int = 30  # 短线重连间隔(秒)
    enable_jitter: bool = True
    enable_packet_loss: bool = True
    enable_reorder: bool = True
    enable_bandwidth_limit: bool = True
    enable_connection_drops: bool = True


@dataclass
class AttackNoiseConfig:
    """攻击噪声配置"""
    malicious_registration_rate: int = 5  # 恶意注册频率(次/分钟)
    spam_message_rate: int = 20  # 垃圾消息频率(次/分钟)
    replay_attack_rate: int = 3  # 重放攻击频率(次/分钟)
    dos_request_rate: int = 50  # DoS请求频率(次/分钟)
    probe_query_rate: int = 10  # 旁路查询频率(次/分钟)
    enable_all: bool = True


@dataclass
class CorrelationTracker:
    """关联ID跟踪器"""
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
    """S1测试结果"""
    test_config: Dict[str, Any]
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    retry_count: int = 0
    reconnection_count: int = 0
    
    # 延迟统计
    latencies_ms: List[float] = field(default_factory=list)
    
    # 错误分布
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # 网络统计
    packets_sent: int = 0
    packets_lost: int = 0
    packets_reordered: int = 0
    jitter_events: int = 0
    
    @property
    def completion_rate(self) -> float:
        """完成率"""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def timeout_rate(self) -> float:
        """超时率"""
        return self.timeout_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """平均延迟"""
        return np.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def p50_latency_ms(self) -> float:
        """P50延迟"""
        return np.percentile(self.latencies_ms, 50) if self.latencies_ms else 0.0
    
    @property
    def p95_latency_ms(self) -> float:
        """P95延迟"""
        return np.percentile(self.latencies_ms, 95) if self.latencies_ms else 0.0
    
    @property
    def p99_latency_ms(self) -> float:
        """P99延迟"""
        return np.percentile(self.latencies_ms, 99) if self.latencies_ms else 0.0
    
    @property
    def packet_loss_rate(self) -> float:
        """实际丢包率"""
        return self.packets_lost / self.packets_sent if self.packets_sent > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
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
    """S1业务连续性测试器"""
    
    def __init__(self, 
                 protocol_name: str,
                 load_config: LoadMatrixConfig,
                 disturbance_config: NetworkDisturbanceConfig,
                 attack_config: AttackNoiseConfig):
        self.protocol_name = protocol_name
        self.load_config = load_config
        self.disturbance_config = disturbance_config
        self.attack_config = attack_config
        
        # 跟踪器
        self.active_trackers: Dict[str, CorrelationTracker] = {}
        self.completed_trackers: List[CorrelationTracker] = []
        
        # 攻击任务
        self.attack_tasks: List[asyncio.Task] = []
        
        # 结果收集
        self.test_results: List[S1TestResult] = []
        
        # 网络扰动状态
        self.network_disturbance_active = False
        self.connection_drop_task: Optional[asyncio.Task] = None
        
    async def start_network_disturbance(self):
        """启动网络扰动"""
        if not self.disturbance_config.enable_jitter and \
           not self.disturbance_config.enable_packet_loss and \
           not self.disturbance_config.enable_reorder and \
           not self.disturbance_config.enable_connection_drops:
            logger.info("所有网络扰动都被禁用，跳过")
            return
            
        self.network_disturbance_active = True
        logger.info(f"🌊 启动网络扰动: 抖动={self.disturbance_config.enable_jitter}, "
                   f"丢包={self.disturbance_config.enable_packet_loss}, "
                   f"乱序={self.disturbance_config.enable_reorder}, "
                   f"断线={self.disturbance_config.enable_connection_drops}")
        
        # 启动连接中断任务
        if self.disturbance_config.enable_connection_drops:
            self.connection_drop_task = asyncio.create_task(self._connection_drop_loop())
    
    async def stop_network_disturbance(self):
        """停止网络扰动"""
        self.network_disturbance_active = False
        if self.connection_drop_task:
            self.connection_drop_task.cancel()
            try:
                await self.connection_drop_task
            except asyncio.CancelledError:
                pass
        logger.info("🌊 网络扰动已停止")
    
    async def _connection_drop_loop(self):
        """连接中断循环"""
        try:
            while self.network_disturbance_active:
                await asyncio.sleep(self.disturbance_config.connection_drop_interval)
                if self.network_disturbance_active:
                    logger.debug("🔌 模拟连接中断")
                    # 这里可以实现具体的连接中断逻辑
                    # 例如关闭现有连接，强制重连
        except asyncio.CancelledError:
            logger.debug("连接中断循环被取消")
    
    async def apply_network_disturbance(self, delay_before_send: bool = True) -> Dict[str, Any]:
        """应用网络扰动效果"""
        disturbance_effects = {}
        
        if not self.network_disturbance_active:
            return disturbance_effects
        
        # 抖动延迟
        if self.disturbance_config.enable_jitter:
            jitter_ms = random.randint(*self.disturbance_config.jitter_ms_range)
            disturbance_effects['jitter_ms'] = jitter_ms
            if delay_before_send:
                await asyncio.sleep(jitter_ms / 1000.0)
        
        # 丢包模拟
        if self.disturbance_config.enable_packet_loss:
            if random.random() < self.disturbance_config.packet_loss_rate:
                disturbance_effects['packet_dropped'] = True
                raise NetworkDisturbanceException("Simulated packet loss")
        
        # 乱序模拟（通过额外延迟）
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
        """启动攻击噪声"""
        if not self.attack_config.enable_all:
            logger.info("攻击噪声被禁用，跳过")
            return
            
        logger.info("⚔️ 启动攻击噪声任务")
        
        # 恶意注册攻击
        self.attack_tasks.append(
            asyncio.create_task(self._malicious_registration_loop(rg_port))
        )
        
        # 垃圾消息攻击
        self.attack_tasks.append(
            asyncio.create_task(self._spam_message_loop(coord_port))
        )
        
        # 重放攻击
        self.attack_tasks.append(
            asyncio.create_task(self._replay_attack_loop(coord_port))
        )
        
        # DoS攻击
        self.attack_tasks.append(
            asyncio.create_task(self._dos_attack_loop(coord_port))
        )
        
        # 旁路查询攻击
        self.attack_tasks.append(
            asyncio.create_task(self._probe_query_loop(obs_port))
        )
    
    async def stop_attack_noise(self):
        """停止攻击噪声"""
        logger.info("⚔️ 停止攻击噪声任务")
        for task in self.attack_tasks:
            task.cancel()
        
        # 等待所有任务完成
        if self.attack_tasks:
            await asyncio.gather(*self.attack_tasks, return_exceptions=True)
        
        self.attack_tasks.clear()
    
    async def _malicious_registration_loop(self, rg_port: int):
        """恶意注册攻击循环"""
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
                        logger.debug(f"发送恶意注册: {fake_agent_id}")
                    except Exception as e:
                        logger.debug(f"恶意注册失败: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("恶意注册攻击循环被取消")
    
    async def _spam_message_loop(self, coord_port: int):
        """垃圾消息攻击循环"""
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
                        logger.debug("发送垃圾消息")
                    except Exception as e:
                        logger.debug(f"垃圾消息发送失败: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("垃圾消息攻击循环被取消")
    
    async def _replay_attack_loop(self, coord_port: int):
        """重放攻击循环"""
        try:
            interval = 60.0 / self.attack_config.replay_attack_rate
            async with httpx.AsyncClient() as client:
                # 收集一些真实消息进行重放
                replay_messages = []
                
                while True:
                    try:
                        if not replay_messages:
                            # 生成一些模拟的"历史"消息用于重放
                            for i in range(5):
                                replay_messages.append({
                                    "sender_id": f"doctor_a_{i}",
                                    "receiver_id": f"doctor_b_{i}",
                                    "text": f"Historical message {i} for replay attack",
                                    "message_id": f"old_msg_{int(time.time()-3600)}_{i}",  # 1小时前
                                    "correlation_id": f"old_corr_{int(time.time()-3600)}_{i}"
                                })
                        
                        # 随机选择一条消息进行重放
                        replay_msg = random.choice(replay_messages)
                        await client.post(f"http://127.0.0.1:{coord_port}/route_message", 
                                        json=replay_msg, timeout=3.0)
                        logger.debug(f"重放消息: {replay_msg['message_id']}")
                    except Exception as e:
                        logger.debug(f"重放攻击失败: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("重放攻击循环被取消")
    
    async def _dos_attack_loop(self, coord_port: int):
        """DoS攻击循环"""
        try:
            interval = 60.0 / self.attack_config.dos_request_rate
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        # 发送大量无意义请求
                        tasks = []
                        for _ in range(5):  # 每次发送5个并发请求
                            task = client.get(f"http://127.0.0.1:{coord_port}/health", timeout=1.0)
                            tasks.append(task)
                        
                        await asyncio.gather(*tasks, return_exceptions=True)
                        logger.debug("发送DoS请求批次")
                    except Exception as e:
                        logger.debug(f"DoS攻击失败: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("DoS攻击循环被取消")
    
    async def _probe_query_loop(self, obs_port: int):
        """旁路查询攻击循环"""
        try:
            interval = 60.0 / self.attack_config.probe_query_rate
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        # 尝试各种旁路查询
                        endpoints = ["/health", "/status", "/metrics", "/agents", "/conversations"]
                        endpoint = random.choice(endpoints)
                        await client.get(f"http://127.0.0.1:{obs_port}{endpoint}", timeout=3.0)
                        logger.debug(f"旁路查询: {endpoint}")
                    except Exception as e:
                        logger.debug(f"旁路查询失败: {e}")
                    
                    await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("旁路查询攻击循环被取消")
    
    def generate_message_content(self, msg_type: MessageType) -> str:
        """生成指定类型的消息内容"""
        if msg_type == MessageType.SHORT:
            templates = [
                "患者血压异常，需要会诊",
                "手术方案需要讨论",
                "药物过敏反应咨询",
                "检查结果异常，请查看",
                "紧急情况需要支援"
            ]
            return random.choice(templates)
        
        elif msg_type == MessageType.LONG:
            base_content = """
患者基本信息：
- 姓名：张某某
- 年龄：45岁
- 性别：男
- 主诉：胸痛3小时，伴有呼吸困难
- 既往史：高血压病史5年，糖尿病病史3年
- 家族史：父亲有冠心病史
- 体格检查：血压150/95mmHg，心率96次/分，呼吸22次/分
- 辅助检查：心电图显示ST段压低，肌钙蛋白升高
- 初步诊断：急性冠脉综合征可能
- 建议：需要紧急心内科会诊，考虑行冠脉造影检查
- 用药情况：已给予阿司匹林、氯吡格雷、阿托伐他汀
- 监护：已转入CCU监护，持续心电监护
- 家属沟通：已告知病情严重性，家属同意进一步治疗
请各位专家给出治疗建议，特别是介入治疗的时机选择。
            """.strip()
            # 重复内容以达到长文本要求
            return base_content + "\n" + base_content[:500]
        
        elif msg_type == MessageType.STREAMING:
            # 流式数据模拟：分段发送的数据
            segments = [
                "[数据流-1/5] 患者生命体征监测中...",
                "[数据流-2/5] 血压: 150/95, 心率: 96",
                "[数据流-3/5] 血氧: 98%, 体温: 36.8°C",
                "[数据流-4/5] ECG数据: 异常ST段变化",
                "[数据流-5/5] 建议立即会诊"
            ]
            return " | ".join(segments)
    
    def create_correlation_tracker(self, 
                                 sender_id: str, 
                                 receiver_id: str, 
                                 msg_type: MessageType) -> CorrelationTracker:
        """创建关联跟踪器"""
        content = self.generate_message_content(msg_type)
        tracker = CorrelationTracker(
            request_id="",  # 将自动生成
            correlation_id="",  # 将自动生成
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
        """发送被跟踪的消息"""
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
            # 应用网络扰动
            result['network_effects'] = await self.apply_network_disturbance(delay_before_send=True)
            
            # 准备发送载荷
            payload = {
                'sender_id': tracker.sender_id,
                'receiver_id': tracker.receiver_id,
                'text': tracker.message_content,
                'message_id': tracker.request_id,
                'correlation_id': tracker.correlation_id,
                **send_kwargs
            }
            
            # 发送消息
            response = await send_func(payload)
            
            end_time = time.time()
            result['latency_ms'] = (end_time - start_time) * 1000
            
            # 检查发送是否成功
            if isinstance(response, dict):
                if response.get('status') in ['success', 'ok', 'processed']:
                    result['success'] = True
                elif 'error' in response:
                    result['error'] = response['error']
            elif hasattr(response, 'status_code'):
                if response.status_code in [200, 202]:
                    result['success'] = True
                else:
                    result['error'] = f"HTTP {response.status_code}"
            
        except NetworkDisturbanceException as e:
            # 网络扰动导致的"丢包"
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
        """检查是否收到了预期的回执"""
        if correlation_id not in self.active_trackers:
            return False
        
        tracker = self.active_trackers[correlation_id]
        
        # 简单的模式匹配检查
        if tracker.expected_response_pattern:
            import re
            if re.search(tracker.expected_response_pattern, response_content, re.IGNORECASE):
                # 移动到已完成列表
                self.completed_trackers.append(tracker)
                del self.active_trackers[correlation_id]
                return True
        
        return False
    
    def cleanup_expired_trackers(self) -> int:
        """清理超时的跟踪器"""
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
        """运行单个负载测试组合"""
        
        test_config = {
            'concurrent_level': concurrent_level,
            'rps_pattern': rps_pattern.value,
            'message_type': message_type.value,
            'duration_seconds': self.load_config.test_duration_seconds,
            'base_rps': self.load_config.base_rps
        }
        
        result = S1TestResult(test_config=test_config)
        
        logger.info(f"🧪 开始负载测试: 并发={concurrent_level}, RPS模式={rps_pattern.value}, "
                   f"消息类型={message_type.value}")
        
        # 计算请求间隔
        base_interval = 1.0 / self.load_config.base_rps
        
        # 创建并发任务
        tasks = []
        for i in range(concurrent_level):
            task = asyncio.create_task(
                self._concurrent_request_loop(
                    i, base_interval, rps_pattern, message_type, 
                    send_func, sender_id, receiver_id, result
                )
            )
            tasks.append(task)
        
        # 运行指定时长
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.load_config.test_duration_seconds
            )
        except asyncio.TimeoutError:
            logger.info("测试时间到，停止任务")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 清理超时跟踪器
        expired_count = self.cleanup_expired_trackers()
        result.timeout_requests = expired_count
        
        # 等待一小段时间以收集可能的延迟回执
        await asyncio.sleep(2)
        final_expired = self.cleanup_expired_trackers()
        result.timeout_requests += final_expired
        
        logger.info(f"✅ 负载测试完成: 成功率={result.completion_rate:.1%}, "
                   f"平均延迟={result.avg_latency_ms:.1f}ms, P95={result.p95_latency_ms:.1f}ms")
        
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
        """并发请求循环"""
        try:
            request_count = 0
            start_time = time.time()
            
            while time.time() - start_time < self.load_config.test_duration_seconds:
                # 计算当前间隔
                if rps_pattern == LoadPattern.CONSTANT:
                    interval = base_interval
                elif rps_pattern == LoadPattern.POISSON:
                    # 泊松分布：使用指数分布间隔
                    interval = np.random.exponential(base_interval)
                elif rps_pattern == LoadPattern.BURST:
                    # 突发模式：周期性高频发送
                    cycle_position = (time.time() - start_time) % 10  # 10秒周期
                    if cycle_position < 2:  # 前2秒突发
                        interval = base_interval / self.load_config.burst_multiplier
                    else:  # 后8秒正常
                        interval = base_interval * 1.5
                
                # 创建跟踪器
                tracker = self.create_correlation_tracker(
                    f"{sender_id}_worker_{worker_id}",
                    receiver_id,
                    message_type
                )
                
                # 发送消息
                send_result = await self.send_tracked_message(tracker, send_func)
                
                # 更新统计
                result.total_requests += 1
                result.packets_sent += 1
                
                if send_result['success']:
                    result.successful_requests += 1
                    result.latencies_ms.append(send_result['latency_ms'])
                else:
                    result.failed_requests += 1
                    error_type = send_result.get('error', 'unknown')
                    result.error_types[error_type] = result.error_types.get(error_type, 0) + 1
                
                # 网络效果统计
                network_effects = send_result.get('network_effects', {})
                if network_effects.get('packet_dropped'):
                    result.packets_lost += 1
                if network_effects.get('jitter_ms'):
                    result.jitter_events += 1
                if network_effects.get('reorder_delay_ms'):
                    result.packets_reordered += 1
                
                request_count += 1
                
                # 等待下次发送
                await asyncio.sleep(max(0.01, interval))  # 最小10ms间隔
                
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} 被取消")
        except Exception as e:
            logger.error(f"Worker {worker_id} 异常: {e}")
    
    async def run_full_test_matrix(self,
                                 send_func: Callable,
                                 sender_id: str,
                                 receiver_id: str,
                                 rg_port: int,
                                 coord_port: int,
                                 obs_port: int) -> List[S1TestResult]:
        """运行完整的测试矩阵"""
        
        logger.info(f"🚀 开始S1业务连续性全矩阵测试 - 协议: {self.protocol_name}")
        logger.info(f"📊 测试矩阵: {len(self.load_config.concurrent_levels)} × "
                   f"{len(self.load_config.rps_patterns)} × "
                   f"{len(self.load_config.message_types)} = "
                   f"{len(self.load_config.concurrent_levels) * len(self.load_config.rps_patterns) * len(self.load_config.message_types)} 种组合")
        
        # 启动网络扰动
        await self.start_network_disturbance()
        
        # 启动攻击噪声
        await self.start_attack_noise(rg_port, coord_port, obs_port)
        
        try:
            all_results = []
            total_combinations = (len(self.load_config.concurrent_levels) * 
                                len(self.load_config.rps_patterns) * 
                                len(self.load_config.message_types))
            current_combination = 0
            
            for concurrent_level in self.load_config.concurrent_levels:
                for rps_pattern in self.load_config.rps_patterns:
                    for message_type in self.load_config.message_types:
                        current_combination += 1
                        
                        logger.info(f"📋 测试组合 {current_combination}/{total_combinations}")
                        
                        # 运行单个组合测试
                        result = await self.run_load_test_combination(
                            concurrent_level, rps_pattern, message_type,
                            send_func, sender_id, receiver_id
                        )
                        
                        all_results.append(result)
                        
                        # 组合间短暂休息
                        if current_combination < total_combinations:
                            logger.info("⏸️ 组合间休息 5 秒...")
                            await asyncio.sleep(5)
            
            self.test_results = all_results
            return all_results
            
        finally:
            # 停止攻击噪声和网络扰动
            await self.stop_attack_noise()
            await self.stop_network_disturbance()
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合测试报告"""
        if not self.test_results:
            return {"error": "没有测试结果"}
        
        # 汇总统计
        total_requests = sum(r.total_requests for r in self.test_results)
        total_successful = sum(r.successful_requests for r in self.test_results)
        total_failed = sum(r.failed_requests for r in self.test_results)
        total_timeout = sum(r.timeout_requests for r in self.test_results)
        
        all_latencies = []
        for r in self.test_results:
            all_latencies.extend(r.latencies_ms)
        
        # 按维度分组分析
        by_concurrent = defaultdict(list)
        by_rps_pattern = defaultdict(list)
        by_message_type = defaultdict(list)
        
        for result in self.test_results:
            config = result.test_config
            by_concurrent[config['concurrent_level']].append(result)
            by_rps_pattern[config['rps_pattern']].append(result)
            by_message_type[config['message_type']].append(result)
        
        # 计算各维度平均性能
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
        
        # 找出最佳和最差组合
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
    """网络扰动异常"""
    pass
