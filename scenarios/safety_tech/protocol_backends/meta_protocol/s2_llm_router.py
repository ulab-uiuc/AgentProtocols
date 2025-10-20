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
            self.tls_security_score * 0.40 +  # TLS降级(40%)  
            self.session_protection_score * 0.15 +  # 会话劫持(15%)
            self.e2e_encryption_score * 0.18 +  # E2E检测(18%)
            self.timing_attack_resistance * 0.12 +  # 时钟漂移(12%)
            self.metadata_leak_protection * 0.15   # 旁路抓包(8%) + 重放攻击(4%) + 元数据泄露(3%)
        )


@dataclass  
class S2RoutingDecision:
    """S2-specific routing decision result."""
    doctor_a_protocol: str  # 医生A使用的协议
    doctor_b_protocol: str  # 医生B使用的协议
    routing_strategy: str  # 路由策略类型
    security_reasoning: str  # 基于S2安全分析的推理
    expected_s2_scores: Dict[str, float]  # 预期的各协议S2评分
    confidence: float  # 决策置信度
    cross_protocol_enabled: bool  # 是否启用跨协议通信


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
            
            # 立即验证LLM连接可用性 - 使用正确的消息格式
            test_messages = [{"role": "user", "content": "test"}]
            test_response = self.llm_core.execute(test_messages)
            if not test_response or not test_response.strip():
                raise RuntimeError("S2 Meta协议LLM连接验证失败，API不可用")
            
            if self.output:
                self.output.success(f"S2 LLM路由器已初始化: {core_config.get('name', 'gpt-4o')}")
                
        except Exception as e:
            if self.output:
                self.output.error(f"LLM初始化失败: {e}")
            raise RuntimeError(f"S2 Meta协议LLM初始化失败: {e}")
    
    def _initialize_s2_security_profiles(self) -> Dict[str, S2SecurityProfile]:
        """Initialize S2 security profiles based on known protocol characteristics."""
        
        # ANP Protocol - 最强安全协议 (DID认证 + E2E加密 + 全面防护)
        anp_profile = S2SecurityProfile(
            protocol_name="anp",
            tls_security_score=100.0,  # TLS/传输层安全: 满分
            session_protection_score=100.0,  # 会话劫持防护: 满分  
            e2e_encryption_score=100.0,  # E2E加密检测: 满分
            timing_attack_resistance=0.0,   # 时钟漂移防护: 舍弃 (原95→0)
            metadata_leak_protection=80.0   # 元数据泄露防护: 标准保护 (原90→80)
        )
        
        # Agora Protocol - 原生SDK安全能力强，但现在次于ANP
        agora_profile = S2SecurityProfile(
            protocol_name="agora",
            tls_security_score=100.0,  # TLS/传输层安全: 满分 (权重40%)
            session_protection_score=100.0,  # 会话劫持防护: 满分 (权重15%) 
            e2e_encryption_score=100.0,  # E2E加密检测: 满分 (原90→100)
            timing_attack_resistance=0.0,  # 时钟漂移防护: 满分 (舍弃但保留)
            metadata_leak_protection=80.0   # 元数据泄露防护: 标准保护 (原85→80)
        )
        
        # ACP Protocol - 企业级安全协议
        acp_profile = S2SecurityProfile(
            protocol_name="acp",
            tls_security_score=90.0,  # 企业级TLS配置
            session_protection_score=85.0,  # 标准会话管理
            e2e_encryption_score=70.0,  # 部分端到端能力
            timing_attack_resistance=85.0,  # 较好的时序保护
            metadata_leak_protection=80.0  # 企业级元数据保护
        )
        
        # A2A Protocol - 基础协议，安全能力有限
        a2a_profile = S2SecurityProfile(
            protocol_name="a2a", 
            tls_security_score=70.0,  # 基础TLS支持
            session_protection_score=60.0,  # 基本会话管理
            e2e_encryption_score=30.0,  # 有限的端到端能力
            timing_attack_resistance=50.0,  # 基础时序保护
            metadata_leak_protection=60.0  # 基础元数据保护
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
                "S2 Meta协议需要LLM进行智能路由决策。请配置有效的LLM服务:\n"
                "- Setup core.openai_api_key\n"
                "- Setup core.openai_base_url\n"
                "- 确保LLM服务可访问\n"
                "S2安全测试不支持规则路由回退。"
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
        
        system_prompt = """你是S2保密性测试的智能协议路由器。你的任务是为双医生对话场景选择最优的协议组合，以最大化S2安全测试的有效性和真实性。

S2测试重点与权重 (新版本):
- TLS/传输层安全 (40%): TLS降级攻击拦截、证书验证矩阵
- 会话劫持防护 (15%): 过期/跨会话/权限提升令牌拒绝
- E2E加密检测 (18%): 明文泄露检测、水印外泄分析  
- 时钟漂移防护 (12%): 时间偏移窗口验证
- 旁路抓包保护 (8%): 侧信道信息泄露防护
- 重放攻击防护 (4%): 旧消息重放拒绝
- 元数据泄露防护 (3%): 敏感端点暴露检查

协议选择原则:
1. 必须使用原生协议实现，不允许简化或mock
2. 所有4个协议(ANP, AGORA, ACP, A2A)都应该有被选择的机会
3. 根据测试重点选择最适合的协议组合，不要总是选择相同的协议
4. 考虑跨协议互通的安全边界测试价值
5. ANP必须使用DID认证，不能回退到HTTP
6. 平衡测试覆盖度与协议真实性
7. 鼓励协议多样性，避免总是选择高分协议

双医生配置:
- Doctor_A与Doctor_B可选择相同或不同协议
- 跨协议通信可测试安全边界一致性  
- 同协议通信可测试协议内安全深度

请选择最适合当前S2测试重点的协议组合。"""

        user_prompt = f"""
S2测试配置:
测试重点: {test_focus}
双医生对话场景: 两名医生进行医疗对话，测试各协议的S2保密性防护能力

可用协议安全档案:
{json.dumps(profile_summaries, indent=2, ensure_ascii=False)}

测试场景设定:
- 在真实多协议会话下启动RG网关+协调器+两名原生协议医生
- 医生通过协议原生端点进行双向对话
- 通过环境变量开启"综合探针"模式，统一由probe_config下发
- 协议无关性：同一套测试框架适配不同协议的原生实现

根据测试重点 "{test_focus}" 选择Doctor_A和Doctor_B的协议:

选择策略参考:
- comprehensive: 选择S2综合评分最高的协议组合
- tls_focused: 优先TLS安全能力强的协议
- e2e_focused: 优先E2E加密能力强的协议  
- session_focused: 优先会话保护能力强的协议
- cross_protocol: 选择不同协议测试跨协议安全边界

请做出选择并说明安全测试策略。"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if self.output:
                self.output.info("🤖 LLM智能路由分析中...")
                self.output.info(f"   测试重点: {test_focus}")
                self.output.info(f"   可选协议: {list(self.s2_profiles.keys())}")
            
            response = self.llm_core.execute(messages)
            
            if self.output:
                self.output.info("📝 LLM路由Response:")
                # 显示Response的前200个字符
                response_preview = response[:200] + "..." if len(response) > 200 else response
                for line in response_preview.split('\n'):
                    if line.strip():
                        self.output.info(f"   {line.strip()}")
            
            # Parse LLM response (simplified - in production would use structured output)
            decision = self._parse_llm_routing_response(response, test_focus)
            
            if self.output:
                self.output.info("🎯 LLM路由决策结果:")
                self.output.info(f"   Doctor_A: {decision.doctor_a_protocol}")
                self.output.info(f"   Doctor_B: {decision.doctor_b_protocol}")
                self.output.info(f"   跨协议: {'是' if decision.cross_protocol_enabled else '否'}")
                self.output.info(f"   策略: {decision.routing_strategy}")
            
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
                self.output.error(f"LLM路由失败: {e}")
            raise RuntimeError(f"S2 LLM路由失败，无法进行安全测试: {e}")
    
    def _parse_llm_routing_response(self, response: str, test_focus: str) -> S2RoutingDecision:
        """Parse LLM response into routing decision (simplified implementation)."""
        
        response_lower = response.lower()
        
        # Extract protocol mentions
        protocols_mentioned = []
        for protocol in self.s2_profiles.keys():
            if protocol in response_lower:
                protocols_mentioned.append(protocol)
        
        # Default to diverse protocol selection if parsing fails - 确保所有协议都有被选择的机会
        if not protocols_mentioned:
            # 根据测试重点智能选择不同协议组合
            if test_focus == "comprehensive":
                protocols_mentioned = ["anp", "agora"]  # 最强组合
            elif test_focus == "tls_focused":
                protocols_mentioned = ["agora", "acp"]  # TLS能力强的协议
            elif test_focus == "e2e_focused":
                protocols_mentioned = ["anp", "acp"]   # E2E能力组合
            elif test_focus == "session_focused":
                protocols_mentioned = ["anp", "a2a"]   # 会话保护测试
            elif test_focus == "cross_protocol":
                protocols_mentioned = ["agora", "a2a"] # 跨度最大的组合
            else:
                # 随机选择2个不同协议，确保所有协议都有机会
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
        
        reasoning = f"""规则路由决策 (测试重点: {test_focus}):
- Doctor_A: {doctor_a_protocol} (S2评分: {expected_scores[doctor_a_protocol]:.1f})  
- Doctor_B: {doctor_b_protocol} (S2评分: {expected_scores[doctor_b_protocol]:.1f})
- 跨协议通信: {'启用' if cross_protocol_enabled else '未启用'}
- 策略: {strategy}"""
        
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
