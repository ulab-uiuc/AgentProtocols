# -*- coding: utf-8 -*-
"""
S1业务连续性测试配置工厂
提供预定义的测试配置和工厂方法
"""

from typing import Dict, Any, List
from .s1_business_continuity import (
    LoadMatrixConfig, NetworkDisturbanceConfig, AttackNoiseConfig,
    LoadPattern, MessageType, S1BusinessContinuityTester
)


class S1ConfigFactory:
    """S1配置工厂"""
    
    @staticmethod
    def create_light_test_config() -> Dict[str, Any]:
        """轻量测试配置：快速验证"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[1],  # 最小并发数
                rps_patterns=[LoadPattern.CONSTANT],
                message_types=[MessageType.SHORT],
                test_duration_seconds=5,  # 极短测试时间
                base_rps=1  # 最小RPS
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(10, 50),
                packet_loss_rate=0.01,
                reorder_probability=0.005,
                enable_connection_drops=False  # 轻量测试不启用连接中断
            ),
            'attack_config': AttackNoiseConfig(
                malicious_registration_rate=1,
                spam_message_rate=1,
                replay_attack_rate=1,
                dos_request_rate=1,
                probe_query_rate=1
            )
        }
    
    @staticmethod
    def create_standard_test_config() -> Dict[str, Any]:
        """标准测试配置：平衡性能和覆盖度"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[8, 32],
                rps_patterns=[LoadPattern.CONSTANT, LoadPattern.POISSON],
                message_types=[MessageType.SHORT, MessageType.LONG],
                test_duration_seconds=60,
                base_rps=10
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(10, 100),
                packet_loss_rate=0.02,
                reorder_probability=0.01,
                bandwidth_limit_kbps=1000,
                connection_drop_interval=45
            ),
            'attack_config': AttackNoiseConfig(
                malicious_registration_rate=5,
                spam_message_rate=20,
                replay_attack_rate=3,
                dos_request_rate=50,
                probe_query_rate=10
            )
        }
    
    @staticmethod
    def create_stress_test_config() -> Dict[str, Any]:
        """压力测试配置：高强度全面测试"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[8, 32, 128],
                rps_patterns=[LoadPattern.CONSTANT, LoadPattern.POISSON, LoadPattern.BURST],
                message_types=[MessageType.SHORT, MessageType.LONG, MessageType.STREAMING],
                test_duration_seconds=90,
                base_rps=15,
                burst_multiplier=4.0
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(20, 200),
                packet_loss_rate=0.05,
                reorder_probability=0.02,
                bandwidth_limit_kbps=500,
                connection_drop_interval=30
            ),
            'attack_config': AttackNoiseConfig(
                malicious_registration_rate=10,
                spam_message_rate=50,
                replay_attack_rate=8,
                dos_request_rate=100,
                probe_query_rate=20
            )
        }
    
    @staticmethod
    def create_protocol_optimized_config(protocol_name: str) -> Dict[str, Any]:
        """为特定协议优化的配置"""
        base_config = S1ConfigFactory.create_standard_test_config()
        
        if protocol_name.lower() == 'acp':
            # ACP (HTTP同步RPC型) - 对延迟尖峰敏感
            base_config['load_config'].base_rps = 8  # 降低RPS避免线程池瓶颈
            base_config['disturbance_config'].jitter_ms_range = (5, 50)  # 降低抖动
            base_config['disturbance_config'].connection_drop_interval = 60  # 减少连接中断
            
        elif protocol_name.lower() == 'anp':
            # ANP (长连接/会话型) - 建立成本高但稳态好
            base_config['load_config'].base_rps = 12  # 可以承受更高RPS
            base_config['disturbance_config'].connection_drop_interval = 90  # 更少连接中断
            base_config['disturbance_config'].packet_loss_rate = 0.03  # 测试对丢包的鲁棒性
            
        elif protocol_name.lower() == 'a2a':
            # A2A (混合型) - 取决于具体部署
            # 保持标准配置，但增加突发测试
            base_config['load_config'].rps_patterns.append(LoadPattern.BURST)
            base_config['load_config'].burst_multiplier = 3.5
            
        elif protocol_name.lower() == 'agora':
            # Agora (平台化网络) - 背压友好但可能厚尾延迟
            base_config['load_config'].base_rps = 15  # 更高RPS测试背压
            base_config['load_config'].test_duration_seconds = 75  # 更长测试时间观察厚尾
            base_config['disturbance_config'].bandwidth_limit_kbps = 800  # 更严格的带宽限制
        
        return base_config
    
    @staticmethod
    def create_network_focus_config() -> Dict[str, Any]:
        """网络扰动重点配置：专注测试网络韧性"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[16],  # 固定中等并发
                rps_patterns=[LoadPattern.CONSTANT],  # 固定模式
                message_types=[MessageType.SHORT, MessageType.LONG],  # 测试不同大小报文
                test_duration_seconds=120,  # 更长时间观察网络效果
                base_rps=8
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(50, 300),  # 高抖动
                packet_loss_rate=0.08,      # 高丢包率
                reorder_probability=0.05,   # 高乱序概率
                bandwidth_limit_kbps=200,   # 严格带宽限制
                connection_drop_interval=20, # 频繁连接中断
                enable_jitter=True,
                enable_packet_loss=True,
                enable_reorder=True,
                enable_bandwidth_limit=True,
                enable_connection_drops=True
            ),
            'attack_config': AttackNoiseConfig(
                # 降低攻击强度，专注网络扰动
                malicious_registration_rate=2,
                spam_message_rate=10,
                replay_attack_rate=1,
                dos_request_rate=20,
                probe_query_rate=5
            )
        }
    
    @staticmethod
    def create_attack_focus_config() -> Dict[str, Any]:
        """攻击噪声重点配置：专注测试攻击抗性"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[16],  # 固定中等并发
                rps_patterns=[LoadPattern.CONSTANT, LoadPattern.BURST],  # 测试突发对攻击的影响
                message_types=[MessageType.SHORT],  # 简化消息类型
                test_duration_seconds=90,
                base_rps=10
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                # 最小网络扰动，专注攻击效果
                jitter_ms_range=(5, 30),
                packet_loss_rate=0.005,
                reorder_probability=0.001,
                enable_connection_drops=False
            ),
            'attack_config': AttackNoiseConfig(
                malicious_registration_rate=20,  # 高频恶意注册
                spam_message_rate=100,           # 高频垃圾消息
                replay_attack_rate=15,           # 高频重放攻击
                dos_request_rate=200,            # 高频DoS
                probe_query_rate=30              # 高频旁路查询
            )
        }
    
    @staticmethod
    def create_latency_focus_config() -> Dict[str, Any]:
        """延迟重点配置：专注测试延迟分布"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[4, 16, 64],   # 多层并发测试
                rps_patterns=[LoadPattern.CONSTANT],  # 稳定模式
                message_types=[MessageType.SHORT, MessageType.LONG, MessageType.STREAMING],  # 全消息类型
                test_duration_seconds=180,       # 长时间测试
                base_rps=5                       # 较低RPS确保质量
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(1, 20),        # 低抖动精确测量
                packet_loss_rate=0.001,         # 极低丢包
                reorder_probability=0.0005,     # 极低乱序
                enable_connection_drops=False   # 不干扰延迟测量
            ),
            'attack_config': AttackNoiseConfig(
                # 最小攻击噪声
                malicious_registration_rate=1,
                spam_message_rate=5,
                replay_attack_rate=1,
                dos_request_rate=10,
                probe_query_rate=2
            )
        }
    
    @staticmethod
    def create_tester_from_config(protocol_name: str, config_dict: Dict[str, Any]) -> S1BusinessContinuityTester:
        """从配置字典创建测试器"""
        return S1BusinessContinuityTester(
            protocol_name=protocol_name,
            load_config=config_dict['load_config'],
            disturbance_config=config_dict['disturbance_config'],
            attack_config=config_dict['attack_config']
        )
    
    @staticmethod
    def get_available_configs() -> List[str]:
        """获取可用配置列表"""
        return [
            'light',
            'standard', 
            'stress',
            'network_focus',
            'attack_focus',
            'latency_focus'
        ]
    
    @staticmethod
    def create_config_by_name(config_name: str, protocol_name: str = None) -> Dict[str, Any]:
        """根据名称创建配置"""
        config_map = {
            'light': S1ConfigFactory.create_light_test_config,
            'standard': S1ConfigFactory.create_standard_test_config,
            'stress': S1ConfigFactory.create_stress_test_config,
            'network_focus': S1ConfigFactory.create_network_focus_config,
            'attack_focus': S1ConfigFactory.create_attack_focus_config,
            'latency_focus': S1ConfigFactory.create_latency_focus_config
        }
        
        if config_name == 'protocol_optimized' and protocol_name:
            return S1ConfigFactory.create_protocol_optimized_config(protocol_name)
        
        if config_name not in config_map:
            raise ValueError(f"未知配置名称: {config_name}. 可用配置: {list(config_map.keys())}")
        
        return config_map[config_name]()


def create_s1_tester(protocol_name: str, 
                     config_name: str = 'standard') -> S1BusinessContinuityTester:
    """便捷函数：创建S1测试器"""
    if config_name == 'protocol_optimized':
        config = S1ConfigFactory.create_protocol_optimized_config(protocol_name)
    else:
        config = S1ConfigFactory.create_config_by_name(config_name)
    
    return S1ConfigFactory.create_tester_from_config(protocol_name, config)


# 预定义的医疗场景消息模板
MEDICAL_SCENARIO_TEMPLATES = {
    MessageType.SHORT: [
        "患者{patient_id}血压异常，请会诊",
        "手术{surgery_id}需要专科医生支援",
        "检查报告{report_id}结果异常",
        "患者{patient_id}出现过敏反应",
        "急诊{emergency_id}需要立即处理",
        "药物{drug_name}剂量需要调整",
        "病房{room_id}患者病情变化",
        "手术室{or_id}设备故障报告"
    ],
    
    MessageType.LONG: [
        """患者{patient_id}详细病历：
年龄：{age}岁，性别：{gender}
主诉：{chief_complaint}
现病史：{present_illness}
既往史：{past_history}
体格检查：{physical_exam}
辅助检查：{lab_results}
初步诊断：{diagnosis}
治疗方案：{treatment_plan}
请各位专家给出会诊意见，特别关注{focus_area}的处理。
患者目前生命体征稳定，但需要密切观察{monitoring_focus}。
建议进一步检查{additional_tests}以明确诊断。
如有疑问请及时联系，谢谢！""",
        
        """多学科会诊病例{case_id}：
科室：{department}
主治医师：{attending_doctor}
患者基本信息：{patient_info}
病情摘要：{case_summary}
目前治疗：{current_treatment}
会诊问题：{consultation_questions}
相关检查：{relevant_tests}
影像学表现：{imaging_findings}
实验室指标：{lab_values}
请各位专家从以下角度给出意见：
1. 诊断是否准确？
2. 治疗方案是否合理？
3. 是否需要调整用药？
4. 预后评估如何？
5. 是否需要进一步检查？
期待您的宝贵意见，谢谢合作！"""
    ],
    
    MessageType.STREAMING: [
        "[数据流{stream_id}-1/5] 患者{patient_id}生命体征监测开始... | [数据流{stream_id}-2/5] 血压：{bp_systolic}/{bp_diastolic}mmHg，心率：{heart_rate}次/分 | [数据流{stream_id}-3/5] 血氧饱和度：{spo2}%，呼吸：{resp_rate}次/分，体温：{temp}°C | [数据流{stream_id}-4/5] 心电图：{ecg_findings}，血气分析：{blood_gas} | [数据流{stream_id}-5/5] 监测结论：{monitoring_conclusion}，建议：{recommendations}",
        
        "[实时数据{data_id}-1/4] 手术{surgery_id}进行中，当前阶段：{surgery_stage} | [实时数据{data_id}-2/4] 麻醉状态：{anesthesia_status}，出血量：{blood_loss}ml | [实时数据{data_id}-3/4] 手术进展：{surgery_progress}，预计剩余时间：{estimated_time} | [实时数据{data_id}-4/4] 需要支援：{support_needed}，联系电话：{contact_number}"
    ]
}


def generate_medical_message(msg_type: MessageType, case_data: Dict[str, Any] = None) -> str:
    """生成医疗场景消息"""
    import random
    
    if case_data is None:
        case_data = {}
    
    templates = MEDICAL_SCENARIO_TEMPLATES.get(msg_type, ["默认消息"])
    template = random.choice(templates)
    
    # 默认数据
    default_data = {
        'patient_id': f"P{random.randint(1000, 9999)}",
        'surgery_id': f"S{random.randint(100, 999)}",
        'report_id': f"R{random.randint(10000, 99999)}",
        'emergency_id': f"E{random.randint(100, 999)}",
        'drug_name': random.choice(['阿司匹林', '美托洛尔', '氯吡格雷', '阿托伐他汀']),
        'room_id': f"{random.randint(1, 20)}0{random.randint(1, 8)}",
        'or_id': f"OR-{random.randint(1, 10)}",
        'age': random.randint(25, 80),
        'gender': random.choice(['男', '女']),
        'chief_complaint': random.choice(['胸痛', '呼吸困难', '腹痛', '头痛', '发热']),
        'case_id': f"MDT{random.randint(1000, 9999)}",
        'department': random.choice(['心内科', '呼吸科', '消化科', '神经科', '急诊科']),
        'attending_doctor': f"Dr.{random.choice(['张', '李', '王', '刘', '陈'])}{random.choice(['明', '华', '强', '伟', '峰'])}",
        'stream_id': f"STR{random.randint(100, 999)}",
        'bp_systolic': random.randint(90, 180),
        'bp_diastolic': random.randint(60, 110),
        'heart_rate': random.randint(60, 120),
        'spo2': random.randint(85, 100),
        'resp_rate': random.randint(12, 30),
        'temp': round(random.uniform(36.0, 39.5), 1),
        'data_id': f"RT{random.randint(100, 999)}"
    }
    
    # 合并用户数据
    default_data.update(case_data)
    
    try:
        return template.format(**default_data)
    except KeyError as e:
        # 如果模板中有未提供的字段，返回简化版本
        return f"医疗消息 - 患者{default_data['patient_id']} - {msg_type.value}类型消息"
