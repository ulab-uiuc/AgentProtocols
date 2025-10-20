# -*- coding: utf-8 -*-
"""
S1 business continuity test configuration factory.
Provides predefined test configurations and factory methods.
"""

from typing import Dict, Any, List
from .s1_business_continuity import (
    LoadMatrixConfig, NetworkDisturbanceConfig, AttackNoiseConfig,
    LoadPattern, MessageType, S1BusinessContinuityTester
)


class S1ConfigFactory:
    """S1 configuration factory"""
    
    @staticmethod
    def create_light_test_config() -> Dict[str, Any]:
        """Lightweight test configuration: quick validation"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[1],  # minimal concurrency
                rps_patterns=[LoadPattern.CONSTANT],
                message_types=[MessageType.SHORT],
                test_duration_seconds=5,  # very short duration
                base_rps=1  # minimal RPS
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(10, 50),
                packet_loss_rate=0.01,
                reorder_probability=0.005,
                enable_connection_drops=False  # no connection drops in light test
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
        """Standard test configuration: balance performance and coverage"""
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
        """Stress test configuration: high-intensity comprehensive test"""
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
        """Configuration optimized for a specific protocol"""
        base_config = S1ConfigFactory.create_standard_test_config()
        
        if protocol_name.lower() == 'acp':
            # ACP (HTTP synchronous RPC) - sensitive to latency spikes
            base_config['load_config'].base_rps = 8  # reduce RPS to avoid thread-pool bottlenecks
            base_config['disturbance_config'].jitter_ms_range = (5, 50)  # lower jitter
            base_config['disturbance_config'].connection_drop_interval = 60  # fewer connection drops
            
        elif protocol_name.lower() == 'anp':
            # ANP (long-lived connections/session-based) - higher setup cost but stable in steady state
            base_config['load_config'].base_rps = 12  # can sustain higher RPS
            base_config['disturbance_config'].connection_drop_interval = 90  # fewer connection drops
            base_config['disturbance_config'].packet_loss_rate = 0.03  # test robustness to packet loss
            
        elif protocol_name.lower() == 'a2a':
            # A2A (hybrid) - depends on specific deployment
            # keep standard config but add burst tests
            base_config['load_config'].rps_patterns.append(LoadPattern.BURST)
            base_config['load_config'].burst_multiplier = 3.5
            
        elif protocol_name.lower() == 'agora':
            # Agora (platform network) - backpressure-friendly but may show heavy-tail latency
            base_config['load_config'].base_rps = 15  # higher RPS to test backpressure
            base_config['load_config'].test_duration_seconds = 75  # longer to observe heavy tail
            base_config['disturbance_config'].bandwidth_limit_kbps = 800  # stricter bandwidth limit
        
        return base_config
    
    @staticmethod
    def create_network_focus_config() -> Dict[str, Any]:
        """Network-disturbance-focused config: emphasize network resilience"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[16],  # fixed medium concurrency
                rps_patterns=[LoadPattern.CONSTANT],  # fixed pattern
                message_types=[MessageType.SHORT, MessageType.LONG],  # test different payload sizes
                test_duration_seconds=120,  # longer to observe network effects
                base_rps=8
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(50, 300),  # high jitter
                packet_loss_rate=0.08,      # high loss rate
                reorder_probability=0.05,   # high reordering probability
                bandwidth_limit_kbps=200,   # strict bandwidth limit
                connection_drop_interval=20, # frequent connection drops
                enable_jitter=True,
                enable_packet_loss=True,
                enable_reorder=True,
                enable_bandwidth_limit=True,
                enable_connection_drops=True
            ),
            'attack_config': AttackNoiseConfig(
                # Lower attack intensity; focus on network disturbances
                malicious_registration_rate=2,
                spam_message_rate=10,
                replay_attack_rate=1,
                dos_request_rate=20,
                probe_query_rate=5
            )
        }
    
    @staticmethod
    def create_attack_focus_config() -> Dict[str, Any]:
        """Attack-noise-focused config: emphasize attack resistance"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[16],  # fixed medium concurrency
                rps_patterns=[LoadPattern.CONSTANT, LoadPattern.BURST],  # test burst impact on attacks
                message_types=[MessageType.SHORT],  # simplified message type
                test_duration_seconds=90,
                base_rps=10
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                # Minimal network disturbance to isolate attack effects
                jitter_ms_range=(5, 30),
                packet_loss_rate=0.005,
                reorder_probability=0.001,
                enable_connection_drops=False
            ),
            'attack_config': AttackNoiseConfig(
                malicious_registration_rate=20,  # high-rate malicious registration
                spam_message_rate=100,           # high-rate spam messages
                replay_attack_rate=15,           # high-rate replay attacks
                dos_request_rate=200,            # high-rate DoS
                probe_query_rate=30              # high-rate side-channel queries
            )
        }
    
    @staticmethod
    def create_latency_focus_config() -> Dict[str, Any]:
        """Latency-focused config: emphasize latency distribution"""
        return {
            'load_config': LoadMatrixConfig(
                concurrent_levels=[4, 16, 64],   # multi-level concurrency
                rps_patterns=[LoadPattern.CONSTANT],  # stable pattern
                message_types=[MessageType.SHORT, MessageType.LONG, MessageType.STREAMING],  # all message types
                test_duration_seconds=180,       # long duration
                base_rps=5                       # lower RPS to ensure quality
            ),
            'disturbance_config': NetworkDisturbanceConfig(
                jitter_ms_range=(1, 20),        # low jitter for precise measurement
                packet_loss_rate=0.001,         # very low loss
                reorder_probability=0.0005,     # very low reordering
                enable_connection_drops=False   # do not disturb latency measurement
            ),
            'attack_config': AttackNoiseConfig(
                # Minimal attack noise
                malicious_registration_rate=1,
                spam_message_rate=5,
                replay_attack_rate=1,
                dos_request_rate=10,
                probe_query_rate=2
            )
        }
    
    @staticmethod
    def create_tester_from_config(protocol_name: str, config_dict: Dict[str, Any]) -> S1BusinessContinuityTester:
        """Create tester from configuration dictionary"""
        return S1BusinessContinuityTester(
            protocol_name=protocol_name,
            load_config=config_dict['load_config'],
            disturbance_config=config_dict['disturbance_config'],
            attack_config=config_dict['attack_config']
        )
    
    @staticmethod
    def get_available_configs() -> List[str]:
        """Get available configuration names"""
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
        """Create configuration by name"""
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
            raise ValueError(f"Unknown configuration name: {config_name}. Available: {list(config_map.keys())}")
        
        return config_map[config_name]()


def create_s1_tester(protocol_name: str, 
                     config_name: str = 'standard') -> S1BusinessContinuityTester:
    """Convenience function: create an S1 tester"""
    if config_name == 'protocol_optimized':
        config = S1ConfigFactory.create_protocol_optimized_config(protocol_name)
    else:
        config = S1ConfigFactory.create_config_by_name(config_name)
    
    return S1ConfigFactory.create_tester_from_config(protocol_name, config)


# Predefined medical scenario message templates
MEDICAL_SCENARIO_TEMPLATES = {
    MessageType.SHORT: [
    "Patient {patient_id} shows abnormal blood pressure; request consultation",
    "Surgery {surgery_id} requires specialist support",
    "Examination report {report_id} indicates abnormal result",
    "Patient {patient_id} has an allergic reaction",
    "Emergency case {emergency_id} requires immediate attention",
    "Medication {drug_name} dosage needs adjustment",
    "Condition change reported for patient in room {room_id}",
    "Operating room {or_id} equipment fault report"
    ],
    
    MessageType.LONG: [
    """Detailed medical record for patient {patient_id}:
Age: {age} years, Gender: {gender}
Chief complaint: {chief_complaint}
Present illness: {present_illness}
Past medical history: {past_history}
Physical examination: {physical_exam}
Auxiliary tests: {lab_results}
Preliminary diagnosis: {diagnosis}
Treatment plan: {treatment_plan}
Please provide consultation opinions, with a special focus on handling {focus_area}.
The patient's vital signs are currently stable, but closely monitor {monitoring_focus}.
Further tests {additional_tests} are recommended to clarify diagnosis.
Please contact us if there are any questions. Thank you!""",
        
    """Multidisciplinary consultation case {case_id}:
Department: {department}
Attending physician: {attending_doctor}
Patient information: {patient_info}
Case summary: {case_summary}
Current treatment: {current_treatment}
Consultation questions: {consultation_questions}
Relevant examinations: {relevant_tests}
Imaging findings: {imaging_findings}
Laboratory indicators: {lab_values}
Please provide opinions from the following perspectives:
1. Is the diagnosis accurate?
2. Is the treatment plan reasonable?
3. Is medication adjustment needed?
4. What is the prognosis assessment?
5. Are further examinations needed?
We appreciate your valuable input. Thank you!"""
    ],
    
    MessageType.STREAMING: [
    "[Stream {stream_id} - 1/5] Patient {patient_id} vital sign monitoring started... | [Stream {stream_id} - 2/5] Blood pressure: {bp_systolic}/{bp_diastolic} mmHg, Heart rate: {heart_rate} bpm | [Stream {stream_id} - 3/5] SpO2: {spo2}%, Respiration: {resp_rate} breaths/min, Temperature: {temp}°C | [Stream {stream_id} - 4/5] ECG: {ecg_findings}, Blood gas: {blood_gas} | [Stream {stream_id} - 5/5] Conclusion: {monitoring_conclusion}, Recommendation: {recommendations}",
        
    "[Realtime {data_id} - 1/4] Surgery {surgery_id} in progress, current stage: {surgery_stage} | [Realtime {data_id} - 2/4] Anesthesia status: {anesthesia_status}, Blood loss: {blood_loss} ml | [Realtime {data_id} - 3/4] Surgical progress: {surgery_progress}, Estimated time remaining: {estimated_time} | [Realtime {data_id} - 4/4] Support needed: {support_needed}, Contact: {contact_number}"
    ]
}


def generate_medical_message(msg_type: MessageType, case_data: Dict[str, Any] = None) -> str:
    """Generate a medical scenario message"""
    import random
    
    if case_data is None:
        case_data = {}
    
    templates = MEDICAL_SCENARIO_TEMPLATES.get(msg_type, ["Default message"])
    template = random.choice(templates)
    
    # Default data
    default_data = {
        'patient_id': f"P{random.randint(1000, 9999)}",
        'surgery_id': f"S{random.randint(100, 999)}",
        'report_id': f"R{random.randint(10000, 99999)}",
        'emergency_id': f"E{random.randint(100, 999)}",
        'drug_name': random.choice(['Aspirin', 'Metoprolol', 'Clopidogrel', 'Atorvastatin']),
        'room_id': f"{random.randint(1, 20)}0{random.randint(1, 8)}",
        'or_id': f"OR-{random.randint(1, 10)}",
        'age': random.randint(25, 80),
        'gender': random.choice(['Male', 'Female']),
        'chief_complaint': random.choice(['Chest pain', 'Dyspnea', 'Abdominal pain', 'Headache', 'Fever']),
        'case_id': f"MDT{random.randint(1000, 9999)}",
        'department': random.choice(['Cardiology', 'Pulmonology', 'Gastroenterology', 'Neurology', 'Emergency']),
        'attending_doctor': f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])} {random.choice(['James', 'William', 'Benjamin', 'Lucas', 'Henry'])}",
        'stream_id': f"STR{random.randint(100, 999)}",
        'bp_systolic': random.randint(90, 180),
        'bp_diastolic': random.randint(60, 110),
        'heart_rate': random.randint(60, 120),
        'spo2': random.randint(85, 100),
        'resp_rate': random.randint(12, 30),
        'temp': round(random.uniform(36.0, 39.5), 1),
        'data_id': f"RT{random.randint(100, 999)}"
    }
    
    # Merge user-provided data
    default_data.update(case_data)
    
    try:
        return template.format(**default_data)
    except KeyError as e:
        # If the template references an unknown field, return a simplified version
        return f"Medical message - patient {default_data['patient_id']} - type {msg_type.value}"
