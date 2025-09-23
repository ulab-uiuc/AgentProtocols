# -*- coding: utf-8 -*-
"""
S2 E2E负载加密存在性检测
通过水印注入与中间点观察，判定协议是否具有端到端负载加密能力
"""

import hashlib
import base64
import time
import re
from typing import Dict, Any, List, Optional
import json


class E2EEncryptionDetector:
    """端到端加密检测器"""
    
    def __init__(self, watermark: str = "S2_E2E_WATERMARK"):
        self.watermark = watermark
        self.detection_results = {}
        self.payload_markers = []
        
    def inject_watermark_payload(self, original_payload: Dict[str, Any]) -> Dict[str, Any]:
        """向payload注入水印标记用于检测"""
        modified_payload = original_payload.copy()
        
        # 生成唯一水印
        timestamp = int(time.time() * 1000)
        unique_watermark = f"{self.watermark}_{timestamp}"
        watermark_hash = hashlib.md5(unique_watermark.encode()).hexdigest()[:8]
        
        # 根据payload结构注入水印
        if 'text' in modified_payload:
            # 文本消息：在末尾添加隐藏标记
            modified_payload['text'] += f" [HIDDEN_MARKER:{unique_watermark}]"
        elif 'content' in modified_payload:
            # content结构：注入到内容中
            if isinstance(modified_payload['content'], str):
                modified_payload['content'] += f" {unique_watermark}"
        
        # 在metadata中也注入标记
        if 'metadata' not in modified_payload:
            modified_payload['metadata'] = {}
        modified_payload['metadata']['s2_watermark'] = unique_watermark
        modified_payload['metadata']['s2_watermark_hash'] = watermark_hash
        
        # 记录注入的标记
        self.payload_markers.append({
            'watermark': unique_watermark,
            'hash': watermark_hash,
            'timestamp': timestamp,
            'injection_method': 'text_and_metadata'
        })
        
        return modified_payload
    
    def create_plaintext_probe_payload(self) -> Dict[str, Any]:
        """创建明显的明文探测payload"""
        timestamp = int(time.time() * 1000)
        
        # 创建包含明显敏感信息的payload
        probe_payload = {
            'text': f'PLAINTEXT_PROBE_{timestamp}: This is sensitive medical data - Patient ID: 12345, SSN: 555-66-7777',
            'sender_id': 'E2E_PROBE_SENDER',
            'receiver_id': 'E2E_PROBE_RECEIVER',
            'metadata': {
                's2_probe_type': 'plaintext_detection',
                's2_probe_id': f'probe_{timestamp}',
                'sensitive_keywords': ['Patient ID', 'SSN', 'medical data'],
                'expected_encryption': True
            },
            'probe_markers': {
                'credit_card': '4111-1111-1111-1111',
                'phone': '(555) 123-4567', 
                'email': 'patient@hospital.com',
                'diagnosis': 'CONFIDENTIAL_MEDICAL_DIAGNOSIS'
            }
        }
        
        # 记录探测payload
        self.payload_markers.append({
            'type': 'plaintext_probe',
            'probe_id': f'probe_{timestamp}',
            'timestamp': timestamp,
            'expected_plaintext_keywords': list(probe_payload['probe_markers'].values())
        })
        
        return probe_payload
    
    def analyze_intercepted_traffic(self, intercepted_data: str) -> Dict[str, Any]:
        """分析拦截到的网络流量，检测是否包含明文水印"""
        analysis = {
            'timestamp': time.time(),
            'traffic_size': len(intercepted_data),
            'watermark_detections': [],
            'plaintext_keywords_found': [],
            'encryption_indicators': [],
            'e2e_encryption_verdict': 'unknown'
        }
        
        # 检测注入的水印是否以明文形式出现
        for marker in self.payload_markers:
            if marker.get('watermark') and marker['watermark'] in intercepted_data:
                analysis['watermark_detections'].append({
                    'watermark': marker['watermark'],
                    'found_plaintext': True,
                    'detection_method': 'direct_string_match'
                })
            elif marker.get('hash') and marker['hash'] in intercepted_data:
                analysis['watermark_detections'].append({
                    'watermark_hash': marker['hash'],
                    'found_plaintext': True,
                    'detection_method': 'hash_match'
                })
        
        # 检测明文敏感关键字
        sensitive_patterns = [
            r'Patient ID:\s*\d+',
            r'SSN:\s*\d{3}-\d{2}-\d{4}',
            r'\d{4}-\d{4}-\d{4}-\d{4}',  # 信用卡
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # 电话
            r'CONFIDENTIAL_MEDICAL_DIAGNOSIS',
            r'PLAINTEXT_PROBE_\d+'
        ]
        
        for pattern in sensitive_patterns:
            matches = re.findall(pattern, intercepted_data)
            if matches:
                analysis['plaintext_keywords_found'].extend(matches)
        
        # 检测加密指标
        self._detect_encryption_indicators(intercepted_data, analysis)
        
        # 做出E2E加密判决
        analysis['e2e_encryption_verdict'] = self._make_e2e_verdict(analysis)
        
        return analysis
    
    def analyze_middleware_logs(self, 
                               rg_logs: Optional[str] = None,
                               coordinator_logs: Optional[str] = None) -> Dict[str, Any]:
        """分析中间件日志，检测payload是否在中间层可读"""
        middleware_analysis = {
            'timestamp': time.time(),
            'rg_analysis': {},
            'coordinator_analysis': {},
            'e2e_breach_detected': False
        }
        
        # 分析RG日志
        if rg_logs:
            middleware_analysis['rg_analysis'] = self._analyze_single_middleware_log(
                rg_logs, 'RG'
            )
        
        # 分析Coordinator日志
        if coordinator_logs:
            middleware_analysis['coordinator_analysis'] = self._analyze_single_middleware_log(
                coordinator_logs, 'Coordinator'
            )
        
        # 检测是否有E2E加密泄露
        rg_breach = middleware_analysis['rg_analysis'].get('watermarks_found', [])
        coord_breach = middleware_analysis['coordinator_analysis'].get('watermarks_found', [])
        
        middleware_analysis['e2e_breach_detected'] = bool(rg_breach or coord_breach)
        middleware_analysis['breach_sources'] = []
        
        if rg_breach:
            middleware_analysis['breach_sources'].append('RG')
        if coord_breach:
            middleware_analysis['breach_sources'].append('Coordinator')
            
        return middleware_analysis
    
    def _analyze_single_middleware_log(self, log_content: str, middleware_name: str) -> Dict[str, Any]:
        """分析单个中间件日志"""
        analysis = {
            'middleware': middleware_name,
            'log_size': len(log_content),
            'watermarks_found': [],
            'sensitive_data_found': [],
            'payload_visibility_score': 0
        }
        
        # 检查水印是否出现在日志中
        for marker in self.payload_markers:
            watermark = marker.get('watermark', '')
            if watermark and watermark in log_content:
                analysis['watermarks_found'].append({
                    'watermark': watermark,
                    'marker_type': marker.get('type', 'text_watermark'),
                    'timestamp': marker.get('timestamp')
                })
        
        # 检查敏感数据泄露
        sensitive_indicators = [
            'Patient ID', 'SSN', 'CONFIDENTIAL', 'credit_card',
            'PLAINTEXT_PROBE', 'medical data', 'diagnosis'
        ]
        
        for indicator in sensitive_indicators:
            if indicator.lower() in log_content.lower():
                analysis['sensitive_data_found'].append(indicator)
        
        # 计算payload可见性评分（0-100，越高越糟糕）
        visibility_factors = len(analysis['watermarks_found']) * 30 + len(analysis['sensitive_data_found']) * 20
        analysis['payload_visibility_score'] = min(100, visibility_factors)
        
        return analysis
    
    def _detect_encryption_indicators(self, data: str, analysis: Dict[str, Any]):
        """检测加密指标"""
        # Base64编码数据检测
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = re.findall(base64_pattern, data)
        if base64_matches:
            analysis['encryption_indicators'].append({
                'type': 'base64_encoded_data',
                'count': len(base64_matches),
                'sample': base64_matches[0][:50] if base64_matches else None
            })
        
        # JSON Web Token检测
        jwt_pattern = r'eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+'
        jwt_matches = re.findall(jwt_pattern, data)
        if jwt_matches:
            analysis['encryption_indicators'].append({
                'type': 'jwt_tokens',
                'count': len(jwt_matches)
            })
        
        # 高熵数据检测（可能是加密数据）
        entropy_score = self._calculate_entropy(data)
        if entropy_score > 4.5:  # 高熵阈值
            analysis['encryption_indicators'].append({
                'type': 'high_entropy_data',
                'entropy_score': entropy_score
            })
    
    def _calculate_entropy(self, data: str) -> float:
        """计算字符串熵值"""
        if not data:
            return 0
        
        # 统计字符频率
        frequency = {}
        for char in data:
            frequency[char] = frequency.get(char, 0) + 1
        
        # 计算熵
        import math
        entropy = 0
        data_len = len(data)
        for count in frequency.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
            
        return entropy
    
    def _make_e2e_verdict(self, analysis: Dict[str, Any]) -> str:
        """基于分析结果做出E2E加密判决"""
        watermark_found = bool(analysis['watermark_detections'])
        plaintext_found = bool(analysis['plaintext_keywords_found'])
        encryption_indicators = bool(analysis['encryption_indicators'])
        
        if watermark_found or plaintext_found:
            return 'NO_E2E_ENCRYPTION'  # 发现明文水印/敏感数据
        elif encryption_indicators:
            return 'LIKELY_E2E_ENCRYPTED'  # 有加密指标，无明文泄露
        else:
            return 'INSUFFICIENT_DATA'  # 数据不足判断
    
    def generate_e2e_report(self, 
                           traffic_analysis: Dict[str, Any],
                           middleware_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成E2E加密检测报告"""
        report = {
            'test_type': 'e2e_encryption_detection',
            'timestamp': time.time(),
            'watermarks_injected': len(self.payload_markers),
            'traffic_analysis': traffic_analysis,
            'middleware_analysis': middleware_analysis or {},
        }
        
        # 综合判决
        traffic_verdict = traffic_analysis.get('e2e_encryption_verdict', 'unknown')
        middleware_breach = middleware_analysis.get('e2e_breach_detected', False) if middleware_analysis else False
        
        if traffic_verdict == 'NO_E2E_ENCRYPTION' or middleware_breach:
            report['final_verdict'] = 'NO_E2E_ENCRYPTION'
            report['confidence'] = 'HIGH'
        elif traffic_verdict == 'LIKELY_E2E_ENCRYPTED' and not middleware_breach:
            report['final_verdict'] = 'E2E_ENCRYPTION_DETECTED'
            report['confidence'] = 'MODERATE'
        else:
            report['final_verdict'] = 'INCONCLUSIVE'
            report['confidence'] = 'LOW'
        
        # 计算E2E评分
        report['e2e_score'] = self._calculate_e2e_score(report)
        
        return report
    
    def _calculate_e2e_score(self, report: Dict[str, Any]) -> int:
        """计算E2E加密评分（0-100）"""
        verdict = report.get('final_verdict', 'INCONCLUSIVE')
        
        if verdict == 'E2E_ENCRYPTION_DETECTED':
            base_score = 90
        elif verdict == 'NO_E2E_ENCRYPTION':
            base_score = 10
        else:  # INCONCLUSIVE
            base_score = 50
        
        # 根据置信度调整
        confidence = report.get('confidence', 'LOW')
        confidence_multiplier = {
            'HIGH': 1.0,
            'MODERATE': 0.8,
            'LOW': 0.6
        }.get(confidence, 0.6)
        
        final_score = int(base_score * confidence_multiplier)
        return max(0, min(100, final_score))


def create_e2e_detector(watermark: str = "S2_E2E_WATERMARK") -> E2EEncryptionDetector:
    """创建E2E加密检测器实例"""
    return E2EEncryptionDetector(watermark)


if __name__ == "__main__":
    # 测试用例
    detector = create_e2e_detector()
    
    # 测试payload注入
    original_payload = {
        'text': 'Hello, this is a test message',
        'sender_id': 'test_sender',
        'receiver_id': 'test_receiver'
    }
    
    modified_payload = detector.inject_watermark_payload(original_payload)
    print(f"修改后的payload: {modified_payload}")
    
    # 模拟流量分析
    fake_traffic = "HTTP/1.1 200 OK\nContent-Type: application/json\n\n" + json.dumps(modified_payload)
    traffic_analysis = detector.analyze_intercepted_traffic(fake_traffic)
    print(f"流量分析结果: {traffic_analysis}")
    
    # 生成报告
    report = detector.generate_e2e_report(traffic_analysis)
    print(f"E2E检测报告: {report}")
