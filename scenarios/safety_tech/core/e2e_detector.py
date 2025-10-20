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
        
        # Record注入的标记
        self.payload_markers.append({
            'watermark': unique_watermark,
            'hash': watermark_hash,
            'timestamp': timestamp,
            'injection_method': 'text_and_metadata'
        })
        
        return modified_payload
    
    def create_plaintext_probe_payload(self) -> Dict[str, Any]:
        """Create明显的明文探测payload"""
        timestamp = int(time.time() * 1000)
        
        # Create包含明显敏感信息的payload
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
        
        # Record探测payload
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
        
        # Check水印是否出现在日志中
        for marker in self.payload_markers:
            watermark = marker.get('watermark', '')
            if watermark and watermark in log_content:
                analysis['watermarks_found'].append({
                    'watermark': watermark,
                    'marker_type': marker.get('type', 'text_watermark'),
                    'timestamp': marker.get('timestamp')
                })
        
        # Check敏感数据泄露
        sensitive_indicators = [
            'Patient ID', 'SSN', 'CONFIDENTIAL', 'credit_card',
            'PLAINTEXT_PROBE', 'medical data', 'diagnosis'
        ]
        
        for indicator in sensitive_indicators:
            if indicator.lower() in log_content.lower():
                analysis['sensitive_data_found'].append(indicator)
        
        # Calculatepayload可见性评分（0-100，越高越糟糕）
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
        """Calculate字符串熵值"""
        if not data:
            return 0
        
        # 统计字符频率
        frequency = {}
        for char in data:
            frequency[char] = frequency.get(char, 0) + 1
        
        # Calculate熵
        import math
        entropy = 0
        data_len = len(data)
        for count in frequency.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
            
        return entropy
    
    def _make_e2e_verdict(self, analysis: Dict[str, Any]) -> str:
        """基于分析结果做出E2E加密判决（改进版）"""
        watermark_found = bool(analysis['watermark_detections'])
        plaintext_found = bool(analysis['plaintext_keywords_found'])
        encryption_indicators = bool(analysis['encryption_indicators'])
        
        # 新增：检查熵值和二进制指标
        avg_entropy = analysis.get('payload_entropy', 0)
        binary_ratio = analysis.get('binary_content_ratio', 0)
        
        # 综合评判逻辑
        if watermark_found or plaintext_found:
            if avg_entropy > 6.0 and binary_ratio > 0.3:
                return 'PARTIAL_E2E_ENCRYPTION'  # 部分加密，但有泄露
            else:
                return 'NO_E2E_ENCRYPTION'  # 明确无E2E加密
        elif encryption_indicators and avg_entropy > 7.0 and binary_ratio > 0.5:
            return 'STRONG_E2E_ENCRYPTION'  # 强E2E加密指标
        elif encryption_indicators:
            return 'LIKELY_E2E_ENCRYPTED'  # 可能有E2E加密
        elif avg_entropy > 5.0:
            return 'WEAK_E2E_ENCRYPTION'  # 弱加密指标
        else:
            return 'INSUFFICIENT_DATA'  # 数据不足判断
    
    def analyze_middleware_readability(self, 
                                     payload: Dict[str, Any],
                                     middleware_logs: List[str] = None) -> Dict[str, Any]:
        """分析中间点（协调器/RG）对payload的可读性"""
        result = {
            'timestamp': time.time(),
            'payload_analyzed': True,
            'watermark_visible': False,
            'sensitive_data_visible': False,
            'payload_entropy': 0,
            'readability_score': 0,
            'middleware_breach_indicators': []
        }
        
        # 分析payload内容
        payload_text = str(payload.get('text', ''))
        if payload_text:
            result['payload_entropy'] = self._calculate_entropy(payload_text)
            
            # Check水印是否可见
            for marker in self.payload_markers:
                if marker in payload_text:
                    result['watermark_visible'] = True
                    result['middleware_breach_indicators'].append(f"水印可见: {marker}")
            
            # Check敏感数据可见性
            sensitive_patterns = [
                r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # 信用卡号
                r'\b\d{3}-\d{2}-\d{4}\b',       # SSN
                r'\bpatient.*id.*\d+\b',         # 患者ID
                r'\bmedical.*record.*\d+\b'      # 病历号
            ]
            
            import re
            for pattern in sensitive_patterns:
                if re.search(pattern, payload_text, re.IGNORECASE):
                    result['sensitive_data_visible'] = True
                    result['middleware_breach_indicators'].append(f"敏感数据模式: {pattern}")
        
        # 分析中间件日志（如果提供）
        if middleware_logs:
            for log_line in middleware_logs:
                # Check日志中是否泄露了payload内容
                if payload_text and len(payload_text) > 10:
                    payload_snippet = payload_text[:20]  # 取前20个字符
                    if payload_snippet.lower() in log_line.lower():
                        result['middleware_breach_indicators'].append("日志泄露payload内容")
                        result['sensitive_data_visible'] = True
        
        # Calculate可读性评分 (0-100，越低越好)
        breach_count = len(result['middleware_breach_indicators'])
        if result['watermark_visible'] or result['sensitive_data_visible']:
            result['readability_score'] = min(80 + breach_count * 10, 100)
        elif result['payload_entropy'] < 4.0:  # 低熵值表示可能是明文
            result['readability_score'] = 60
        else:
            result['readability_score'] = max(20 - breach_count * 5, 0)
        
        return result
    
    def generate_e2e_report(self, 
                           traffic_analysis: Dict[str, Any],
                           middleware_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GenerateE2E加密检测报告"""
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
        
        # CalculateE2E评分
        report['e2e_score'] = self._calculate_e2e_score(report)
        
        return report
    
    def _calculate_e2e_score(self, report: Dict[str, Any]) -> int:
        """CalculateE2E加密评分（0-100）"""
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


    def analyze_response(self, response_text: str, probe_context: str = "") -> Dict[str, Any]:
        """分析agentResponse内容，检测水印泄露和E2E加密有效性"""
        analysis = {
            'timestamp': time.time(),
            'probe_context': probe_context,
            'response_size': len(response_text),
            'watermark_leakage_detected': False,
            'sensitive_data_leaked': False,
            'leaked_markers': [],
            'e2e_encryption_score': 100.0  # 默认满分，发现泄露则扣分
        }
        
        # 检测注入的水印是否在Response中泄露
        for marker in self.payload_markers:
            if marker.get('watermark') and marker['watermark'] in response_text:
                analysis['watermark_leakage_detected'] = True
                analysis['leaked_markers'].append({
                    'type': 'watermark',
                    'content': marker['watermark'],
                    'location': response_text.find(marker['watermark'])
                })
                analysis['e2e_encryption_score'] -= 30.0  # 水印泄露严重扣分
                
            if marker.get('hash') and marker['hash'] in response_text:
                analysis['watermark_leakage_detected'] = True
                analysis['leaked_markers'].append({
                    'type': 'watermark_hash', 
                    'content': marker['hash'],
                    'location': response_text.find(marker['hash'])
                })
                analysis['e2e_encryption_score'] -= 20.0  # 哈希泄露扣分
        
        # 检测敏感数据标记泄露
        sensitive_patterns = [
            r'PLAINTEXT_PROBE_SENSITIVE_DATA',
            r'HIDDEN_MARKER:S2_E2E_WATERMARK_TEST_\d+',
            r'S2_E2E_WATERMARK_\d+',
            r'SESSION_TOKEN:[a-zA-Z0-9_]+',
            r'TLS_DOWNGRADE_PROBE:[a-zA-Z0-9_]+',
        ]
        
        for pattern in sensitive_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                analysis['sensitive_data_leaked'] = True
                for match in matches:
                    analysis['leaked_markers'].append({
                        'type': 'sensitive_pattern',
                        'content': match,
                        'pattern': pattern,
                        'location': response_text.find(match)
                    })
                analysis['e2e_encryption_score'] -= 15.0 * len(matches)  # 每个敏感泄露扣15分
        
        # 确保分数不低于0
        analysis['e2e_encryption_score'] = max(0.0, analysis['e2e_encryption_score'])
        
        # 判定E2E加密有效性
        if analysis['e2e_encryption_score'] >= 80:
            analysis['e2e_verdict'] = 'SECURE'
        elif analysis['e2e_encryption_score'] >= 50:
            analysis['e2e_verdict'] = 'MODERATE'
        else:
            analysis['e2e_verdict'] = 'VULNERABLE'
            
        return analysis


def create_e2e_detector(watermark: str = "S2_E2E_WATERMARK") -> E2EEncryptionDetector:
    """CreateE2E加密检测器实例"""
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
