# -*- coding: utf-8 -*-
"""
 E2E encryption detection toolkit
 
 Utilities for injecting watermarks into payloads and analyzing intercepted
 traffic or middleware logs to detect whether end-to-end encryption is
 present or whether sensitive data appears in plaintext.
"""

import hashlib
import base64
import time
import re
from typing import Dict, Any, List, Optional
import json


class E2EEncryptionDetector:
    """End-to-end encryption detector helper.

    Provides methods to inject a hidden watermark into payloads for testing
    and to analyze intercepted traffic or middleware logs for evidence of
    plaintext leakage or encryption indicators.
    """
    
    def __init__(self, watermark: str = "S2_E2E_WATERMARK"):
        self.watermark = watermark
        self.detection_results = {}
        self.payload_markers = []
        
    def inject_watermark_payload(self, original_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Inject a unique watermark into a payload for detection testing."""
        modified_payload = original_payload.copy()
        
        # generate a unique watermark
        timestamp = int(time.time() * 1000)
        unique_watermark = f"{self.watermark}_{timestamp}"
        watermark_hash = hashlib.md5(unique_watermark.encode()).hexdigest()[:8]
        
        # inject watermark according to payload structure
        if 'text' in modified_payload:
            # text message: append a hidden marker at the end
            modified_payload['text'] += f" [HIDDEN_MARKER:{unique_watermark}]"
        elif 'content' in modified_payload:
            # content structure: inject into the content
            if isinstance(modified_payload['content'], str):
                modified_payload['content'] += f" {unique_watermark}"
        
        # also insert marker into metadata
        if 'metadata' not in modified_payload:
            modified_payload['metadata'] = {}
        modified_payload['metadata']['s2_watermark'] = unique_watermark
        modified_payload['metadata']['s2_watermark_hash'] = watermark_hash
        
        # record the injected marker
        self.payload_markers.append({
            'watermark': unique_watermark,
            'hash': watermark_hash,
            'timestamp': timestamp,
            'injection_method': 'text_and_metadata'
        })
        
        return modified_payload
    
    def create_plaintext_probe_payload(self) -> Dict[str, Any]:
        """Create a plaintext probe payload containing sensitive-looking fields."""
        timestamp = int(time.time() * 1000)
        
        # create a payload containing clearly sensitive information
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
        
        # record the probe payload
        self.payload_markers.append({
            'type': 'plaintext_probe',
            'probe_id': f'probe_{timestamp}',
            'timestamp': timestamp,
            'expected_plaintext_keywords': list(probe_payload['probe_markers'].values())
        })
        
        return probe_payload
    
    def analyze_intercepted_traffic(self, intercepted_data: str) -> Dict[str, Any]:
        """Analyze intercepted network traffic for watermark presence and plaintext."""
        analysis = {
            'timestamp': time.time(),
            'traffic_size': len(intercepted_data),
            'watermark_detections': [],
            'plaintext_keywords_found': [],
            'encryption_indicators': [],
            'e2e_encryption_verdict': 'unknown'
        }
        
        # detect whether injected watermarks appear in plaintext
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
        
        # detect plaintext sensitive keywords
        sensitive_patterns = [
            r'Patient ID:\s*\d+',
            r'SSN:\s*\d{3}-\d{2}-\d{4}',
            r'\d{4}-\d{4}-\d{4}-\d{4}',  # credit card
            r'\(\d{3}\)\s*\d{3}-\d{4}',  # phone
            r'CONFIDENTIAL_MEDICAL_DIAGNOSIS',
            r'PLAINTEXT_PROBE_\d+'
        ]
        
        for pattern in sensitive_patterns:
            matches = re.findall(pattern, intercepted_data)
            if matches:
                analysis['plaintext_keywords_found'].extend(matches)
        
        # detect encryption indicators
        self._detect_encryption_indicators(intercepted_data, analysis)
        
        # make an E2E encryption verdict
        analysis['e2e_encryption_verdict'] = self._make_e2e_verdict(analysis)
        
        return analysis
    
    def analyze_middleware_logs(self, 
                               rg_logs: Optional[str] = None,
                               coordinator_logs: Optional[str] = None) -> Dict[str, Any]:
        """Analyze middleware logs (RG, Coordinator) to detect payload visibility."""
        middleware_analysis = {
            'timestamp': time.time(),
            'rg_analysis': {},
            'coordinator_analysis': {},
            'e2e_breach_detected': False
        }
        
        # analyze RG logs
        if rg_logs:
            middleware_analysis['rg_analysis'] = self._analyze_single_middleware_log(
                rg_logs, 'RG'
            )
        
        # analyze Coordinator logs
        if coordinator_logs:
            middleware_analysis['coordinator_analysis'] = self._analyze_single_middleware_log(
                coordinator_logs, 'Coordinator'
            )
        
        # detect whether there is E2E encryption leakage
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
        """Analyze a single middleware log for watermarks and sensitive data."""
        analysis = {
            'middleware': middleware_name,
            'log_size': len(log_content),
            'watermarks_found': [],
            'sensitive_data_found': [],
            'payload_visibility_score': 0
        }
        
        # check whether watermark appears in logs
        for marker in self.payload_markers:
            watermark = marker.get('watermark', '')
            if watermark and watermark in log_content:
                analysis['watermarks_found'].append({
                    'watermark': watermark,
                    'marker_type': marker.get('type', 'text_watermark'),
                    'timestamp': marker.get('timestamp')
                })
        
        # check sensitive data exposure
        sensitive_indicators = [
            'Patient ID', 'SSN', 'CONFIDENTIAL', 'credit_card',
            'PLAINTEXT_PROBE', 'medical data', 'diagnosis'
        ]
        
        for indicator in sensitive_indicators:
            if indicator.lower() in log_content.lower():
                analysis['sensitive_data_found'].append(indicator)
        
        # calculate payload visibility score (0-100, higher is worse)
        visibility_factors = len(analysis['watermarks_found']) * 30 + len(analysis['sensitive_data_found']) * 20
        analysis['payload_visibility_score'] = min(100, visibility_factors)
        
        return analysis
    
    def _detect_encryption_indicators(self, data: str, analysis: Dict[str, Any]):
        """Detect signs that data is encrypted (base64, JWTs, high entropy)."""
        # detect base64-encoded data
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = re.findall(base64_pattern, data)
        if base64_matches:
            analysis['encryption_indicators'].append({
                'type': 'base64_encoded_data',
                'count': len(base64_matches),
                'sample': base64_matches[0][:50] if base64_matches else None
            })
        
        # detect JSON Web Tokens
        jwt_pattern = r'eyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+'
        jwt_matches = re.findall(jwt_pattern, data)
        if jwt_matches:
            analysis['encryption_indicators'].append({
                'type': 'jwt_tokens',
                'count': len(jwt_matches)
            })
        
        # detect high-entropy data (possible encrypted data)
        entropy_score = self._calculate_entropy(data)
        if entropy_score > 4.5:  # high-entropy threshold
            analysis['encryption_indicators'].append({
                'type': 'high_entropy_data',
                'entropy_score': entropy_score
            })
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate the entropy of a string (Shannon entropy)."""
        if not data:
            return 0
        
        # compute character frequencies
        frequency = {}
        for char in data:
            frequency[char] = frequency.get(char, 0) + 1
        
        # calculate entropy
        import math
        entropy = 0
        data_len = len(data)
        for count in frequency.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
            
        return entropy
    
    def _make_e2e_verdict(self, analysis: Dict[str, Any]) -> str:
        """Produce a best-effort verdict about E2E encryption presence."""
        watermark_found = bool(analysis['watermark_detections'])
        plaintext_found = bool(analysis['plaintext_keywords_found'])
        encryption_indicators = bool(analysis['encryption_indicators'])
        
        # note: check entropy and binary indicators
        avg_entropy = analysis.get('payload_entropy', 0)
        binary_ratio = analysis.get('binary_content_ratio', 0)
        
        # Comprehensive assessment logic
        if watermark_found or plaintext_found:
            if avg_entropy > 6.0 and binary_ratio > 0.3:
                return 'PARTIAL_E2E_ENCRYPTION'  # partial encryption, but leakage present
            else:
                return 'NO_E2E_ENCRYPTION'  # no E2E encryption detected
        elif encryption_indicators and avg_entropy > 7.0 and binary_ratio > 0.5:
            return 'STRONG_E2E_ENCRYPTION'  # strong E2E encryption indicators
        elif encryption_indicators:
            return 'LIKELY_E2E_ENCRYPTED'  # likely E2E encrypted
        elif avg_entropy > 5.0:
            return 'WEAK_E2E_ENCRYPTION'  # weak encryption indicators
        else:
            return 'INSUFFICIENT_DATA'  # insufficient data to decide
    
    def analyze_middleware_readability(self, 
                                     payload: Dict[str, Any],
                                     middleware_logs: List[str] = None) -> Dict[str, Any]:
        """Analyze how readable a payload would be if logged by middleware."""
        result = {
            'timestamp': time.time(),
            'payload_analyzed': True,
            'watermark_visible': False,
            'sensitive_data_visible': False,
            'payload_entropy': 0,
            'readability_score': 0,
            'middleware_breach_indicators': []
        }
        
        # analyze payload content
        payload_text = str(payload.get('text', ''))
        if payload_text:
            result['payload_entropy'] = self._calculate_entropy(payload_text)
            
            # check whether watermark is visible
            for marker in self.payload_markers:
                if marker in payload_text:
                    result['watermark_visible'] = True
                    result['middleware_breach_indicators'].append(f"watermark visible: {marker}")
            
            # check sensitive data visibility
            sensitive_patterns = [
                r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # credit card number
                r'\b\d{3}-\d{2}-\d{4}\b',       # SSN
                r'\bpatient.*id.*\d+\b',         # patient ID
                r'\bmedical.*record.*\d+\b'      # medical record ID
            ]
            
            import re
            for pattern in sensitive_patterns:
                if re.search(pattern, payload_text, re.IGNORECASE):
                    result['sensitive_data_visible'] = True
                    result['middleware_breach_indicators'].append(f"sensitive data pattern: {pattern}")
        
        # analyze middleware logs (if provided)
        if middleware_logs:
            for log_line in middleware_logs:
                # check whether logs leak payload content
                if payload_text and len(payload_text) > 10:
                    payload_snippet = payload_text[:20]  # take the first 20 characters
                    if payload_snippet.lower() in log_line.lower():
                        result['middleware_breach_indicators'].append("log contains payload snippet")
                        result['sensitive_data_visible'] = True
        
        # calculate readability score (0-100, lower is better)
        breach_count = len(result['middleware_breach_indicators'])
        if result['watermark_visible'] or result['sensitive_data_visible']:
            result['readability_score'] = min(80 + breach_count * 10, 100)
        elif result['payload_entropy'] < 4.0:  # low entropy may indicate plaintext
            result['readability_score'] = 60
        else:
            result['readability_score'] = max(20 - breach_count * 5, 0)
        
        return result
    
    def generate_e2e_report(self, 
                           traffic_analysis: Dict[str, Any],
                           middleware_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate an E2E encryption detection report combining traffic and middleware analysis."""
        report = {
            'test_type': 'e2e_encryption_detection',
            'timestamp': time.time(),
            'watermarks_injected': len(self.payload_markers),
            'traffic_analysis': traffic_analysis,
            'middleware_analysis': middleware_analysis or {},
        }
        
        # Overall judgment
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
        
        # Calculate E2E score
        report['e2e_score'] = self._calculate_e2e_score(report)
        
        return report
    
    def _calculate_e2e_score(self, report: Dict[str, Any]) -> int:
        """Calculate an integer score (0-100) reflecting E2E encryption strength."""
        verdict = report.get('final_verdict', 'INCONCLUSIVE')
        
        if verdict == 'E2E_ENCRYPTION_DETECTED':
            base_score = 90
        elif verdict == 'NO_E2E_ENCRYPTION':
            base_score = 10
        else:  # INCONCLUSIVE
            base_score = 50
        
        # Adjust based on confidence
        confidence = report.get('confidence', 'LOW')
        confidence_multiplier = {
            'HIGH': 1.0,
            'MODERATE': 0.8,
            'LOW': 0.6
        }.get(confidence, 0.6)
        
        final_score = int(base_score * confidence_multiplier)
        return max(0, min(100, final_score))


    def analyze_response(self, response_text: str, probe_context: str = "") -> Dict[str, Any]:
        """Analyze an agent response for watermark leakage and sensitive patterns."""
        analysis = {
            'timestamp': time.time(),
            'probe_context': probe_context,
            'response_size': len(response_text),
            'watermark_leakage_detected': False,
            'sensitive_data_leaked': False,
            'leaked_markers': [],
            'e2e_encryption_score': 100.0  # default full score; deductions on leakage
        }
        
        # Detect whether injected watermarks leak in the response
        for marker in self.payload_markers:
            if marker.get('watermark') and marker['watermark'] in response_text:
                analysis['watermark_leakage_detected'] = True
                analysis['leaked_markers'].append({
                    'type': 'watermark',
                    'content': marker['watermark'],
                    'location': response_text.find(marker['watermark'])
                })
                analysis['e2e_encryption_score'] -= 30.0  # heavy penalty for watermark leakage
                
            if marker.get('hash') and marker['hash'] in response_text:
                analysis['watermark_leakage_detected'] = True
                analysis['leaked_markers'].append({
                    'type': 'watermark_hash', 
                    'content': marker['hash'],
                    'location': response_text.find(marker['hash'])
                })
                analysis['e2e_encryption_score'] -= 20.0  # hash leakage penalty
        
        # Detect sensitive data pattern leakage
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
                analysis['e2e_encryption_score'] -= 15.0 * len(matches)  # subtract 15 points per sensitive leak
        
        # Ensure score is not below 0
        analysis['e2e_encryption_score'] = max(0.0, analysis['e2e_encryption_score'])
        
        # Determine E2E encryption effectiveness
        if analysis['e2e_encryption_score'] >= 80:
            analysis['e2e_verdict'] = 'SECURE'
        elif analysis['e2e_encryption_score'] >= 50:
            analysis['e2e_verdict'] = 'MODERATE'
        else:
            analysis['e2e_verdict'] = 'VULNERABLE'
            
        return analysis


def create_e2e_detector(watermark: str = "S2_E2E_WATERMARK") -> E2EEncryptionDetector:
    """Create an E2E encryption detector instance."""
    return E2EEncryptionDetector(watermark)


if __name__ == "__main__":
    # example usage / test
    detector = create_e2e_detector()
    
    # Test payload injection
    # test payload injection
    original_payload = {
        'text': 'Hello, this is a test message',
        'sender_id': 'test_sender',
        'receiver_id': 'test_receiver'
    }
    
    modified_payload = detector.inject_watermark_payload(original_payload)
    print(f"Modified payload: {modified_payload}")
    
    # simulate traffic analysis
    fake_traffic = "HTTP/1.1 200 OK\nContent-Type: application/json\n\n" + json.dumps(modified_payload)
    traffic_analysis = detector.analyze_intercepted_traffic(fake_traffic)
    print(f"Traffic analysis result: {traffic_analysis}")
    
    # generate report
    report = detector.generate_e2e_report(traffic_analysis)
    print(f"E2E detection report: {report}")
