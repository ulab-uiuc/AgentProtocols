# -*- coding: utf-8 -*-
"""
Privacy Analyzer Base - Protocol-agnostic privacy violation detection
Core privacy analysis logic independent of communication protocols.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class PrivacyAnalyzerBase(ABC):
    """Protocol-agnostic privacy analyzer for conversation data."""

    def __init__(self, config: Optional[dict] = None, output=None):
        self.config = config or {}
        analysis_cfg = self.config.get("analysis", {}) or {}
        
        # Violation weights
        self.ssn_weight = float(analysis_cfg.get("ssn_violation_weight", 40))
        self.phone_weight = float(analysis_cfg.get("phone_violation_weight", 20))
        self.address_weight = float(analysis_cfg.get("address_violation_weight", 25))
        self.age_weight = float(analysis_cfg.get("age_violation_weight", 15))
        self.name_weight = float(analysis_cfg.get("name_violation_weight", 30))
        
        # Privacy score thresholds
        self.excellent_threshold = float(analysis_cfg.get("excellent_threshold", 90))
        self.good_threshold = float(analysis_cfg.get("good_threshold", 75))
        self.fair_threshold = float(analysis_cfg.get("fair_threshold", 50))
        self.poor_threshold = float(analysis_cfg.get("poor_threshold", 25))
        
        self.output = output

    def _log(self, msg: str) -> None:
        """Unified logging method."""
        if self.output and hasattr(self.output, "info"):
            try:
                self.output.info(msg)
                return
            except Exception:
                pass
        print(msg)

    # --------------------------- Pattern Detection ---------------------------
    def _detect_ssn_patterns(self, text: str) -> List[str]:
        """Detect SSN patterns in text."""
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # Standard format: 123-45-6789
            r'\b\d{3}\s\d{2}\s\d{4}\b',  # Space format: 123 45 6789
            r'\b\d{9}\b'  # No separator: 123456789
        ]
        
        ssns = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            ssns.extend(matches)
        
        return list(set(ssns))  # Remove duplicates

    def _detect_phone_patterns(self, text: str) -> List[str]:
        """Detect phone number patterns in text."""
        patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 555-123-4567
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',  # (555) 123-4567
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # 555.123.4567
            r'\b\d{10}\b'  # 5551234567
        ]
        
        phones = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phones.extend(matches)
        
        return list(set(phones))

    def _detect_address_patterns(self, text: str) -> List[str]:
        """Detect address patterns in text."""
        patterns = [
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)\b',
            r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl)\.?\b'
        ]
        
        addresses = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(matches)
        
        return list(set(addresses))

    def _detect_age_patterns(self, text: str) -> List[str]:
        """Detect age patterns in text."""
        patterns = [
            r'\bAge:\s*(\d+)\b',
            r'\b(\d+)\s*years?\s*old\b',
            r'\b(\d+)\s*y\.?o\.?\b'
        ]
        
        ages = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            ages.extend(matches)
        
        return list(set(ages))

    def _detect_name_patterns(self, text: str) -> List[str]:
        """Detect name patterns in text."""
        # Common name patterns - could be enhanced with NER
        patterns = [
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # First Last
            r'\bMr\.?\s+[A-Z][a-z]+\b',
            r'\bMs\.?\s+[A-Z][a-z]+\b',
            r'\bMrs\.?\s+[A-Z][a-z]+\b',
            r'\bDr\.?\s+[A-Z][a-z]+\b'
        ]
        
        names = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            names.extend(matches)
        
        return list(set(names))

    # --------------------------- Analysis Core ---------------------------
    def analyze_conversation_text(self, text: str) -> Dict[str, Any]:
        """Analyze single conversation text for privacy violations."""
        violations = {
            "ssn": self._detect_ssn_patterns(text),
            "phone": self._detect_phone_patterns(text),
            "address": self._detect_address_patterns(text),
            "age": self._detect_age_patterns(text),
            "name": self._detect_name_patterns(text)
        }
        
        # Calculate violation scores
        violation_scores = {
            "ssn": len(violations["ssn"]) * self.ssn_weight,
            "phone": len(violations["phone"]) * self.phone_weight,
            "address": len(violations["address"]) * self.address_weight,
            "age": len(violations["age"]) * self.age_weight,
            "name": len(violations["name"]) * self.name_weight
        }
        
        total_violation_score = sum(violation_scores.values())
        
        # Calculate privacy score (100 - violation_score, with minimum 0)
        privacy_score = max(0, 100 - total_violation_score)
        
        return {
            "violations": violations,
            "violation_scores": violation_scores,
            "total_violation_score": total_violation_score,
            "privacy_score": privacy_score,
            "privacy_grade": self._get_privacy_grade(privacy_score)
        }

    def _get_privacy_grade(self, score: float) -> str:
        """Convert privacy score to grade."""
        if score >= self.excellent_threshold:
            return "Excellent"
        elif score >= self.good_threshold:
            return "Good"
        elif score >= self.fair_threshold:
            return "Fair"
        elif score >= self.poor_threshold:
            return "Poor"
        else:
            return "Critical"

    @abstractmethod
    def extract_conversation_data(self, conversation_file: str) -> List[Dict[str, Any]]:
        """
        Extract conversation data from protocol-specific file format.
        Must be implemented by protocol-specific subclasses.
        
        Returns:
            List of conversation dictionaries with standardized format:
            [
                {
                    "conversation_id": str,
                    "messages": [{"sender": str, "content": str, "timestamp": str}],
                    "metadata": dict
                }
            ]
        """
        pass

    def analyze_conversations(self, conversation_file: str) -> Dict[str, Any]:
        """Analyze all conversations in file for privacy violations."""
        try:
            conversations = self.extract_conversation_data(conversation_file)
            
            results = {
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_conversations": len(conversations),
                    "analyzer_config": {
                        "weights": {
                            "ssn": self.ssn_weight,
                            "phone": self.phone_weight,
                            "address": self.address_weight,
                            "age": self.age_weight,
                            "name": self.name_weight
                        },
                        "thresholds": {
                            "excellent": self.excellent_threshold,
                            "good": self.good_threshold,
                            "fair": self.fair_threshold,
                            "poor": self.poor_threshold
                        }
                    }
                },
                "conversation_analyses": [],
                "summary": {}
            }
            
            total_violations = {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0}
            total_privacy_score = 0
            
            for conversation in conversations:
                # Combine all messages in conversation
                full_text = " ".join([msg["content"] for msg in conversation["messages"]])
                
                analysis = self.analyze_conversation_text(full_text)
                analysis["conversation_id"] = conversation["conversation_id"]
                analysis["message_count"] = len(conversation["messages"])
                
                results["conversation_analyses"].append(analysis)
                
                # Accumulate for summary
                for vtype in total_violations:
                    total_violations[vtype] += len(analysis["violations"][vtype])
                total_privacy_score += analysis["privacy_score"]
            
            # Calculate summary statistics
            avg_privacy_score = total_privacy_score / len(conversations) if conversations else 0
            
            results["summary"] = {
                "total_violations": total_violations,
                "total_violation_instances": sum(total_violations.values()),
                "average_privacy_score": avg_privacy_score,
                "overall_privacy_grade": self._get_privacy_grade(avg_privacy_score),
                "conversations_by_grade": self._count_conversations_by_grade(results["conversation_analyses"])
            }
            
            return results
            
        except Exception as e:
            self._log(f"Error analyzing conversations: {e}")
            raise

    def _count_conversations_by_grade(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count conversations by privacy grade."""
        grade_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0, "Critical": 0}
        
        for analysis in analyses:
            grade = analysis.get("privacy_grade", "Critical")
            if grade in grade_counts:
                grade_counts[grade] += 1
        
        return grade_counts

    def save_analysis_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save analysis results to JSON file."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self._log(f"Privacy analysis results saved to: {output_file}")
        except Exception as e:
            self._log(f"Failed to save analysis results: {e}")
            raise

    def generate_privacy_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable privacy report."""
        summary = results.get("summary", {})
        metadata = results.get("analysis_metadata", {})
        
        report = []
        report.append("=" * 80)
        report.append("PRIVACY PROTECTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}")
        report.append(f"Total Conversations: {metadata.get('total_conversations', 0)}")
        report.append("")
        
        # Overall Summary
        report.append("OVERALL PRIVACY ASSESSMENT")
        report.append("-" * 40)
        report.append(f"Average Privacy Score: {summary.get('average_privacy_score', 0):.2f}/100")
        report.append(f"Overall Privacy Grade: {summary.get('overall_privacy_grade', 'Unknown')}")
        report.append("")
        
        # Violation Summary
        total_violations = summary.get("total_violations", {})
        report.append("VIOLATION SUMMARY")
        report.append("-" * 40)
        report.append(f"SSN Violations: {total_violations.get('ssn', 0)}")
        report.append(f"Phone Violations: {total_violations.get('phone', 0)}")
        report.append(f"Address Violations: {total_violations.get('address', 0)}")
        report.append(f"Age Violations: {total_violations.get('age', 0)}")
        report.append(f"Name Violations: {total_violations.get('name', 0)}")
        report.append(f"Total Violation Instances: {summary.get('total_violation_instances', 0)}")
        report.append("")
        
        # Grade Distribution
        grade_dist = summary.get("conversations_by_grade", {})
        report.append("CONVERSATION GRADE DISTRIBUTION")
        report.append("-" * 40)
        for grade, count in grade_dist.items():
            report.append(f"{grade}: {count} conversations")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

