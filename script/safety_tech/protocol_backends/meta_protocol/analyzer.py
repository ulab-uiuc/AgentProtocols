# -*- coding: utf-8 -*-
"""
Meta Protocol Privacy Analyzer - Protocol-specific privacy analysis for meta protocol conversations
Extends the base privacy analyzer for meta protocol conversation format with routing information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import base privacy analyzer
try:
    from ...core.privacy_analyzer_base import PrivacyAnalyzerBase
except ImportError:
    try:
        from core.privacy_analyzer_base import PrivacyAnalyzerBase
    except ImportError:
        # Create minimal base class if not available
        class PrivacyAnalyzerBase:
            def __init__(self, config, output):
                self.config = config
                self.output = output
            
            def detect_privacy_violations(self, content):
                # Basic privacy violation detection
                violations = {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0}
                content_lower = content.lower()
                
                # Simple detection patterns
                if "ssn" in content_lower or "social security" in content_lower:
                    violations["ssn"] += 1
                if "phone" in content_lower or "number" in content_lower:
                    violations["phone"] += 1
                if "address" in content_lower or "street" in content_lower:
                    violations["address"] += 1
                if "age" in content_lower or "years old" in content_lower:
                    violations["age"] += 1
                if "name" in content_lower:
                    violations["name"] += 1
                    
                return violations
            
            def calculate_privacy_score(self, content, violations):
                # Simple scoring based on violations
                total_violations = sum(violations.values())
                if total_violations == 0:
                    return 100.0
                elif total_violations <= 2:
                    return 80.0
                elif total_violations <= 5:
                    return 60.0
                else:
                    return 40.0


class MetaPrivacyAnalyzer(PrivacyAnalyzerBase):
    """Meta protocol specific privacy analyzer for conversation data with routing analysis."""

    def extract_conversation_data(self, conversation_file: str) -> List[Dict[str, Any]]:
        """
        Extract conversation data from meta protocol conversation file format.
        
        Args:
            conversation_file: Path to meta protocol conversation JSON file
            
        Returns:
            List of standardized conversation dictionaries
        """
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # Handle different meta protocol file formats
            if isinstance(meta_data, dict):
                if "conversations" in meta_data:
                    # Format: {"conversations": [...], "protocol": "meta_xxx", "routing_info": {...}}
                    conversations = meta_data["conversations"]
                    routing_info = meta_data.get("routing_info", {})
                elif "data" in meta_data:
                    # Format: {"data": [...]}
                    conversations = meta_data["data"]
                    routing_info = {}
                else:
                    # Assume single conversation
                    conversations = [meta_data]
                    routing_info = {}
            elif isinstance(meta_data, list):
                # Direct list of conversations
                conversations = meta_data
                routing_info = {}
            else:
                self.output.warning(f"Unknown meta protocol conversation format in {conversation_file}")
                return []
            
            # Standardize conversation format with meta protocol information
            standardized_conversations = []
            for conv in conversations:
                standardized_conv = self._standardize_meta_conversation(conv, routing_info)
                if standardized_conv:
                    standardized_conversations.append(standardized_conv)
            
            return standardized_conversations
            
        except Exception as e:
            self.output.error(f"Error loading meta protocol conversation file {conversation_file}: {e}")
            return []

    def _standardize_meta_conversation(self, conv: Dict[str, Any], routing_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Standardize a single meta protocol conversation to common format.
        
        Args:
            conv: Raw conversation data
            routing_info: Meta protocol routing information
            
        Returns:
            Standardized conversation dictionary or None if invalid
        """
        try:
            # Extract basic conversation info
            conversation_id = conv.get("conversation_id", "unknown")
            protocol = conv.get("protocol", "meta_unknown")
            
            # Extract routing decision information
            routing_decision = conv.get("routing_decision") or routing_info.get("routing_decision", {})
            selected_protocol = routing_decision.get("selected_protocol", "unknown")
            
            # Extract messages
            messages = conv.get("messages", [])
            
            # Build standardized conversation
            standardized = {
                "conversation_id": conversation_id,
                "protocol": protocol,
                "selected_protocol": selected_protocol,
                "routing_info": routing_decision,
                "messages": [],
                "timestamp": conv.get("timestamp", 0)
            }
            
            # Standardize messages
            for msg in messages:
                standardized_msg = self._standardize_meta_message(msg)
                if standardized_msg:
                    standardized["messages"].append(standardized_msg)
            
            return standardized
            
        except Exception as e:
            self.output.warning(f"Error standardizing meta conversation: {e}")
            return None

    def _standardize_meta_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Standardize a single meta protocol message to common format.
        
        Args:
            msg: Raw message data
            
        Returns:
            Standardized message dictionary or None if invalid
        """
        try:
            # Extract message content
            content = msg.get("message", msg.get("content", msg.get("text", "")))
            if not content:
                return None
            
            # Extract sender and recipient
            sender = msg.get("sender", "unknown")
            recipient = msg.get("recipient", "unknown")
            
            # Extract meta protocol specific info
            meta_info = msg.get("meta_info", {})
            response_time = msg.get("response_time", 0)
            fallback = msg.get("fallback", False)
            
            return {
                "sender": sender,
                "recipient": recipient,
                "content": content,
                "timestamp": msg.get("timestamp", 0),
                "response_time": response_time,
                "meta_info": meta_info,
                "fallback_used": fallback
            }
            
        except Exception as e:
            self.output.warning(f"Error standardizing meta message: {e}")
            return None

    def analyze_conversations_from_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze privacy violations from meta protocol conversation data.
        
        Args:
            conversation_data: Meta protocol conversation data dictionary
            
        Returns:
            Privacy analysis results with meta protocol routing information
        """
        try:
            conversations = conversation_data.get("conversations", [])
            protocol = conversation_data.get("protocol", "meta_unknown")
            routing_info = conversation_data.get("routing_info", {})
            
            # Perform base privacy analysis
            analysis_results = {
                "analysis_timestamp": time.time(),
                "protocol": protocol,
                "selected_protocol": routing_info.get("selected_protocol", "unknown"),
                "routing_decision": routing_info.get("routing_decision", {}),
                "total_conversations": len(conversations),
                "conversations_analyzed": [],
                "summary": {
                    "total_violations": {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0},
                    "total_violation_instances": 0,
                    "average_privacy_score": 0.0,
                    "overall_privacy_grade": "Unknown"
                }
            }
            
            total_privacy_score = 0.0
            total_violations = {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0}
            
            # Analyze each conversation
            for conv in conversations:
                conv_analysis = self._analyze_single_meta_conversation(conv)
                analysis_results["conversations_analyzed"].append(conv_analysis)
                
                # Aggregate violations
                for violation_type, count in conv_analysis["violations"].items():
                    total_violations[violation_type] += count
                
                total_privacy_score += conv_analysis["privacy_score"]
            
            # Calculate summary statistics
            if conversations:
                analysis_results["summary"]["average_privacy_score"] = total_privacy_score / len(conversations)
            else:
                analysis_results["summary"]["average_privacy_score"] = 100.0
            
            analysis_results["summary"]["total_violations"] = total_violations
            analysis_results["summary"]["total_violation_instances"] = sum(total_violations.values())
            analysis_results["summary"]["overall_privacy_grade"] = self._calculate_privacy_grade(
                analysis_results["summary"]["average_privacy_score"]
            )
            
            # Add meta protocol specific metrics
            analysis_results["meta_protocol_metrics"] = self._analyze_routing_effectiveness(
                conversations, routing_info
            )
            
            return analysis_results
            
        except Exception as e:
            self.output.error(f"Error analyzing meta protocol conversations: {e}")
            return self._get_fallback_analysis_results(conversation_data)

    def _analyze_single_meta_conversation(self, conv: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze privacy violations in a single meta protocol conversation.
        
        Args:
            conv: Standardized conversation data
            
        Returns:
            Conversation analysis results
        """
        conversation_id = conv.get("conversation_id", "unknown")
        messages = conv.get("messages", [])
        routing_info = conv.get("routing_info", {})
        
        # Analyze privacy violations in all messages
        violations = {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0}
        privacy_scores = []
        
        for msg in messages:
            content = msg.get("content", "")
            msg_violations = self.detect_privacy_violations(content)
            msg_score = self.calculate_privacy_score(content, msg_violations)
            
            # Aggregate violations
            for violation_type, count in msg_violations.items():
                violations[violation_type] += count
            
            privacy_scores.append(msg_score)
        
        # Calculate overall conversation privacy score
        avg_privacy_score = sum(privacy_scores) / len(privacy_scores) if privacy_scores else 100.0
        
        # Check if fallback was used (may affect privacy)
        fallback_used = any(msg.get("fallback_used", False) for msg in messages)
        if fallback_used:
            # Slightly reduce privacy score if fallback was used
            avg_privacy_score = max(0, avg_privacy_score - 5.0)
        
        return {
            "conversation_id": conversation_id,
            "protocol": conv.get("protocol", "meta_unknown"),
            "selected_protocol": conv.get("selected_protocol", "unknown"),
            "routing_confidence": routing_info.get("confidence", 0.0),
            "violations": violations,
            "privacy_score": avg_privacy_score,
            "privacy_grade": self._calculate_privacy_grade(avg_privacy_score),
            "total_messages": len(messages),
            "fallback_used": fallback_used
        }

    def _analyze_routing_effectiveness(self, conversations: List[Dict[str, Any]], 
                                     routing_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the effectiveness of meta protocol routing decisions.
        
        Args:
            conversations: List of conversation data
            routing_info: Meta protocol routing information
            
        Returns:
            Routing effectiveness analysis
        """
        try:
            selected_protocol = routing_info.get("selected_protocol", "unknown")
            routing_confidence = routing_info.get("confidence", 0.0)
            routing_strategy = routing_info.get("strategy", "unknown")
            
            # Calculate average privacy score achieved
            privacy_scores = []
            fallback_count = 0
            
            for conv in conversations:
                for msg in conv.get("messages", []):
                    content = msg.get("content", "")
                    violations = self.detect_privacy_violations(content)
                    score = self.calculate_privacy_score(content, violations)
                    privacy_scores.append(score)
                    
                    if msg.get("fallback_used", False):
                        fallback_count += 1
            
            avg_achieved_privacy = sum(privacy_scores) / len(privacy_scores) if privacy_scores else 100.0
            fallback_rate = fallback_count / len(privacy_scores) if privacy_scores else 0.0
            
            # Determine if routing was effective
            routing_effectiveness = "good"
            if avg_achieved_privacy < 70:
                routing_effectiveness = "poor"
            elif avg_achieved_privacy < 85:
                routing_effectiveness = "fair"
            elif fallback_rate > 0.2:  # More than 20% fallback usage
                routing_effectiveness = "fair"
            
            return {
                "selected_protocol": selected_protocol,
                "routing_confidence": routing_confidence,
                "routing_strategy": routing_strategy,
                "achieved_privacy_score": avg_achieved_privacy,
                "fallback_rate": fallback_rate,
                "routing_effectiveness": routing_effectiveness,
                "total_messages_analyzed": len(privacy_scores)
            }
            
        except Exception as e:
            self.output.warning(f"Error analyzing routing effectiveness: {e}")
            return {
                "selected_protocol": "unknown",
                "routing_effectiveness": "unknown",
                "error": str(e)
            }

    def _get_fallback_analysis_results(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback analysis results when main analysis fails."""
        return {
            "analysis_timestamp": time.time(),
            "protocol": conversation_data.get("protocol", "meta_unknown"),
            "selected_protocol": "unknown",
            "total_conversations": len(conversation_data.get("conversations", [])),
            "error": "Analysis failed, using fallback results",
            "summary": {
                "total_violations": {"ssn": 0, "phone": 0, "address": 0, "age": 0, "name": 0},
                "total_violation_instances": 0,
                "average_privacy_score": 50.0,  # Conservative estimate
                "overall_privacy_grade": "Unknown"
            }
        }

    def _calculate_privacy_grade(self, score: float) -> str:
        """Calculate privacy grade from score."""
        if score >= 95:
            return "Excellent"
        elif score >= 90:
            return "Very Good"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"


# Import time for timestamps
import time
