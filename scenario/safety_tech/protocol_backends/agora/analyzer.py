# -*- coding: utf-8 -*-
"""
Agora Privacy Analyzer - Protocol-specific privacy analysis for Agora conversations
Extends the base privacy analyzer for Agora protocol conversation format.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import base privacy analyzer
try:
    from ...core.privacy_analyzer_base import PrivacyAnalyzerBase
except ImportError:
    from core.privacy_analyzer_base import PrivacyAnalyzerBase


class AgoraPrivacyAnalyzer(PrivacyAnalyzerBase):
    """Agora-specific privacy analyzer for conversation data."""

    def extract_conversation_data(self, conversation_file: str) -> List[Dict[str, Any]]:
        """
        Extract conversation data from Agora conversation file format.
        
        Args:
            conversation_file: Path to Agora conversation JSON file
            
        Returns:
            List of standardized conversation dictionaries
        """
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                agora_data = json.load(f)
            
            # Handle different Agora file formats
            if isinstance(agora_data, dict):
                if "conversations" in agora_data:
                    # Format: {"conversations": [...], "metadata": {...}}
                    conversations = agora_data["conversations"]
                elif "data" in agora_data:
                    # Format: {"data": [...]}
                    conversations = agora_data["data"]
                else:
                    # Assume single conversation
                    conversations = [agora_data]
            elif isinstance(agora_data, list):
                # Direct list of conversations
                conversations = agora_data
            else:
                raise ValueError("Unsupported Agora conversation file format")
            
            standardized_conversations = []
            
            for i, conv in enumerate(conversations):
                standardized_conv = self._standardize_agora_conversation(conv, i)
                if standardized_conv:
                    standardized_conversations.append(standardized_conv)
            
            self._log(f"Extracted {len(standardized_conversations)} conversations from Agora file")
            return standardized_conversations
            
        except Exception as e:
            self._log(f"Error extracting Agora conversation data: {e}")
            raise

    def _standardize_agora_conversation(self, agora_conversation: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Convert Agora conversation format to standardized format.
        
        Args:
            agora_conversation: Agora-specific conversation data
            index: Conversation index for ID generation
            
        Returns:
            Standardized conversation dictionary or None if invalid
        """
        try:
            # Extract conversation ID
            conv_id = agora_conversation.get("conversation_id", f"agora_conv_{index + 1}")
            
            # Extract messages from different Agora formats
            messages = self._extract_agora_messages(agora_conversation)
            
            if not messages:
                self._log(f"No messages found in conversation {conv_id}")
                return None
            
            # Extract metadata
            metadata = {
                "protocol": "agora",
                "original_question": agora_conversation.get("original_question", ""),
                "timestamp": agora_conversation.get("timestamp", ""),
                "agora_metadata": agora_conversation.get("metadata", {}),
                "protocol_hash": agora_conversation.get("protocolHash"),
                "protocol_sources": agora_conversation.get("protocolSources", [])
            }
            
            return {
                "conversation_id": conv_id,
                "messages": messages,
                "metadata": metadata
            }
            
        except Exception as e:
            self._log(f"Error standardizing Agora conversation {index}: {e}")
            return None

    def _extract_agora_messages(self, agora_conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and standardize messages from Agora conversation format."""
        messages = []
        
        # Try different Agora message formats
        if "messages" in agora_conversation:
            # Format: {"messages": [{"sender": "...", "message": "...", ...}]}
            for msg in agora_conversation["messages"]:
                standardized_msg = self._standardize_agora_message(msg)
                if standardized_msg:
                    messages.append(standardized_msg)
                    
        elif "body" in agora_conversation:
            # Single message format
            messages.append({
                "sender": "agora_agent",
                "content": agora_conversation["body"],
                "timestamp": agora_conversation.get("timestamp", "")
            })
                    
        elif "conversation" in agora_conversation:
            # Nested conversation format
            return self._extract_agora_messages(agora_conversation["conversation"])
            
        else:
            # Try to extract from conversation keys directly
            for key in ["original_question", "question", "patient_query"]:
                if key in agora_conversation:
                    messages.append({
                        "sender": "patient",
                        "content": agora_conversation[key],
                        "timestamp": agora_conversation.get("timestamp", "")
                    })
                    break
        
        return messages

    def _standardize_agora_message(self, agora_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize individual Agora message format."""
        try:
            # Handle different message field names
            content = (
                agora_message.get("message") or 
                agora_message.get("body") or
                agora_message.get("content") or 
                agora_message.get("text") or 
                agora_message.get("data") or 
                str(agora_message)
            )
            
            sender = (
                agora_message.get("sender") or 
                agora_message.get("agent") or 
                agora_message.get("from") or 
                "unknown"
            )
            
            timestamp = (
                agora_message.get("timestamp") or 
                agora_message.get("time") or 
                ""
            )
            
            return {
                "sender": sender,
                "content": content,
                "timestamp": timestamp
            }
            
        except Exception as e:
            self._log(f"Error standardizing Agora message: {e}")
            return None

    def analyze_conversations_from_data(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversations from direct data structure."""
        try:
            conversations = conversation_data.get("conversations", [])
            all_violations = []
            conversation_analyses = []
            
            for conv in conversations:
                conv_violations = []
                messages = conv.get("messages", [])
                
                # Combine all messages in conversation for analysis
                full_text = " ".join([msg.get("message", "") for msg in messages])
                
                # Use base analyzer for conversation analysis
                analysis = self.analyze_conversation_text(full_text)
                violations_data = analysis.get("violations", {})
                
                # Convert to violation list format
                for violation_type, instances in violations_data.items():
                    for instance in instances:
                        violation = {
                            "type": violation_type.upper(),
                            "content": str(instance),
                            "conversation_id": conv.get("conversation_id", "unknown"),
                            "protocol": "agora",
                            "timestamp": time.time()
                        }
                        conv_violations.append(violation)
                        all_violations.append(violation)
                
                # Analyze conversation-level metrics
                privacy_score = analysis.get("privacy_score", 100.0)
                conv_analysis = {
                    "conversation_id": conv.get("conversation_id", "unknown"),
                    "total_messages": len(messages),
                    "violations": conv_violations,
                    "violation_count": len(conv_violations),
                    "privacy_score": privacy_score
                }
                conversation_analyses.append(conv_analysis)
            
            # Calculate overall statistics
            total_conversations = len(conversations)
            total_violations = len(all_violations)
            
            # Calculate type-specific violation counts
            violation_counts = {}
            for violation in all_violations:
                v_type = violation.get("type", "UNKNOWN")
                violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
            
            # Calculate overall privacy score
            if total_conversations > 0:
                avg_privacy_score = sum(conv["privacy_score"] for conv in conversation_analyses) / total_conversations
            else:
                avg_privacy_score = 100.0
            
            # Calculate grade distribution
            grade_dist = {}
            for conv in conversation_analyses:
                grade = self._score_to_grade(conv["privacy_score"])
                grade_dist[grade] = grade_dist.get(grade, 0) + 1
            
            # Format for base analyzer compatibility
            return {
                "analysis_timestamp": time.time(),
                "protocol": "agora",
                "total_conversations": total_conversations,
                "total_violations": total_violations,
                "violation_counts": violation_counts,
                "average_privacy_score": avg_privacy_score,
                "privacy_grade": self._score_to_grade(avg_privacy_score),
                "conversation_analyses": conversation_analyses,
                # Add structure expected by base analyzer
                "analysis_metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "total_conversations": total_conversations,
                    "protocol": "agora"
                },
                "summary": {
                    "average_privacy_score": avg_privacy_score,
                    "overall_privacy_grade": self._score_to_grade(avg_privacy_score),
                    "total_violations": {
                        "ssn": violation_counts.get("SSN", 0),
                        "phone": violation_counts.get("PHONE", 0),
                        "address": violation_counts.get("ADDRESS", 0),
                        "age": violation_counts.get("AGE", 0),
                        "name": violation_counts.get("NAME", 0)
                    },
                    "total_violation_instances": total_violations,
                    "conversations_by_grade": grade_dist
                }
            }
            
        except Exception as e:
            self._log(f"Error in Agora conversation analysis: {e}")
            raise

    def _standardize_conversation_data(self, conv_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize conversation data from simulator."""
        try:
            messages = []
            for msg in conv_data.get("messages", []):
                messages.append({
                    "sender": msg.get("sender", "unknown"),
                    "content": msg.get("message", ""),
                    "timestamp": str(msg.get("timestamp", ""))
                })
            return {
                "conversation_id": conv_data.get("conversation_id", "unknown"),
                "messages": messages,
                "metadata": {
                    "protocol": "agora",
                    "original_question": conv_data.get("original_question", "")
                }
            }
        except Exception as e:
            self._log(f"Error standardizing conversation: {e}")
            return None

    def _score_to_grade(self, score: float) -> str:
        """Convert privacy score to letter grade."""
        if score >= 90: return "Excellent"
        elif score >= 80: return "Good"
        elif score >= 70: return "Fair"
        elif score >= 60: return "Poor"
        else: return "Critical"

    def generate_agora_privacy_report(self, conversation_file: str, results: Dict[str, Any] = None) -> str:
        """Generate Agora-specific privacy report."""
        if results is None:
            results = self.analyze_conversations(conversation_file)
        
        # Generate base report
        base_report = self.generate_privacy_report(results)
        
        # Add Agora-specific information
        agora_header = [
            "=" * 80,
            "AGORA PROTOCOL PRIVACY PROTECTION ANALYSIS",
            "=" * 80,
            f"Agora Conversation File: {conversation_file}",
            f"Protocol: Agora Agent Network Protocol",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Extract Agora-specific metrics
        agora_metrics = []
        agora_metrics.append("AGORA PROTOCOL SPECIFICS")
        agora_metrics.append("-" * 40)
        
        total_conversations = results.get("analysis_metadata", {}).get("total_conversations", 0)
        if total_conversations > 0:
            agora_metrics.append(f"Total Agora Conversations Analyzed: {total_conversations}")
            
            # Check for common Agora privacy patterns
            conversation_analyses = results.get("conversation_analyses", [])
            agora_violations = 0
            
            for analysis in conversation_analyses:
                total_violations = sum(
                    len(violations) for violations in analysis.get("violations", {}).values()
                )
                if total_violations > 0:
                    agora_violations += 1
            
            agora_metrics.append(f"Conversations with Privacy Violations: {agora_violations}")
            agora_metrics.append(f"Agora Privacy Compliance Rate: {((total_conversations - agora_violations) / total_conversations * 100):.1f}%")
            
            # Agora-specific features analysis
            agora_metrics.append("")
            agora_metrics.append("AGORA PROTOCOL FEATURES")
            agora_metrics.append("-" * 30)
            agora_metrics.append("- Tool-based agent interaction")
            agora_metrics.append("- LangChain integration support")
            agora_metrics.append("- Structured message format with protocolHash")
            agora_metrics.append("- Protocol source tracking")
        
        agora_metrics.append("")
        
        # Combine reports
        return "\n".join(agora_header + agora_metrics) + "\n" + base_report


def analyze_agora_privacy(conversation_file: str, config: Optional[dict] = None, output_file: str = None) -> Dict[str, Any]:
    """
    Convenience function to analyze Agora conversation privacy.
    
    Args:
        conversation_file: Path to Agora conversation file
        config: Optional analyzer configuration
        output_file: Optional output file for results
        
    Returns:
        Privacy analysis results
    """
    analyzer = AgoraPrivacyAnalyzer(config)
    return analyzer.analyze_conversations(conversation_file)