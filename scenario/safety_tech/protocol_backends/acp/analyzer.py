# -*- coding: utf-8 -*-
"""
ACP Privacy Analyzer - Protocol-specific privacy analysis for ACP conversations
Extends the base privacy analyzer for ACP protocol conversation format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import base privacy analyzer
try:
    from ...core.privacy_analyzer_base import PrivacyAnalyzerBase
except ImportError:
    from core.privacy_analyzer_base import PrivacyAnalyzerBase


class ACPPrivacyAnalyzer(PrivacyAnalyzerBase):
    """ACP-specific privacy analyzer for conversation data."""

    def extract_conversation_data(self, conversation_file: str) -> List[Dict[str, Any]]:
        """
        Extract conversation data from ACP conversation file format.
        
        Args:
            conversation_file: Path to ACP conversation JSON file
            
        Returns:
            List of standardized conversation dictionaries
        """
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                acp_data = json.load(f)
            
            # Handle different ACP file formats
            if isinstance(acp_data, dict):
                if "conversations" in acp_data:
                    # Format: {"conversations": [...], "metadata": {...}}
                    conversations = acp_data["conversations"]
                elif "data" in acp_data:
                    # Format: {"data": [...]}
                    conversations = acp_data["data"]
                else:
                    # Assume single conversation
                    conversations = [acp_data]
            elif isinstance(acp_data, list):
                # Direct list of conversations
                conversations = acp_data
            else:
                raise ValueError("Unsupported ACP conversation file format")
            
            standardized_conversations = []
            
            for i, conv in enumerate(conversations):
                standardized_conv = self._standardize_acp_conversation(conv, i)
                if standardized_conv:
                    standardized_conversations.append(standardized_conv)
            
            self._log(f"Extracted {len(standardized_conversations)} conversations from ACP file")
            return standardized_conversations
            
        except Exception as e:
            self._log(f"Error extracting ACP conversation data: {e}")
            raise

    def _standardize_acp_conversation(self, acp_conversation: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Convert ACP conversation format to standardized format.
        
        Args:
            acp_conversation: ACP-specific conversation data
            index: Conversation index for ID generation
            
        Returns:
            Standardized conversation dictionary or None if invalid
        """
        try:
            # Extract conversation ID
            conv_id = acp_conversation.get("conversation_id", f"acp_conv_{index + 1}")
            
            # Extract messages from different ACP formats
            messages = self._extract_acp_messages(acp_conversation)
            
            if not messages:
                self._log(f"No messages found in conversation {conv_id}")
                return None
            
            # Extract metadata
            metadata = {
                "protocol": "acp",
                "original_question": acp_conversation.get("original_question", ""),
                "timestamp": acp_conversation.get("timestamp", ""),
                "acp_metadata": acp_conversation.get("metadata", {})
            }
            
            return {
                "conversation_id": conv_id,
                "messages": messages,
                "metadata": metadata
            }
            
        except Exception as e:
            self._log(f"Error standardizing ACP conversation {index}: {e}")
            return None

    def _extract_acp_messages(self, acp_conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and standardize messages from ACP conversation format."""
        messages = []
        
        # Try different ACP message formats
        if "messages" in acp_conversation:
            # Format: {"messages": [{"sender": "...", "message": "...", ...}]}
            for msg in acp_conversation["messages"]:
                standardized_msg = self._standardize_acp_message(msg)
                if standardized_msg:
                    messages.append(standardized_msg)
                    
        elif "events" in acp_conversation:
            # Format: {"events": [{"type": "agent_text_message", "data": "...", "agent": "..."}]}
            for event in acp_conversation["events"]:
                if event.get("type") == "agent_text_message":
                    standardized_msg = {
                        "sender": event.get("agent", "unknown"),
                        "content": event.get("data", ""),
                        "timestamp": event.get("timestamp", "")
                    }
                    messages.append(standardized_msg)
                    
        elif "conversation" in acp_conversation:
            # Nested conversation format
            return self._extract_acp_messages(acp_conversation["conversation"])
            
        else:
            # Try to extract from conversation keys directly
            for key in ["original_question", "question", "patient_query"]:
                if key in acp_conversation:
                    messages.append({
                        "sender": "patient",
                        "content": acp_conversation[key],
                        "timestamp": acp_conversation.get("timestamp", "")
                    })
                    break
        
        return messages

    def _standardize_acp_message(self, acp_message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize individual ACP message format."""
        try:
            # Handle different message field names
            content = (
                acp_message.get("message") or 
                acp_message.get("content") or 
                acp_message.get("text") or 
                acp_message.get("data") or 
                str(acp_message)
            )
            
            sender = (
                acp_message.get("sender") or 
                acp_message.get("agent") or 
                acp_message.get("from") or 
                "unknown"
            )
            
            timestamp = (
                acp_message.get("timestamp") or 
                acp_message.get("time") or 
                ""
            )
            
            return {
                "sender": sender,
                "content": content,
                "timestamp": timestamp
            }
            
        except Exception as e:
            self._log(f"Error standardizing ACP message: {e}")
            return None

    def analyze_acp_conversation_file(self, conversation_file: str, output_file: str = None) -> Dict[str, Any]:
        """
        Analyze ACP conversation file and save results.
        
        Args:
            conversation_file: Path to ACP conversation file
            output_file: Optional output file path
            
        Returns:
            Complete analysis results
        """
        # Perform base analysis
        results = self.analyze_conversations(conversation_file)
        
        # Add ACP-specific metadata
        results["analysis_metadata"]["protocol"] = "acp"
        results["analysis_metadata"]["conversation_file"] = conversation_file
        
        # Save results if output file specified
        if output_file:
            self.save_analysis_results(results, output_file)
        
        return results

    def generate_acp_privacy_report(self, conversation_file: str, results: Dict[str, Any] = None) -> str:
        """Generate ACP-specific privacy report."""
        if results is None:
            results = self.analyze_conversations(conversation_file)
        
        # Generate base report
        base_report = self.generate_privacy_report(results)
        
        # Add ACP-specific information
        acp_header = [
            "=" * 80,
            "ACP PROTOCOL PRIVACY PROTECTION ANALYSIS",
            "=" * 80,
            f"ACP Conversation File: {conversation_file}",
            f"Protocol: Agent Communication Protocol (ACP)",
            ""
        ]
        
        # Extract ACP-specific metrics
        acp_metrics = []
        acp_metrics.append("ACP PROTOCOL SPECIFICS")
        acp_metrics.append("-" * 40)
        
        total_conversations = results.get("analysis_metadata", {}).get("total_conversations", 0)
        if total_conversations > 0:
            acp_metrics.append(f"Total ACP Conversations Analyzed: {total_conversations}")
            
            # Check for common ACP privacy patterns
            conversation_analyses = results.get("conversation_analyses", [])
            acp_violations = 0
            
            for analysis in conversation_analyses:
                total_violations = sum(
                    len(violations) for violations in analysis.get("violations", {}).values()
                )
                if total_violations > 0:
                    acp_violations += 1
            
            acp_metrics.append(f"Conversations with Privacy Violations: {acp_violations}")
            acp_metrics.append(f"ACP Privacy Compliance Rate: {((total_conversations - acp_violations) / total_conversations * 100):.1f}%")
        
        acp_metrics.append("")
        
        # Combine reports
        return "\n".join(acp_header + acp_metrics) + "\n" + base_report


def analyze_acp_privacy(conversation_file: str, config: Optional[dict] = None, output_file: str = None) -> Dict[str, Any]:
    """
    Convenience function to analyze ACP conversation privacy.
    
    Args:
        conversation_file: Path to ACP conversation file
        config: Optional analyzer configuration
        output_file: Optional output file for results
        
    Returns:
        Privacy analysis results
    """
    analyzer = ACPPrivacyAnalyzer(config)
    return analyzer.analyze_acp_conversation_file(conversation_file, output_file)

