"""
Answer Validator for Fail Storm Recovery Scenario

Provides enhanced answer validation and quality assessment for shard QA tasks.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AnswerQuality(Enum):
    """Answer quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class AnswerValidationResult:
    """Result of answer validation."""
    is_valid: bool
    quality: AnswerQuality
    confidence: float  # 0.0 to 1.0
    reasoning: str
    answer_text: str
    source_info: Dict[str, Any]
    validation_time: float


class FailStormAnswerValidator:
    """
    Enhanced answer validator for fail storm recovery scenario.
    
    Provides multiple validation strategies:
    1. LLM-based validation for quality assessment
    2. Rule-based validation for basic checks
    3. Consensus validation when multiple answers are available
    """
    
    def __init__(self, llm_client=None, validation_timeout: float = 10.0):
        """
        Initialize answer validator.
        
        Args:
            llm_client: LLM client for quality validation
            validation_timeout: Timeout for validation operations
        """
        self.llm_client = llm_client
        self.validation_timeout = validation_timeout
        self.validation_history: List[Dict[str, Any]] = []
        
    def set_llm_client(self, llm_client):
        """Set LLM client for validation."""
        self.llm_client = llm_client
        print("🧠 LLM client configured for answer validation")
    
    async def validate_answer(self, question: str, answer: str, 
                            source_context: str = "", 
                            worker_id: str = "") -> AnswerValidationResult:
        """
        Validate an answer with comprehensive quality assessment.
        
        Args:
            question: Original question
            answer: Answer to validate
            source_context: Context from source document
            worker_id: ID of worker that found the answer
            
        Returns:
            AnswerValidationResult with validation details
        """
        start_time = time.time()
        
        try:
            # Basic validation first
            basic_result = self._basic_validation(question, answer)
            if not basic_result.is_valid:
                return basic_result
            
            # LLM validation if available
            if self.llm_client:
                llm_result = await self._llm_validation(question, answer, source_context)
                if llm_result:
                    llm_result.source_info.update({
                        "worker_id": worker_id,
                        "validation_method": "llm",
                        "basic_checks_passed": True
                    })
                    llm_result.validation_time = time.time() - start_time
                    self._record_validation(llm_result)
                    return llm_result
            
            # Fallback to enhanced basic validation
            enhanced_result = self._enhanced_basic_validation(question, answer, source_context)
            enhanced_result.source_info.update({
                "worker_id": worker_id,
                "validation_method": "enhanced_basic"
            })
            enhanced_result.validation_time = time.time() - start_time
            self._record_validation(enhanced_result)
            return enhanced_result
            
        except Exception as e:
            return AnswerValidationResult(
                is_valid=False,
                quality=AnswerQuality.INVALID,
                confidence=0.0,
                reasoning=f"Validation error: {str(e)}",
                answer_text=answer,
                source_info={"error": str(e), "worker_id": worker_id},
                validation_time=time.time() - start_time
            )
    
    def _basic_validation(self, question: str, answer: str) -> AnswerValidationResult:
        """Basic validation checks."""
        if not answer or not answer.strip():
            return AnswerValidationResult(
                is_valid=False,
                quality=AnswerQuality.INVALID,
                confidence=0.0,
                reasoning="Empty or null answer",
                answer_text=answer,
                source_info={"validation_method": "basic"},
                validation_time=0.0
            )
        
        # Check minimum length
        if len(answer.strip()) < 3:
            return AnswerValidationResult(
                is_valid=False,
                quality=AnswerQuality.POOR,
                confidence=0.2,
                reasoning="Answer too short to be meaningful",
                answer_text=answer,
                source_info={"validation_method": "basic"},
                validation_time=0.0
            )
        
        # Check for obvious error patterns
        error_patterns = ["error", "failed", "unknown", "not found", "no information"]
        answer_lower = answer.lower()
        if any(pattern in answer_lower for pattern in error_patterns):
            return AnswerValidationResult(
                is_valid=False,
                quality=AnswerQuality.POOR,
                confidence=0.3,
                reasoning="Answer contains error indicators",
                answer_text=answer,
                source_info={"validation_method": "basic"},
                validation_time=0.0
            )
        
        # Basic validation passed
        return AnswerValidationResult(
            is_valid=True,
            quality=AnswerQuality.FAIR,
            confidence=0.6,
            reasoning="Basic validation passed",
            answer_text=answer,
            source_info={"validation_method": "basic"},
            validation_time=0.0
        )
    
    async def _llm_validation(self, question: str, answer: str, 
                            source_context: str) -> Optional[AnswerValidationResult]:
        """LLM-based answer validation."""
        try:
            system_prompt = """You are an expert answer validator for collaborative retrieval systems.
Your job is to assess whether an answer correctly and completely addresses the given question.

Evaluation criteria:
1. Factual accuracy
2. Completeness of response
3. Relevance to the question
4. Quality of reasoning

Respond with a JSON object containing your assessment."""

            user_prompt = f"""
QUESTION: {question}

PROPOSED ANSWER: {answer}

SOURCE CONTEXT: {source_context}

Please evaluate this answer and respond with JSON:
{{
    "is_valid": true/false,
    "quality": "excellent/good/fair/poor/invalid",
    "confidence": 0.0-1.0,
    "reasoning": "Detailed explanation of your assessment",
    "specific_issues": ["list", "of", "any", "issues"],
    "completeness_score": 0.0-1.0
}}
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use asyncio timeout for validation
            response = await asyncio.wait_for(
                self.llm_client.ask(messages=messages),
                timeout=self.validation_timeout
            )
            
            # Parse LLM response
            validation_data = self._parse_llm_validation_response(response)
            if validation_data:
                quality_mapping = {
                    "excellent": AnswerQuality.EXCELLENT,
                    "good": AnswerQuality.GOOD,
                    "fair": AnswerQuality.FAIR,
                    "poor": AnswerQuality.POOR,
                    "invalid": AnswerQuality.INVALID
                }
                
                return AnswerValidationResult(
                    is_valid=validation_data.get("is_valid", False),
                    quality=quality_mapping.get(validation_data.get("quality", "fair"), AnswerQuality.FAIR),
                    confidence=validation_data.get("confidence", 0.5),
                    reasoning=validation_data.get("reasoning", "LLM validation completed"),
                    answer_text=answer,
                    source_info={
                        "validation_method": "llm",
                        "completeness_score": validation_data.get("completeness_score", 0.5),
                        "specific_issues": validation_data.get("specific_issues", [])
                    },
                    validation_time=0.0  # Will be set by caller
                )
            
        except asyncio.TimeoutError:
            print(f"⚠️ LLM validation timed out after {self.validation_timeout}s")
        except Exception as e:
            print(f"⚠️ LLM validation failed: {e}")
        
        return None
    
    def _parse_llm_validation_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM validation response."""
        try:
            # Try to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse LLM validation response: {e}")
        
        return None
    
    def _enhanced_basic_validation(self, question: str, answer: str, 
                                 source_context: str) -> AnswerValidationResult:
        """Enhanced basic validation with heuristics."""
        # Start with basic validation
        basic_result = self._basic_validation(question, answer)
        if not basic_result.is_valid:
            return basic_result
        
        # Enhanced checks
        confidence = 0.6  # Base confidence
        quality = AnswerQuality.FAIR
        reasoning_parts = ["Enhanced basic validation"]
        
        # Length-based quality assessment
        answer_length = len(answer.strip())
        if answer_length > 100:
            confidence += 0.1
            quality = AnswerQuality.GOOD
            reasoning_parts.append("substantial answer length")
        elif answer_length < 20:
            confidence -= 0.1
            reasoning_parts.append("short answer")
        
        # Question-answer relevance (simple keyword matching)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words.intersection(answer_words))
        
        if overlap > 2:
            confidence += 0.1
            reasoning_parts.append("good keyword overlap with question")
        elif overlap == 0:
            confidence -= 0.2
            reasoning_parts.append("no keyword overlap with question")
        
        # Source context relevance
        if source_context and len(source_context) > 50:
            context_words = set(source_context.lower().split())
            context_overlap = len(answer_words.intersection(context_words))
            if context_overlap > 3:
                confidence += 0.1
                reasoning_parts.append("good context alignment")
        
        # Adjust quality based on confidence
        if confidence >= 0.8:
            quality = AnswerQuality.EXCELLENT
        elif confidence >= 0.7:
            quality = AnswerQuality.GOOD
        elif confidence < 0.4:
            quality = AnswerQuality.POOR
        
        return AnswerValidationResult(
            is_valid=confidence >= 0.4,
            quality=quality,
            confidence=min(1.0, max(0.0, confidence)),
            reasoning=" | ".join(reasoning_parts),
            answer_text=answer,
            source_info={"validation_method": "enhanced_basic"},
            validation_time=0.0
        )
    
    def validate_multiple_answers(self, question: str, 
                                answers: List[Tuple[str, str, str]]) -> AnswerValidationResult:
        """
        Validate multiple answers and select the best one.
        
        Args:
            question: Original question
            answers: List of (answer, source_context, worker_id) tuples
            
        Returns:
            Best validated answer
        """
        if not answers:
            return AnswerValidationResult(
                is_valid=False,
                quality=AnswerQuality.INVALID,
                confidence=0.0,
                reasoning="No answers provided",
                answer_text="",
                source_info={"validation_method": "consensus"},
                validation_time=0.0
            )
        
        # If only one answer, validate it normally
        if len(answers) == 1:
            answer, context, worker_id = answers[0]
            result = self._enhanced_basic_validation(question, answer, context)
            result.source_info.update({"worker_id": worker_id, "consensus_size": 1})
            return result
        
        # Multiple answers - find consensus or best quality
        best_result = None
        best_confidence = 0.0
        
        for answer, context, worker_id in answers:
            result = self._enhanced_basic_validation(question, answer, context)
            result.source_info.update({"worker_id": worker_id})
            
            if result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result
        
        if best_result:
            best_result.source_info.update({
                "validation_method": "consensus",
                "consensus_size": len(answers),
                "selected_from": len(answers)
            })
            best_result.reasoning += f" | Selected best from {len(answers)} answers"
        
        return best_result or AnswerValidationResult(
            is_valid=False,
            quality=AnswerQuality.POOR,
            confidence=0.0,
            reasoning="All answers failed validation",
            answer_text="",
            source_info={"validation_method": "consensus", "consensus_size": len(answers)},
            validation_time=0.0
        )
    
    def _record_validation(self, result: AnswerValidationResult):
        """Record validation result for analysis."""
        record = {
            "timestamp": time.time(),
            "is_valid": result.is_valid,
            "quality": result.quality.value,
            "confidence": result.confidence,
            "answer_length": len(result.answer_text),
            "validation_time": result.validation_time,
            "validation_method": result.source_info.get("validation_method", "unknown")
        }
        self.validation_history.append(record)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        if not self.validation_history:
            return {"message": "No validation history available"}
        
        total = len(self.validation_history)
        valid_count = sum(1 for r in self.validation_history if r["is_valid"])
        
        quality_counts = {}
        for r in self.validation_history:
            quality = r["quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        avg_confidence = sum(r["confidence"] for r in self.validation_history) / total
        avg_validation_time = sum(r["validation_time"] for r in self.validation_history) / total
        
        return {
            "total_validations": total,
            "valid_answers": valid_count,
            "validation_rate": valid_count / total,
            "quality_distribution": quality_counts,
            "average_confidence": avg_confidence,
            "average_validation_time": avg_validation_time,
            "validation_methods": list(set(r["validation_method"] for r in self.validation_history))
        }


# Singleton instance for global use
answer_validator = FailStormAnswerValidator()


# Utility functions
async def validate_shard_answer(question: str, answer: str, 
                              source_context: str = "", 
                              worker_id: str = "",
                              llm_client=None) -> AnswerValidationResult:
    """Convenience function for validating shard answers."""
    if llm_client and not answer_validator.llm_client:
        answer_validator.set_llm_client(llm_client)
    
    return await answer_validator.validate_answer(question, answer, source_context, worker_id)


def validate_multiple_shard_answers(question: str, 
                                   answers: List[Tuple[str, str, str]]) -> AnswerValidationResult:
    """Convenience function for validating multiple answers."""
    return answer_validator.validate_multiple_answers(question, answers)


def get_validation_stats() -> Dict[str, Any]:
    """Convenience function for getting validation statistics."""
    return answer_validator.get_validation_statistics()
