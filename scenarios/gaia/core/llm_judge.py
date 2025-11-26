"""Enhanced LLM as Judge module for GAIA framework with timeout handling."""
import asyncio
import json
import time
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .llm import call_llm, LLM


class JudgeResult(Enum):
    """Possible judgment results."""
    CORRECT = "correct"
    INCORRECT = "incorrect" 
    TIMEOUT = "timeout"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class JudgmentOutput:
    """Structured output from LLM judge."""
    result: JudgeResult
    quality_score: int  # 1 (poor) .. 5 (excellent)
    reasoning: str
    answer_quality: str  # excellent/good/fair/poor
    final_answer_present: bool
    execution_time: float
    judge_used: str  # llm/fallback/timeout
    error_message: Optional[str] = None


class EnhancedLLMJudge:
    """Enhanced LLM as Judge with robust timeout and error handling."""
    
    def __init__(self, config_path: Optional[str] = None, judge_timeout: int = 30, 
                 judge_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced LLM judge.
        
        Args:
            config_path: Path to LLM configuration (deprecated, use judge_config instead)
            judge_timeout: Timeout for judge evaluation in seconds
            judge_config: Judge-specific LLM configuration (model, base_url, temperature, etc.)
        """
        self.judge_timeout = judge_timeout
        self.config_path = config_path
        self.judge_config = judge_config
        self._llm_instance = None
        
    def _get_llm(self) -> LLM:
        """Get or create LLM instance for judge using judge-specific config."""
        if self._llm_instance is None:
            if self.judge_config:
                # Use judge-specific configuration
                from .llm import LLMConfig
                import tempfile
                import yaml
                
                # Create temporary config file with judge settings
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump({'model': self.judge_config}, f)
                    temp_config_path = f.name
                
                self._llm_instance = call_llm(config_path=temp_config_path)
            else:
                # Fall back to provided config_path or default
                self._llm_instance = call_llm(config_path=self.config_path)
        return self._llm_instance
        
    def _create_judge_prompt(self, question: str, ground_truth: str, 
                           predicted_answer: str, final_answer: str, network_log_path: Optional[str] = None) -> str:
        """Create comprehensive judge prompt with a process-oriented quality rubric.

        If `network_log_path` is provided and readable, the prompt will include the
        entire contents of the network execution log (JSON) so the judge evaluates
        inter-agent communication, tool inputs/outputs, and message content in detail.
        """
        # Try to read network execution log if provided
        network_log_content = ""
        if network_log_path:
            try:
                p = Path(network_log_path)
                if p.exists():
                    network_log_content = p.read_text(encoding='utf-8')
                else:
                    network_log_content = f"<NETWORK LOG NOT FOUND at {network_log_path}>"
            except Exception as e:
                network_log_content = f"<ERROR READING NETWORK LOG: {e}>"

        return f"""You are an expert judge evaluating AI system responses for the GAIA benchmark. Your evaluation must consider both the final answer's correctness and the quality of the process taken by the AI.

**TASK DETAILS:**
- **ORIGINAL QUESTION:** {question}
- **GROUND TRUTH ANSWER:** {ground_truth}

- **EXTRACTED FINAL ANSWER:** {final_answer}

- **FULL AI SYSTEM RESPONSE (TRACE) (Brief summary / final output):**
{predicted_answer}

---
IMPORTANT: When assessing the agent, PRIORITIZE the FULL NETWORK EXECUTION LOG (JSON) below if provided. This log contains all inter-agent messages, tool calls, and intermediate data exchanges. Your process-quality judgment MUST be based primarily on the content, clarity, correctness, and completeness of inter-agent communication and tool interactions recorded in the network execution log. Do NOT rely only on any short summary or the extracted final answer.

**FULL NETWORK EXECUTION LOG (JSON):**
{network_log_content}

If the network log is unavailable, fall back to using the FULL AI SYSTEM RESPONSE (TRACE) above.

**EVALUATION INSTRUCTIONS:**

Your task is to perform a two-part evaluation:
1.  **Correctness (`is_correct`):** First, determine if the `EXTRACTED FINAL ANSWER` is correct when compared to the `GROUND TRUTH ANSWER`. Consider semantic equivalence and allow for minor formatting differences.
2.  **Process Quality (`quality_score`):** Second, and just as importantly, evaluate the agent's problem-solving process based on the FULL NETWORK EXECUTION LOG (preferred) or the `FULL AI SYSTEM RESPONSE (TRACE)` when the log is unavailable. Use the detailed rubric below to assign a score from 1 to 5.

---
**QUALITY SCORE RUBRIC (1-5):**

Your primary focus for the `quality_score` is the agent's methodology and the quality of inter-agent communication. A high score can be given for a good process even if the final answer is incorrect.

- **Score 5 (Excellent):**
  - The final answer is correct.
  - Inter-agent communication is clear, complete, and correct. Tools are used correctly and efficiently. Intermediate results are validated and shared appropriately.

- **Score 4 (Good):**
  - The final answer is correct, but the process was sub-optimal or the reasoning is not solid in a minor way.

- **Score 3 (Fair / Good Process):**
  - Solid reasoning and reasonable communication,  but the final answer is incorrect due to either a late-stage reasoning error or an unrecoverable tool failure.

- **Score 2 (Poor):**
  - Communication is incomplete or incorrect, tools are misused, or agents fail to share necessary details.

- **Score 1 (Very Poor):**
  - No meaningful communication, hallucinated tool use, or completely irrelevant traces.

---
**RESPONSE FORMAT:**

Respond with a single JSON object. Do not include any other text or explanations outside the JSON.
{{
  "is_correct": true/false,
  "quality_score": 1-5,
  "reasoning": "Detailed explanation for your judgment. Justify BOTH the correctness of the final answer and the quality score based on the process trace and the rubric.",
  "answer_quality": "excellent/good/fair/poor",
  "final_answer_present": true/false,
  "partial_credit": 0.0-1.0
}}

Be thorough but fair in your evaluation. Provide specific reasoning for your judgment.
"""

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with robust parsing."""
        try:
            # Try to find JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            
        # Try alternative parsing - look for key patterns
        try:
            import re
            
            # Extract key-value pairs using regex
            patterns = {
                "is_correct": r'"is_correct":\s*(true|false)',
                "quality_score": r'"quality_score":\s*([0-9]+)',
                "quality": r'"quality":\s*([0-9]+)',
                "reasoning": r'"reasoning":\s*"([^"]*)"',
                "answer_quality": r'"answer_quality":\s*"([^"]*)"',
                "final_answer_present": r'"final_answer_present":\s*(true|false)',
                "partial_credit": r'"partial_credit":\s*([0-9.]+)'
            }
            
            extracted = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if key in ["is_correct", "final_answer_present"]:
                        extracted[key] = value.lower() == "true"
                    elif key in ["quality_score", "quality"]:
                        try:
                            extracted["quality_score"] = int(value)
                        except Exception:
                            extracted["quality_score"] = None
                    elif key in ["partial_credit"]:
                        extracted[key] = float(value)
                    else:
                        extracted[key] = value
                        
            if extracted:
                return extracted
                
        except Exception as e:
            print(f"Alternative parsing error: {e}")
            
        return None

    def _fallback_judgment(self, final_answer: str, ground_truth: str, 
                         error_msg: str = "") -> JudgmentOutput:
        """Provide fallback judgment when LLM judge fails."""
        # Simple string comparison fallback
        final_clean = final_answer.lower().strip()
        truth_clean = ground_truth.lower().strip()
        
        # Exact match
        if final_clean == truth_clean:
            return JudgmentOutput(
                result=JudgeResult.CORRECT,
                quality_score=5,  # Excellent for exact match
                reasoning=f"Exact string match between final answer and ground truth. {error_msg}",
                answer_quality="good",
                final_answer_present=bool(final_answer.strip()),
                execution_time=0.0,
                judge_used="fallback",
                error_message=error_msg if error_msg else None
            )
        
        # Check if final answer contains ground truth or vice versa
        contains_match = (final_clean in truth_clean and len(final_clean) > 3) or \
                        (truth_clean in final_clean and len(truth_clean) > 3)
        
        if contains_match:
            return JudgmentOutput(
                result=JudgeResult.PARTIAL,
                quality_score=3,
                reasoning=f"Partial match detected between answers. {error_msg}",
                answer_quality="fair", 
                final_answer_present=bool(final_answer.strip()),
                execution_time=0.0,
                judge_used="fallback",
                error_message=error_msg if error_msg else None
            )
        
        # No match
        return JudgmentOutput(
            result=JudgeResult.INCORRECT,
            quality_score=1,
            reasoning=f"No match found between final answer and ground truth. {error_msg}",
            answer_quality="poor",
            final_answer_present=bool(final_answer.strip()),
            execution_time=0.0,
            judge_used="fallback",
            error_message=error_msg if error_msg else None
        )

    def _handle_timeout_case(self, question: str, ground_truth: str, 
                           execution_time: float) -> JudgmentOutput:
        """Handle timeout cases with appropriate judgment."""
        return JudgmentOutput(
            result=JudgeResult.TIMEOUT,
            quality_score=1,
            reasoning=f"Task execution timed out after {execution_time:.1f} seconds. "
                     "Unable to generate a complete response for evaluation.",
            answer_quality="poor",
            final_answer_present=False,
            execution_time=execution_time,
            judge_used="timeout",
            error_message="Execution timeout"
        )

    async def judge_answer(self, question: str, ground_truth: str, 
                          predicted_answer: str, execution_time: float,
                          status: str = "success", network_log_path: Optional[str] = None) -> JudgmentOutput:
        """
        Judge an answer with comprehensive error and timeout handling.
        
        Args:
            question: Original question
            ground_truth: Ground truth answer
            predicted_answer: AI system's predicted answer
            execution_time: Time taken for execution
            status: Execution status (success/timeout/error)
            network_log_path: Optional path to network execution log (JSON)
            
        Returns:
            JudgmentOutput with comprehensive evaluation
        """
        start_time = time.time()
        
        # Handle timeout cases
        if status == "timeout":
            return self._handle_timeout_case(question, ground_truth, execution_time)
        
        # Handle error cases
        if status in ["error", "execution_error"]:
            return JudgmentOutput(
                result=JudgeResult.ERROR,
                quality_score=1,
                reasoning=f"Task execution failed with error. Predicted answer: {predicted_answer[:100]}...",
                answer_quality="poor",
                final_answer_present=False,
                execution_time=execution_time,
                judge_used="error",
                error_message="Execution error"
            )
        
        # Extract final answer
        final_answer = predicted_answer
        if "FINAL ANSWER:" in predicted_answer:
            final_answer = predicted_answer.split("FINAL ANSWER:")[-1].strip()
        elif "Final Answer:" in predicted_answer:
            final_answer = predicted_answer.split("Final Answer:")[-1].strip()
        
        # Create judge prompt
        judge_prompt = self._create_judge_prompt(question, ground_truth, 
                                                predicted_answer, final_answer, network_log_path=network_log_path)
        
        try:
            # Get LLM judge response with timeout
            llm = self._get_llm()
            # Add a strong system instruction to force focus on the full trace
            judge_messages = [
                {"role": "system", "content": "You are a strict, process-focused judge. Base your evaluation on the FULL AI SYSTEM RESPONSE (TRACE) provided and output ONLY the required JSON object. Do not add any commentary or extra text."},
                {"role": "user", "content": judge_prompt}
            ]

            # Use asyncio timeout for judge evaluation
            judge_response = await asyncio.wait_for(
                llm.ask(messages=judge_messages, temperature=0.0),
                timeout=self.judge_timeout
            )
            
            # Parse judge response
            judge_data = self._extract_json_from_response(judge_response)
            judge_time = time.time() - start_time
            
            if judge_data:
                # Successfully parsed LLM judge response
                is_correct = judge_data.get("is_correct", False)
                partial_credit = judge_data.get("partial_credit", 0.0)
                quality_score = judge_data.get("quality_score", None)
                if quality_score is None:
                    # Try legacy 'quality' key
                    quality_score = judge_data.get("quality", 3)
                
                # Normalize quality_score
                try:
                    quality_score = int(quality_score)
                    if quality_score < 1:
                        quality_score = 1
                    if quality_score > 5:
                        quality_score = 5
                except Exception:
                    quality_score = 3
                
                # Determine result based on correctness and partial credit
                if is_correct:
                    result = JudgeResult.CORRECT
                elif partial_credit > 0.5:
                    result = JudgeResult.PARTIAL
                else:
                    result = JudgeResult.INCORRECT
                
                return JudgmentOutput(
                    result=result,
                    quality_score=quality_score,
                    reasoning=judge_data.get("reasoning", "LLM judge evaluation completed"),
                    answer_quality=judge_data.get("answer_quality", "unknown"),
                    final_answer_present=judge_data.get("final_answer_present", bool(final_answer.strip())),
                    execution_time=judge_time,
                    judge_used="llm"
                )
            else:
                # Failed to parse JSON, use fallback
                return self._fallback_judgment(
                    final_answer, ground_truth, 
                    "LLM judge response could not be parsed as JSON"
                )
                
        except asyncio.TimeoutError:
            # Judge evaluation timed out
            return self._fallback_judgment(
                final_answer, ground_truth,
                f"LLM judge evaluation timed out after {self.judge_timeout}s"
            )
            
        except Exception as e:
            # Other errors during judge evaluation
            return self._fallback_judgment(
                final_answer, ground_truth,
                f"LLM judge evaluation failed: {str(e)}"
            )

    def format_judgment_for_result(self, judgment: JudgmentOutput, 
                                 task_id: str, question: str, ground_truth: str,
                                 predicted_answer: str, execution_time: float,
                                 level: int) -> Dict[str, Any]:
        """Format judgment output for final result structure."""
        # Extract final answer for display
        final_answer = predicted_answer
        if "FINAL ANSWER:" in predicted_answer:
            final_answer = predicted_answer.split("FINAL ANSWER:")[-1].strip()
        elif "Final Answer:" in predicted_answer:
            final_answer = predicted_answer.split("Final Answer:")[-1].strip()
            
        # Determine status based on judgment result
        status_mapping = {
            JudgeResult.CORRECT: "success",
            JudgeResult.PARTIAL: "partial_success", 
            JudgeResult.INCORRECT: "failed",
            JudgeResult.TIMEOUT: "timeout",
            JudgeResult.ERROR: "error"
        }
        
        return {
            "task_id": task_id,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "final_answer_extracted": final_answer,
            "execution_time": execution_time,
            "status": status_mapping[judgment.result],
            "level": level,
            "enhanced_llm_judge": {
                "result": judgment.result.value,
                "is_correct": judgment.result in [JudgeResult.CORRECT, JudgeResult.PARTIAL],
                "quality_score": judgment.quality_score,
                "reasoning": judgment.reasoning,
                "answer_quality": judgment.answer_quality,
                "final_answer_present": judgment.final_answer_present,
                "judge_execution_time": judgment.execution_time,
                "judge_method": judgment.judge_used,
                "error_message": judgment.error_message
            }
        }


# Utility function to create judge instance
def create_llm_judge(config_path: Optional[str] = None, 
                    judge_timeout: int = 30,
                    judge_config: Optional[Dict[str, Any]] = None) -> EnhancedLLMJudge:
    """Create an enhanced LLM judge instance.
    
    Args:
        config_path: Path to LLM configuration (deprecated)
        judge_timeout: Timeout for judge evaluation
        judge_config: Judge-specific configuration dict with keys:
            - name: model name (e.g., "qwen-vl-72b-instruct")
            - base_url: API endpoint (e.g., "http://localhost:8000/v1")
            - temperature: sampling temperature (default 0.0)
            - api_key: API key if required
    """
    return EnhancedLLMJudge(config_path=config_path, judge_timeout=judge_timeout, judge_config=judge_config)


# Test function
async def test_llm_judge():
    """Test the enhanced LLM judge."""
    print("=== Enhanced LLM Judge Test ===")
    
    judge = create_llm_judge(judge_timeout=10)
    
    # Test case 1: Correct answer
    print("\n--- Test 1: Correct Answer ---")
    result1 = await judge.judge_answer(
        question="What is the capital of France?",
        ground_truth="Paris",
        predicted_answer="The capital of France is Paris.",
        execution_time=2.5
    )
    print(f"Result: {result1.result.value}")
    print(f"Quality score: {result1.quality_score}")
    print(f"Reasoning: {result1.reasoning[:100]}...")
    
    # Test case 2: Timeout case
    print("\n--- Test 2: Timeout Case ---")
    result2 = await judge.judge_answer(
        question="Solve this complex equation: x^3 + 2x^2 - x - 2 = 0",
        ground_truth="x = 1, x = -1, x = -2",
        predicted_answer="TIMEOUT: Framework execution exceeded time limit",
        execution_time=30.0,
        status="timeout"
    )
    print(f"Result: {result2.result.value}")
    print(f"Judge used: {result2.judge_used}")
    print(f"Reasoning: {result2.reasoning}")
    
    # Test case 3: Error case
    print("\n--- Test 3: Error Case ---")
    result3 = await judge.judge_answer(
        question="What is 2+2?",
        ground_truth="4",
        predicted_answer="ERROR: Division by zero",
        execution_time=0.5,
        status="error"
    )
    print(f"Result: {result3.result.value}")
    print(f"Judge used: {result3.judge_used}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    """Run LLM judge tests."""
    asyncio.run(test_llm_judge())
