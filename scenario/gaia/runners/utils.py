"""Evaluation utilities for Gaia benchmark results."""
import json
import re
from typing import Any, Dict, List, Optional


def rouge_l(prediction: str, reference: str) -> float:
    """
    Calculate ROUGE-L score between prediction and reference.
    
    Args:
        prediction: Predicted text
        reference: Reference text
        
    Returns:
        ROUGE-L F1 score
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Tokenize
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    
    # Calculate precision and recall
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer.strip())
    
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove common punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    
    return answer


def extract_numerical_answer(text: str) -> Optional[str]:
    """Extract numerical answer from text."""
    # Look for numbers (including decimals and commas)
    numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
    
    if numbers:
        # Return the first significant number found
        return numbers[0]
    
    return None


def calculate_similarity_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate various similarity metrics."""
    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)
    
    # Exact match
    exact_match = float(pred_norm == ref_norm)
    
    # ROUGE-L
    rouge_score = rouge_l(prediction, reference)
    
    # Word overlap
    pred_words = set(pred_norm.split())
    ref_words = set(ref_norm.split())
    
    if pred_words and ref_words:
        word_overlap = len(pred_words & ref_words) / len(pred_words | ref_words)
    else:
        word_overlap = 0.0
    
    # Numerical match (for numerical answers)
    numerical_match = 0.0
    pred_num = extract_numerical_answer(prediction)
    ref_num = extract_numerical_answer(reference)
    
    if pred_num and ref_num:
        numerical_match = float(pred_num == ref_num)
    
    return {
        "exact_match": exact_match,
        "rouge_l": rouge_score,
        "word_overlap": word_overlap,
        "numerical_match": numerical_match
    }


async def eval_runner(prediction: str, truth_path: str) -> Dict[str, Any]:
    """
    Main evaluation function for Gaia benchmark.
    
    Args:
        prediction: Model's prediction
        truth_path: Path to ground truth file
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        # Load ground truth
        with open(truth_path, encoding='utf-8') as f:
            truth_data = json.load(f)
        
        # Extract reference answer
        if isinstance(truth_data, dict):
            reference = truth_data.get("answer", "")
        elif isinstance(truth_data, list) and truth_data:
            reference = truth_data[0].get("answer", "")
        else:
            reference = str(truth_data)
        
        # Calculate metrics
        metrics = calculate_similarity_metrics(prediction, reference)
        
        # Overall quality score (weighted combination)
        quality_score = (
            metrics["exact_match"] * 0.4 +
            metrics["rouge_l"] * 0.3 +
            metrics["word_overlap"] * 0.2 +
            metrics["numerical_match"] * 0.1
        )
        
        return {
            "quality_score": quality_score,
            **metrics,
            "prediction": prediction[:200] + "..." if len(prediction) > 200 else prediction,
            "reference": reference[:200] + "..." if len(reference) > 200 else reference
        }
        
    except Exception as e:
        return {
            "quality_score": 0.0,
            "exact_match": 0.0,
            "rouge_l": 0.0,
            "word_overlap": 0.0,
            "numerical_match": 0.0,
            "error": str(e),
            "prediction": prediction[:200] + "..." if len(prediction) > 200 else prediction,
            "reference": "Error loading reference"
        }


def evaluate_batch(predictions: List[str], truth_path: str) -> Dict[str, Any]:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: List of predictions
        truth_path: Path to ground truth file
        
    Returns:
        Aggregated evaluation metrics
    """
    import asyncio
    
    async def run_evaluations():
        results = []
        for pred in predictions:
            result = await eval_runner(pred, truth_path)
            results.append(result)
        return results
    
    results = asyncio.run(run_evaluations())
    
    # Aggregate results
    if not results:
        return {"error": "No results to aggregate"}
    
    aggregated = {
        "num_predictions": len(results),
        "avg_quality_score": sum(r.get("quality_score", 0) for r in results) / len(results),
        "avg_exact_match": sum(r.get("exact_match", 0) for r in results) / len(results),
        "avg_rouge_l": sum(r.get("rouge_l", 0) for r in results) / len(results),
        "avg_word_overlap": sum(r.get("word_overlap", 0) for r in results) / len(results),
        "avg_numerical_match": sum(r.get("numerical_match", 0) for r in results) / len(results),
        "individual_results": results
    }
    
    return aggregated


if __name__ == "__main__":
    """Test the evaluation functions."""
    import asyncio
    
    async def test_evaluation():
        # Test cases
        test_cases = [
            {
                "prediction": "The population of Tokyo is approximately 14.2 million people.",
                "reference": "Tokyo has a population of 14.2 million residents.",
                "expected_high_score": True
            },
            {
                "prediction": "I don't know the answer.",
                "reference": "Tokyo has a population of 14.2 million residents.",
                "expected_high_score": False
            },
            {
                "prediction": "Tokyo's population is 37.4 million in the metropolitan area.",
                "reference": "The Tokyo metropolitan area has 37.4 million people.",
                "expected_high_score": True
            }
        ]
        
        print("=== Evaluation Test ===")
        
        for i, case in enumerate(test_cases):
            print(f"\nTest Case {i+1}:")
            print(f"Prediction: {case['prediction']}")
            print(f"Reference: {case['reference']}")
            
            metrics = calculate_similarity_metrics(case['prediction'], case['reference'])
            
            print("Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.3f}")
            
            quality_score = (
                metrics["exact_match"] * 0.4 +
                metrics["rouge_l"] * 0.3 +
                metrics["word_overlap"] * 0.2 +
                metrics["numerical_match"] * 0.1
            )
            
            print(f"  quality_score: {quality_score:.3f}")
            print(f"  Expected high score: {case['expected_high_score']}")
            print(f"  Result: {'✅' if (quality_score > 0.5) == case['expected_high_score'] else '❌'}")
    
    asyncio.run(test_evaluation())
