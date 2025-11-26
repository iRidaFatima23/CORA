import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from transformers import pipeline
from collections import deque
import re

logger = logging.getLogger(__name__)

class SOL:
    """
    Self-Optimization Loops
    Handles metacognitive reasoning and self-evaluation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = deque(maxlen=config.get("history_size", 500))
        self.strategy_performance = {}
        
        # Initialize evaluation components
        self._setup_evaluators()
        
        # Reasoning strategies
        self.strategies = {
            "step_by_step": "Let's think through this step by step:",
            "fact_verification": "I should verify the facts and assumptions:",
            "alternative_perspectives": "What are alternative perspectives or approaches?",
            "confidence_assessment": "How confident am I about this reasoning?",
            "error_detection": "Let me check for potential errors or inconsistencies:"
        }
        
        self.current_strategy = "step_by_step"
        logger.info("SOL initialized successfully")
    
    def _setup_evaluators(self):
        """Setup evaluation pipelines for different aspects"""
        try:
            # Coherence evaluator
            self.coherence_evaluator = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
            
            # Factuality checker (simplified - would be enhanced in practice)
            self.fact_checker = None  # Could integrate with knowledge bases
            
            logger.info("SOL evaluators setup completed")
            
        except Exception as e:
            logger.warning(f"Some evaluators could not be loaded: {str(e)}")
            self.coherence_evaluator = None
    
    def evaluate_response(self, query: str, response: str, ground_truth: Optional[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of model response
        
        Args:
            query: Original input query
            response: Model-generated response
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Dictionary of evaluation scores
        """
        scores = {}
        
        # 1. Coherence evaluation
        coherence_score = self._evaluate_coherence(query, response)
        scores["coherence"] = coherence_score
        
        # 2. Factuality evaluation
        factuality_score = self._evaluate_factuality(response, ground_truth)
        scores["factuality"] = factuality_score
        
        # 3. Completeness evaluation
        completeness_score = self._evaluate_completeness(query, response)
        scores["completeness"] = completeness_score
        
        # 4. Confidence calibration
        confidence_score = self._calibrate_confidence(response)
        scores["confidence"] = confidence_score
        
        # Composite score (weighted average)
        weights = self.config.get("evaluation_weights", {
            "coherence": 0.3,
            "factuality": 0.4,
            "completeness": 0.2,
            "confidence": 0.1
        })
        
        composite_score = sum(scores[metric] * weight 
                            for metric, weight in weights.items())
        scores["composite"] = composite_score
        
        # Update performance history
        self.performance_history.append({
            "query": query,
            "response": response,
            "scores": scores,
            "strategy": self.current_strategy,
            "timestamp": np.datetime64('now')
        })
        
        logger.debug(f"SOL evaluation completed: {composite_score:.3f}")
        return scores
    
    def _evaluate_coherence(self, query: str, response: str) -> float:
        """Evaluate response coherence and relevance"""
        if self.coherence_evaluator is None:
            # Fallback: simple length and keyword matching
            response_lower = response.lower()
            query_keywords = set(query.lower().split())
            response_keywords = set(response_lower.split())
            
            overlap = len(query_keywords.intersection(response_keywords))
            keyword_score = overlap / max(len(query_keywords), 1)
            
            # Length appropriateness (not too short, not too long)
            words = response.split()
            length_score = 1.0 - min(abs(len(words) - 50) / 100, 1.0)  # Ideal around 50 words
            
            return (keyword_score + length_score) / 2
        
        else:
            # Use trained evaluator
            evaluation_text = f"Query: {query}\nResponse: {response}"
            try:
                result = self.coherence_evaluator(evaluation_text[:512])[0]
                score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
                return score
            except:
                return 0.7  # Default fallback
    
    def _evaluate_factuality(self, response: str, ground_truth: Optional[str] = None) -> float:
        """Evaluate factual accuracy"""
        if ground_truth:
            # Simple string similarity with ground truth
            response_lower = response.lower()
            truth_lower = ground_truth.lower()
            
            response_words = set(response_lower.split())
            truth_words = set(truth_lower.split())
            
            overlap = len(response_words.intersection(truth_words))
            return overlap / max(len(truth_words), 1)
        
        else:
            # Heuristic factuality check (simplified)
            # Look for hedging language and uncertainty markers
            uncertainty_indicators = [
                "i think", "probably", "maybe", "perhaps", "might be",
                "could be", "not sure", "i believe", "it seems"
            ]
            
            response_lower = response.lower()
            uncertainty_count = sum(1 for indicator in uncertainty_indicators 
                                  if indicator in response_lower)
            
            # More uncertainty indicators → lower factuality score
            base_score = 0.8
            penalty = min(uncertainty_count * 0.1, 0.5)
            return max(base_score - penalty, 0.3)
    
    def _evaluate_completeness(self, query: str, response: str) -> float:
        """Evaluate how completely the response addresses the query"""
        # Check if response contains different types of information
        question_types = {
            "what": ["is", "are", "was", "were"],
            "how": ["to", "does", "do", "can"],
            "why": ["reason", "because", "therefore", "thus"],
            "when": ["time", "date", "year", "month", "day"],
            "where": ["location", "place", "country", "city"]
        }
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Determine question type from query
        detected_types = []
        for q_type, indicators in question_types.items():
            if any(indicator in query_lower for indicator in [q_type] + indicators):
                detected_types.append(q_type)
        
        # Score based on addressing detected question types
        if not detected_types:
            return 0.7  # Default for non-standard questions
        
        completeness = 0.0
        for q_type in detected_types:
            # Check if response contains relevant information for this question type
            if self._check_completeness_for_type(q_type, response_lower):
                completeness += 1.0
        
        return completeness / len(detected_types)
    
    def _check_completeness_for_type(self, question_type: str, response: str) -> bool:
        """Check if response adequately addresses specific question type"""
        type_requirements = {
            "what": ["is", "are", "means", "refers to", "definition"],
            "how": ["steps", "process", "method", "way to", "by"],
            "why": ["because", "reason", "since", "due to", "caused by"],
            "when": ["in", "on", "during", "year", "time", "date"],
            "where": ["in", "at", "location", "place", "country", "city"]
        }
        
        requirements = type_requirements.get(question_type, [])
        return any(req in response for req in requirements)
    
    def _calibrate_confidence(self, response: str) -> float:
        """Calibrate confidence based on response characteristics"""
        # Analyze language patterns associated with confidence
        high_confidence_indicators = [
            "definitely", "certainly", "without doubt", "clearly",
            "obviously", "undoubtedly", "is always", "is never"
        ]
        
        low_confidence_indicators = [
            "i think", "maybe", "perhaps", "possibly", "might be",
            "could be", "not sure", "i believe", "it seems", "probably"
        ]
        
        response_lower = response.lower()
        
        high_count = sum(1 for indicator in high_confidence_indicators 
                        if indicator in response_lower)
        low_count = sum(1 for indicator in low_confidence_indicators 
                       if indicator in response_lower)
        
        # Base confidence score
        if high_count > 0 and low_count == 0:
            return 0.9
        elif low_count > 0 and high_count == 0:
            return 0.5
        elif high_count > low_count:
            return 0.7
        elif low_count > high_count:
            return 0.6
        else:
            return 0.7  # Neutral
    
    def adapt_strategy(self, current_performance: float) -> str:
        """
        Adapt reasoning strategy based on recent performance
        
        Args:
            current_performance: Latest performance score
            
        Returns:
            New strategy to use
        """
        if len(self.performance_history) < 10:
            return self.current_strategy  # Not enough data
        
        # Calculate recent performance average
        recent_scores = [entry["scores"]["composite"] 
                        for entry in list(self.performance_history)[-10:]]
        recent_avg = np.mean(recent_scores)
        
        # Track strategy performance
        if self.current_strategy not in self.strategy_performance:
            self.strategy_performance[self.current_strategy] = []
        
        self.strategy_performance[self.current_strategy].append(current_performance)
        
        # Switch strategy if performance is poor
        if recent_avg < self.config.get("strategy_switch_threshold", 0.6):
            available_strategies = [s for s in self.strategies 
                                  if s != self.current_strategy]
            
            # Choose strategy with best historical performance
            best_strategy = self.current_strategy
            best_avg = recent_avg
            
            for strategy in available_strategies:
                if strategy in self.strategy_performance:
                    strategy_avg = np.mean(self.strategy_performance[strategy][-5:])
                    if strategy_avg > best_avg:
                        best_strategy = strategy
                        best_avg = strategy_avg
                else:
                    # Try unexplored strategy
                    best_strategy = strategy
                    break
            
            if best_strategy != self.current_strategy:
                logger.info(f"SOL switching strategy from {self.current_strategy} to {best_strategy}")
                self.current_strategy = best_strategy
        
        return self.current_strategy
    
    def generate_reflection(self, query: str, response: str, scores: Dict[str, float]) -> str:
        """Generate metacognitive reflection on the response"""
        reflection_parts = []
        
        if scores["coherence"] < 0.6:
            reflection_parts.append("The response could be more coherent and directly address the query.")
        
        if scores["factuality"] < 0.6:
            reflection_parts.append("I should verify the factual accuracy of this information.")
        
        if scores["completeness"] < 0.6:
            reflection_parts.append("The response might be missing important aspects of the query.")
        
        if not reflection_parts:
            reflection_parts.append("The response appears reasonable based on current evaluation.")
        
        # Add strategy reflection
        reflection_parts.append(f"Using {self.current_strategy.replace('_', ' ')} strategy.")
        
        return " ".join(reflection_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get SOL performance statistics"""
        if not self.performance_history:
            return {}
        
        recent_scores = [entry["scores"]["composite"] 
                        for entry in list(self.performance_history)[-50:]]
        
        return {
            "recent_performance_mean": np.mean(recent_scores),
            "recent_performance_std": np.std(recent_scores),
            "strategy_performance": {k: np.mean(v[-10:]) if v else 0 
                                   for k, v in self.strategy_performance.items()},
            "total_evaluations": len(self.performance_history),
            "current_strategy": self.current_strategy
        }
