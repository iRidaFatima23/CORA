import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
import time
import json

from .ckal import CKAL
from .sol import SOL
from .macl import MACL

logger = logging.getLogger(__name__)

class CORA:
    """
    Main CORA engine that integrates CKAL, SOL, and MACL components
    """
    
    def __init__(self, base_model, config: Dict[str, Any]):
        self.config = config
        self.base_model = base_model
        self.tokenizer = base_model.tokenizer
        
        # Initialize core components
        self.ckal = CKAL(base_model, config.get("ckal", {}))
        self.sol = SOL(config.get("sol", {}))
        self.macl = MACL(base_model, config.get("macl", {}))
        
        # System state
        self.interaction_count = 0
        self.performance_history = []
        self.system_metrics = {
            "total_processing_time": 0,
            "average_confidence": 0,
            "component_usage": {"ckal": 0, "sol": 0, "macl": 0}
        }
        
        logger.info("CORA engine initialized successfully")
    
    @classmethod
    def from_pretrained(cls, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Create CORA instance from pretrained model
        
        Args:
            model_name: HuggingFace model identifier
            config: Optional configuration overrides
            
        Returns:
            CORA instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Add tokenizer to model for convenience
        model.tokenizer = tokenizer
        
        # Default configuration
        default_config = {
            "model_name": model_name,
            "ckal": {
                "memory_size": 1000,
                "learning_rate": 1e-4,
                "lora_r": 16,
                "lora_alpha": 32
            },
            "sol": {
                "history_size": 500,
                "evaluation_weights": {
                    "coherence": 0.3,
                    "factuality": 0.4,
                    "completeness": 0.2,
                    "confidence": 0.1
                }
            },
            "macl": {
                "coordination_strategy": "weighted_consensus",
                "agent_timeout": 30
            }
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        return cls(model, default_config)
    
    def process(self, query: str, ground_truth: Optional[str] = None) -> Tuple[str, float]:
        """
        Process query through full CORA pipeline
        
        Args:
            query: Input query to process
            ground_truth: Optional ground truth for learning
            
        Returns:
            Tuple of (response, confidence_score)
        """
        start_time = time.time()
        self.interaction_count += 1
        
        try:
            # Phase 1: Multi-Agent Processing (MACL)
            macl_start = time.time()
            macl_result = self.macl.process_query(query)
            macl_time = time.time() - macl_start
            self.system_metrics["component_usage"]["macl"] += 1
            
            response = macl_result["final_response"]
            
            # Phase 2: Self-Evaluation (SOL)
            sol_start = time.time()
            evaluation_scores = self.sol.evaluate_response(query, response, ground_truth)
            sol_time = time.time() - sol_start
            self.system_metrics["component_usage"]["sol"] += 1
            
            confidence = evaluation_scores["composite"]
            
            # Phase 3: Continuous Learning (CKAL)
            if confidence < 0.8:  # Learn more from uncertain responses
                ckal_start = time.time()
                experience = {
                    "input": query,
                    "output": response,
                    "feedback": confidence,
                    "ground_truth": ground_truth,
                    "evaluation_scores": evaluation_scores,
                    "timestamp": time.time()
                }
                
                self.ckal.consolidate_experience(experience)
                ckal_time = time.time() - ckal_start
                self.system_metrics["component_usage"]["ckal"] += 1
            else:
                ckal_time = 0
            
            # Phase 4: Strategy Adaptation (SOL)
            self.sol.adapt_strategy(confidence)
            
            # Update system metrics
            total_time = time.time() - start_time
            self.system_metrics["total_processing_time"] += total_time
            
            # Update performance history
            performance_entry = {
                "interaction_id": self.interaction_count,
                "query": query,
                "response": response,
                "confidence": confidence,
                "evaluation_scores": evaluation_scores,
                "processing_time": total_time,
                "component_times": {
                    "macl": macl_time,
                    "sol": sol_time,
                    "ckal": ckal_time
                },
                "timestamp": time.time()
            }
            self.performance_history.append(performance_entry)
            
            # Update average confidence
            recent_confidences = [entry["confidence"] for entry in self.performance_history[-50:]]
            self.system_metrics["average_confidence"] = sum(recent_confidences) / len(recent_confidences)
            
            logger.info(f"CORA processing completed in {total_time:.2f}s with confidence {confidence:.3f}")
            
            return response, confidence
            
        except Exception as e:
            logger.error(f"Error in CORA processing: {str(e)}")
            # Fallback to base model
            inputs = self.tokenizer(query, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self.base_model.generate(**inputs, max_length=512)
            fallback_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return fallback_response, 0.5  # Default confidence for errors
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_metrics": self.system_metrics.copy(),
            "interaction_count": self.interaction_count,
            "average_confidence": self.system_metrics["average_confidence"],
            "components": {
                "ckal": self.ckal.get_memory_stats(),
                "sol": self.sol.get_performance_stats(),
                "macl": self.macl.get_coordination_stats()
            }
        }
        
        # Calculate component usage percentages
        total_usage = sum(self.system_metrics["component_usage"].values())
        if total_usage > 0:
            for component in status["system_metrics"]["component_usage"]:
                status["system_metrics"]["component_usage"][component] /= total_usage
        
        return status
    
    def save_checkpoint(self, directory: str):
        """Save complete CORA state to checkpoint"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save component states
        self.ckal.save_checkpoint(f"{directory}/ckal.pt")
        
        # Save system state
        system_state = {
            "interaction_count": self.interaction_count,
            "performance_history": self.performance_history,
            "system_metrics": self.system_metrics,
            "config": self.config
        }
        
        with open(f"{directory}/system_state.json", "w") as f:
            json.dump(system_state, f, indent=2)
        
        logger.info(f"CORA checkpoint saved to {directory}")
    
    def load_checkpoint(self, directory: str):
        """Load CORA state from checkpoint"""
        import os
        
        # Load component states
        if os.path.exists(f"{directory}/ckal.pt"):
            self.ckal.load_checkpoint(f"{directory}/ckal.pt")
        
        # Load system state
        if os.path.exists(f"{directory}/system_state.json"):
            with open(f"{directory}/system_state.json", "r") as f:
                system_state = json.load(f)
            
            self.interaction_count = system_state["interaction_count"]
            self.performance_history = system_state["performance_history"]
            self.system_metrics = system_state["system_metrics"]
        
        logger.info(f"CORA checkpoint loaded from {directory}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-100:]
        
        report = {
            "summary": {
                "total_interactions": self.interaction_count,
                "average_confidence": self.system_metrics["average_confidence"],
                "average_processing_time": self.system_metrics["total_processing_time"] / max(self.interaction_count, 1)
            },
            "recent_performance": {
                "confidence_trend": [entry["confidence"] for entry in recent_performance],
                "processing_time_trend": [entry["processing_time"] for entry in recent_performance],
                "evaluation_breakdown": {
                    "coherence": [entry["evaluation_scores"]["coherence"] for entry in recent_performance],
                    "factuality": [entry["evaluation_scores"]["factuality"] for entry in recent_performance],
                    "completeness": [entry["evaluation_scores"]["completeness"] for entry in recent_performance]
                }
            },
            "component_analysis": {
                "usage_distribution": self.system_metrics["component_usage"],
                "efficiency": {
                    "macl_avg_time": sum(entry["component_times"]["macl"] for entry in recent_performance) / len(recent_performance),
                    "sol_avg_time": sum(entry["component_times"]["sol"] for entry in recent_performance) / len(recent_performance),
                    "ckal_avg_time": sum(entry["component_times"]["ckal"] for entry in recent_performance) / len(recent_performance)
                }
            }
        }
        
        return report
