import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from collections import deque
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class CKAL:
    """
    Continuous Knowledge Adaptation Layer
    Handles dynamic memory consolidation and parameter-efficient learning
    """
    
    def __init__(self, base_model, config: Dict[str, Any]):
        self.config = config
        self.base_model = base_model
        self.memory_buffer = deque(maxlen=config.get("memory_size", 1000))
        self.consolidation_threshold = config.get("consolidation_threshold", 0.7)
        
        # Initialize LoRA for parameter-efficient fine-tuning
        self._setup_lora_adapter()
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        self.training_history = {
            "losses": [],
            "performance": [],
            "memory_usage": []
        }
        
        logger.info("CKAL initialized successfully")
    
    def _setup_lora_adapter(self):
        """Setup LoRA adapter for parameter-efficient fine-tuning"""
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj", "k_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        logger.info(f"LoRA adapter configured with r={lora_config.r}, alpha={lora_config.lora_alpha}")
    
    def consolidate_experience(self, experience: Dict[str, Any]) -> float:
        """
        Consolidate new experience into memory and update parameters
        
        Args:
            experience: Dictionary containing input, output, feedback, metadata
            
        Returns:
            loss: Computed loss value
        """
        try:
            # Store experience in memory buffer
            self.memory_buffer.append(experience)
            
            # Extract components for learning
            input_text = experience["input"]
            feedback_score = experience.get("feedback", 0.5)
            ground_truth = experience.get("ground_truth")
            
            # Tokenize input
            inputs = self.base_model.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.config.get("max_length", 512),
                truncation=True,
                padding=True
            )
            
            # Enable training mode
            self.model.train()
            
            # Forward pass with labels for language modeling
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"]  # Auto-regressive language modeling
            )
            
            # Weight loss by feedback (learn more from poor performances)
            base_loss = outputs.loss
            weighted_loss = base_loss * (1.0 - feedback_score)
            
            # Backward pass and optimization
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Switch back to evaluation mode
            self.model.eval()
            
            # Update training history
            self.training_history["losses"].append(weighted_loss.item())
            
            logger.debug(f"CKAL consolidation completed with loss: {weighted_loss.item():.4f}")
            return weighted_loss.item()
            
        except Exception as e:
            logger.error(f"Error in CKAL consolidation: {str(e)}")
            raise
    
    def retrieve_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query similarity
        
        Args:
            query: Input query to find relevant memories for
            top_k: Number of top memories to retrieve
            
        Returns:
            List of relevant memory entries
        """
        if not self.memory_buffer:
            return []
        
        # Simple keyword-based retrieval (can be enhanced with embeddings)
        query_lower = query.lower()
        relevant_memories = []
        
        for memory in list(self.memory_buffer):
            memory_text = f"{memory.get('input', '')} {memory.get('output', '')}".lower()
            
            # Simple word overlap scoring
            query_words = set(query_lower.split())
            memory_words = set(memory_text.split())
            overlap = len(query_words.intersection(memory_words))
            score = overlap / max(len(query_words), 1)
            
            if score > 0.1:  # Minimum similarity threshold
                relevant_memories.append((score, memory))
        
        # Sort by relevance and return top-k
        relevant_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for score, memory in relevant_memories[:top_k]]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the current memory state"""
        return {
            "memory_size": len(self.memory_buffer),
            "average_loss": np.mean(self.training_history["losses"][-100:]) if self.training_history["losses"] else 0,
            "recent_performance": self.training_history["performance"][-10:] if self.training_history["performance"] else [],
            "memory_usage_mb": len(json.dumps(list(self.memory_buffer)).encode('utf-8')) / (1024 * 1024)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save CKAL state to checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "memory_buffer": list(self.memory_buffer),
            "training_history": self.training_history,
            "config": self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"CKAL checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load CKAL state from checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.memory_buffer = deque(checkpoint["memory_buffer"], maxlen=self.config.get("memory_size", 1000))
        self.training_history = checkpoint["training_history"]
        
        logger.info(f"CKAL checkpoint loaded from {filepath}")
