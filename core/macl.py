import torch
from typing import Dict, List, Optional, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..agents.analyst import AnalystAgent
from ..agents.critic import CriticAgent
from ..agents.synthesizer import SynthesizerAgent
from ..agents.innovator import InnovatorAgent

logger = logging.getLogger(__name__)

class MACL:
    """
    Multi-Agent Cognitive Coordination Layer
    Handles distributed reasoning through specialized agents
    """
    
    def __init__(self, base_model, config: Dict[str, Any]):
        self.config = config
        self.base_model = base_model
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
        # Coordination state
        self.conversation_history = []
        self.agent_performance = {}
        self.coordination_strategy = config.get("coordination_strategy", "weighted_consensus")
        
        logger.info("MACL initialized with {} agents".format(len(self.agents)))
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents"""
        agent_config = self.config.get("agent_config", {})
        
        agents = {
            "analyst": AnalystAgent(self.base_model, agent_config.get("analyst", {})),
            "critic": CriticAgent(self.base_model, agent_config.get("critic", {})),
            "synthesizer": SynthesizerAgent(self.base_model, agent_config.get("synthesizer", {})),
            "innovator": InnovatorAgent(self.base_model, agent_config.get("innovator", {}))
        }
        
        return agents
    
    def process_query(self, query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process query through multi-agent collaboration
        
        Args:
            query: Input query to process
            context: Optional conversation context
            
        Returns:
            Dictionary containing final response and agent contributions
        """
        start_time = time.time()
        
        # Update conversation history
        if context:
            self.conversation_history.extend(context)
        self.conversation_history.append(f"User: {query}")
        
        try:
            # Phase 1: Parallel agent processing
            agent_responses = self._parallel_agent_processing(query)
            
            # Phase 2: Agent coordination and consensus
            final_response = self._reach_consensus(agent_responses)
            
            # Phase 3: Response refinement
            refined_response = self._refine_response(final_response, agent_responses)
            
            # Update performance tracking
            processing_time = time.time() - start_time
            self._update_agent_performance(agent_responses, processing_time)
            
            result = {
                "final_response": refined_response,
                "agent_contributions": agent_responses,
                "coordination_strategy": self.coordination_strategy,
                "processing_time": processing_time,
                "agent_performance": self.agent_performance.copy()
            }
            
            # Update conversation history
            self.conversation_history.append(f"Assistant: {refined_response}")
            
            logger.info(f"MACL processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in MACL processing: {str(e)}")
            # Fallback to single-agent response
            fallback_response = self.agents["analyst"].process(query)
            return {
                "final_response": fallback_response,
                "agent_contributions": {"analyst": fallback_response},
                "coordination_strategy": "fallback",
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _parallel_agent_processing(self, query: str) -> Dict[str, Any]:
        """Process query in parallel across all agents"""
        agent_responses = {}
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit tasks to all agents
            future_to_agent = {
                executor.submit(self._safe_agent_process, agent_name, agent, query): agent_name
                for agent_name, agent in self.agents.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    response = future.result(timeout=self.config.get("agent_timeout", 30))
                    agent_responses[agent_name] = response
                except Exception as e:
                    logger.warning(f"Agent {agent_name} failed: {str(e)}")
                    agent_responses[agent_name] = f"Agent {agent_name} unavailable: {str(e)}"
        
        return agent_responses
    
    def _safe_agent_process(self, agent_name: str, agent: Any, query: str) -> str:
        """Safely process query with individual agent"""
        try:
            return agent.process(query, self.conversation_history[-5:])  # Last 5 turns as context
        except Exception as e:
            logger.error(f"Agent {agent_name} error: {str(e)}")
            return f"Error in {agent_name} processing: {str(e)}"
    
    def _reach_consensus(self, agent_responses: Dict[str, str]) -> str:
        """Reach consensus among agent responses"""
        if self.coordination_strategy == "weighted_consensus":
            return self._weighted_consensus(agent_responses)
        elif self.coordination_strategy == "synthesizer_lead":
            return self._synthesizer_lead(agent_responses)
        elif self.coordination_strategy == "critical_review":
            return self._critical_review(agent_responses)
        else:
            return self._default_consensus(agent_responses)
    
    def _weighted_consensus(self, agent_responses: Dict[str, str]) -> str:
        """Weighted consensus based on agent expertise and confidence"""
        weights = self.config.get("agent_weights", {
            "analyst": 0.25,
            "critic": 0.30,
            "synthesizer": 0.30,
            "innovator": 0.15
        })
        
        # Calculate weighted response
        weighted_parts = []
        for agent_name, response in agent_responses.items():
            weight = weights.get(agent_name, 0.1)
            if response and not response.startswith("Error"):
                weighted_parts.append(f"[{agent_name.upper()}] {response}")
        
        if weighted_parts:
            consensus = "Based on collaborative analysis:\n\n" + "\n\n".join(weighted_parts)
        else:
            consensus = "I need more time to analyze this question thoroughly."
        
        return consensus
    
    def _synthesizer_lead(self, agent_responses: Dict[str, str]) -> str:
        """Synthesizer agent leads the response integration"""
        synthesizer_response = agent_responses.get("synthesizer", "")
        other_responses = {k: v for k, v in agent_responses.items() if k != "synthesizer"}
        
        if synthesizer_response and not synthesizer_response.startswith("Error"):
            integration_note = "Integrating perspectives from specialized analysis:\n\n"
            for agent_name, response in other_responses.items():
                if response and not response.startswith("Error"):
                    integration_note += f"- {agent_name.title()}: {response}\n"
            
            return f"{synthesizer_response}\n\n{integration_note}"
        else:
            return self._default_consensus(agent_responses)
    
    def _critical_review(self, agent_responses: Dict[str, str]) -> str:
        """Critic agent reviews and refines the consensus"""
        base_consensus = self._default_consensus(agent_responses)
        critic_response = agent_responses.get("critic", "")
        
        if critic_response and not critic_response.startswith("Error"):
            return f"{base_consensus}\n\nCritical Review: {critic_response}"
        else:
            return base_consensus
    
    def _default_consensus(self, agent_responses: Dict[str, str]) -> str:
        """Default consensus mechanism"""
        valid_responses = [resp for resp in agent_responses.values() 
                         if resp and not resp.startswith("Error")]
        
        if valid_responses:
            # Simple concatenation
            return "Collaborative analysis suggests:\n\n" + "\n\n".join(valid_responses)
        else:
            return "I apologize, but I'm unable to provide a comprehensive analysis at the moment."
    
    def _refine_response(self, base_response: str, agent_responses: Dict[str, str]) -> str:
        """Refine the final response based on agent feedback"""
        # Get refinement suggestions from critic
        critic_response = agent_responses.get("critic", "")
        
        if critic_response and "suggestion" in critic_response.lower():
            # Extract suggestions from critic response
            lines = critic_response.split('\n')
            suggestions = [line for line in lines if any(word in line.lower() 
                                                       for word in ['suggest', 'recommend', 'improve', 'better'])]
            
            if suggestions:
                refinement_note = "\n\nRefinement based on critical analysis:\n" + "\n".join(suggestions[:2])
                return base_response + refinement_note
        
        return base_response
    
    def _update_agent_performance(self, agent_responses: Dict[str, str], processing_time: float):
        """Update agent performance tracking"""
        for agent_name, response in agent_responses.items():
            if agent_name not in self.agent_performance:
                self.agent_performance[agent_name] = {
                    "response_count": 0,
                    "error_count": 0,
                    "avg_response_time": 0,
                    "recent_responses": []
                }
            
            agent_perf = self.agent_performance[agent_name]
            agent_perf["response_count"] += 1
            
            if response.startswith("Error"):
                agent_perf["error_count"] += 1
            
            # Update average response time (simplified)
            individual_time = processing_time / len(agent_responses)
            if agent_perf["avg_response_time"] == 0:
                agent_perf["avg_response_time"] = individual_time
            else:
                agent_perf["avg_response_time"] = (agent_perf["avg_response_time"] + individual_time) / 2
            
            # Track recent responses
            agent_perf["recent_responses"].append({
                "timestamp": time.time(),
                "response_length": len(response),
                "had_error": response.startswith("Error")
            })
            
            # Keep only recent history
            agent_perf["recent_responses"] = agent_perf["recent_responses"][-100:]
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get MACL coordination statistics"""
        stats = {
            "total_queries_processed": len(self.conversation_history) // 2,  # Each query has user and assistant turn
            "active_agents": len(self.agents),
            "coordination_strategy": self.coordination_strategy,
            "agent_performance": {}
        }
        
        for agent_name, perf in self.agent_performance.items():
            stats["agent_performance"][agent_name] = {
                "response_count": perf["response_count"],
                "error_rate": perf["error_count"] / max(perf["response_count"], 1),
                "avg_response_time": perf["avg_response_time"],
                "reliability": 1 - (perf["error_count"] / max(perf["response_count"], 1))
            }
        
        return stats
    
    def set_coordination_strategy(self, strategy: str):
        """Set coordination strategy"""
        valid_strategies = ["weighted_consensus", "synthesizer_lead", "critical_review", "default"]
        if strategy in valid_strategies:
            self.coordination_strategy = strategy
            logger.info(f"MACL coordination strategy set to: {strategy}")
        else:
            logger.warning(f"Invalid coordination strategy: {strategy}")
    
    def add_custom_agent(self, agent_name: str, agent_instance: Any):
        """Add custom agent to the coordination system"""
        self.agents[agent_name] = agent_instance
        logger.info(f"Custom agent '{agent_name}' added to MACL")
    
    def get_conversation_history(self, last_n: Optional[int] = None) -> List[str]:
        """Get conversation history"""
        if last_n:
            return self.conversation_history[-last_n:]
        return self.conversation_history.copy()
