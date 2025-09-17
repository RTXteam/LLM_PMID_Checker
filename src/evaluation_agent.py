"""LLM agent for checking triples against PMID abstracts using Ollama."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .config import settings

logger = logging.getLogger(__name__)

@dataclass
class TripleEvaluation:
    """Result of triple check against an abstract."""
    pmid: str
    is_supported: bool
    supporting_sentence: Optional[str] = None
    confidence: float = 0.5
    reasoning: str = ""

@dataclass
class TripleData:
    """Input triple data for evaluation."""
    subject: str
    predicate: str
    object: str
    subject_names: List[str] = None
    object_names: List[str] = None
    
    def __post_init__(self):
        """Initialize names lists if not provided."""
        if self.subject_names is None:
            self.subject_names = [self.subject]
        if self.object_names is None:
            self.object_names = [self.object]
    
    def to_string(self) -> str:
        """Convert triple to human-readable string."""
        return f"'{self.subject}' {self.predicate} '{self.object}'"

class EvaluationAgent:
    """Agent for checking whether abstracts support research triples."""
    
    def __init__(self, llm_provider):
        """Initialize the checking agent.
        
        Args:
            llm_provider: LLM provider to use ('hermes4', 'gpt-oss')
        """
        # Use factory to create appropriate LLM client
        try:
            from .llm_factory import create_llm_client
            self.llm_client = create_llm_client(llm_provider)
        except Exception as e:
            logger.error(f"Failed to create LLM client: {e}")
            logger.error("Please configure a valid LLM provider and ensure Ollama is running")
            raise
    
    async def evaluate_triple_against_abstract(self, 
                                             triple: TripleData, 
                                             abstract: str, 
                                             pmid: str,
                                             title: str = "") -> TripleEvaluation:
        """Check whether an abstract supports a research triple.
        
        Args:
            triple: The research triple to evaluate
            abstract: The abstract text to analyze
            pmid: PubMed ID for the abstract
            title: Article title (optional)
            
        Returns:
            TripleEvaluation result
        """
        
        try:
            # Use the LLM client to evaluate
            result = await self.llm_client.evaluate_triple_support(
                triple=triple,
                abstract=abstract
            )
            
            return TripleEvaluation(
                pmid=pmid,
                is_supported=result["is_supported"],
                supporting_sentence=result["supporting_sentence"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )
            
        except Exception as e:
            logger.error(f"Error evaluating PMID {pmid}: {e}")
            return TripleEvaluation(
                pmid=pmid,
                is_supported=False,
                supporting_sentence=None,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}"
            )
