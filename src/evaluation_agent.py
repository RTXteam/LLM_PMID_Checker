"""LLM agent for checking triples against PMID abstracts using Ollama."""
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TripleEvaluation:
    """Result of triple check against an abstract."""
    pmid: str
    is_supported: bool
    evidence_category: str = "dont_know"
    supporting_sentence: Optional[str] = None
    reasoning: str = ""
    subject_mentioned: bool = False
    object_mentioned: bool = False

@dataclass
class TripleData:
    """Input triple data for evaluation."""
    subject: str
    predicate: str
    object: str
    subject_names: List[str] = None
    object_names: List[str] = None
    # Qualifier fields
    qualified_predicate: Optional[str] = None
    qualified_object_aspect: Optional[str] = None
    qualified_object_direction: Optional[str] = None
    
    def __post_init__(self):
        """Initialize names lists if not provided and validate qualifiers."""
        if self.subject_names is None:
            self.subject_names = [self.subject]
        if self.object_names is None:
            self.object_names = [self.object]
        
        # Validate qualifier constraints
        self._validate_qualifiers()
    
    def _validate_qualifiers(self):
        """Validate qualifier constraints."""
        has_any_qualifier = any([
            self.qualified_predicate,
            self.qualified_object_aspect,
            self.qualified_object_direction
        ])
        
        if has_any_qualifier:
            # If any qualifier is provided, qualified_predicate must be provided
            if not self.qualified_predicate:
                raise ValueError("qualified_predicate is required when using qualifiers")
            
            # At least one of qualified_object_aspect or qualified_object_direction must be provided
            if not self.qualified_object_aspect and not self.qualified_object_direction:
                raise ValueError("At least one of qualified_object_aspect or qualified_object_direction must be provided when using qualifiers")
    
    def has_qualifiers(self) -> bool:
        """Check if this triple has qualifier information."""
        return bool(self.qualified_predicate)
    
    def to_string(self) -> str:
        """Convert triple to human-readable string."""
        if self.has_qualifiers():
            # Build qualified description
            qualified_parts = []
            if self.qualified_object_direction:
                qualified_parts.append(self.qualified_object_direction)
            if self.qualified_object_aspect:
                qualified_parts.append(self.qualified_object_aspect)
            
            qualified_description = " ".join(qualified_parts)
            return f"'{self.subject}' {self.qualified_predicate} {qualified_description} of '{self.object}'"
        else:
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
                evidence_category=result.get("evidence_category", "dont_know"),
                supporting_sentence=result["supporting_sentence"],
                reasoning=result["reasoning"],
                subject_mentioned=result.get("subject_mentioned", False),
                object_mentioned=result.get("object_mentioned", False)
            )
            
        except Exception as e:
            logger.error(f"Error evaluating PMID {pmid}: {e}")
            return TripleEvaluation(
                pmid=pmid,
                is_supported=False,
                evidence_category="dont_know",
                supporting_sentence=None,
                reasoning=f"Evaluation failed: {str(e)}",
                subject_mentioned=False,
                object_mentioned=False
            )
