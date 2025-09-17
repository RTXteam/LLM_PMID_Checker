"""Main triple checking system orchestrating PMID extraction and Ollama LLM checking."""
import logging
from typing import List, Dict, Any
from .pmid_extractor import PMIDExtractor
from .evaluation_agent import EvaluationAgent, TripleData, TripleEvaluation
from .config import settings

logger = logging.getLogger(__name__)

class TripleEvaluationResult:
    """Container for the complete evaluation result."""
    
    def __init__(self, triple: TripleData, evaluations: List[TripleEvaluation]):
        self.triple = triple
        self.evaluations = evaluations
    
    def format_output(self, verbose: bool = False) -> str:
        """Format the evaluation results"""
        lines = []
        for eval_result in self.evaluations:
            if eval_result.is_supported:
                supporting_text = f", [{eval_result.supporting_sentence}]" if eval_result.supporting_sentence else ""
                lines.append(f"Yes, PMID:{eval_result.pmid}{supporting_text}")
            else:
                lines.append(f"No, PMID:{eval_result.pmid}")
            
            # Add detailed reasoning in verbose mode
            if verbose:
                lines.append(f"  Confidence: {eval_result.confidence}")
                lines.append(f"  Supporting sentence: {eval_result.supporting_sentence}")
                lines.append(f"  Reasoning: {eval_result.reasoning}")
                lines.append("")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        supported_count = sum(1 for eval_result in self.evaluations if eval_result.is_supported)
        total_count = len(self.evaluations)
        unsupported_count = total_count - supported_count
        
        # Calculate percentages (handle division by zero)
        supported_percentage = (supported_count / total_count * 100) if total_count > 0 else 0.0
        unsupported_percentage = (unsupported_count / total_count * 100) if total_count > 0 else 0.0
        
        return {
            "total_pmids": total_count,
            "supported_pmids": supported_count,
            "unsupported_pmids": unsupported_count,
            "supported_percentage": round(supported_percentage, 1),
            "unsupported_percentage": round(unsupported_percentage, 1)
        }

class TripleEvaluatorSystem:
    """Main system for checking triples against PMID abstracts using Ollama LLMs."""
    
    def __init__(self, llm_provider):
        """Initialize the triple checking system.
        
        Args:
            llm_provider: LLM provider to use ('hermes4', 'gpt-oss').
        """
        self.pmid_extractor = PMIDExtractor(
            api_key=settings.ncbi_api_key,
            email=settings.ncbi_email
        )
        
        self.evaluation_agent = EvaluationAgent(llm_provider=llm_provider)
    
    async def evaluate_triple_with_names(self, 
                                       subject: str, 
                                       predicate: str, 
                                       object_: str,
                                       subject_names: List[str] = None,
                                       object_names: List[str] = None,
                                       pmids: List[str] = None) -> TripleEvaluationResult:
        """Check a research triple with equivalent names against a list of PMIDs.
        
        Args:
            subject: Subject of the triple (e.g., 'SIX1')
            predicate: Predicate/relation (e.g., 'affects') 
            object_: Object of the triple (e.g., 'Cell Proliferation')
            subject_names: List of equivalent names for the subject
            object_names: List of equivalent names for the object
            pmids: List of PubMed identifiers
            
        Returns:
            TripleEvaluationResult with evaluation for each PMID
        """
        logger.info(f"Checking triple ['{subject}' {predicate} '{object_}'] with equivalent names against {len(pmids)} PMIDs")
        
        # Create enriched triple data
        triple = TripleData(
            subject=subject, 
            predicate=predicate, 
            object=object_,
            subject_names=subject_names or [subject],
            object_names=object_names or [object_]
        )
        
        # Log equivalent names
        logger.info(f"Subject equivalent names: {triple.subject_names}")
        logger.info(f"Object equivalent names: {triple.object_names}")
        
        # Step 1: Extract abstracts from PMIDs
        logger.info("Extracting abstracts from PMIDs...")
        abstract_data = self.pmid_extractor.extract_abstracts(pmids)
        
        # Step 2: Prepare abstracts for evaluation (filter out errors)
        valid_abstracts = []
        evaluations = []
        
        for pmid in pmids:
            data = abstract_data.get(pmid)
            if not data:
                # PMID not found in results
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning="PMID not found in results"
                ))
                continue
                
            if data.error:
                # Error extracting abstract
                logger.warning(f"Error for PMID {pmid}: {data.error}")
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning=f"Error: {data.error}"
                ))
                continue
            
            if not data.abstract.strip():
                # No abstract available
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    supporting_sentence=None,
                    confidence=0.0,
                    reasoning="No abstract available"
                ))
                continue
            
            valid_abstracts.append((pmid, data.title, data.abstract))
        
        # Step 3: Evaluate valid abstracts using LLM
        if valid_abstracts:
            logger.info(f"Evaluating {len(valid_abstracts)} valid abstracts using LLM...")
            
            for pmid, title, abstract in valid_abstracts:
                try:
                    evaluation = await self.evaluation_agent.evaluate_triple_against_abstract(
                        triple=triple,
                        abstract=abstract,
                        pmid=pmid,
                        title=title
                    )
                    
                    # Apply validation rules to ensure logical consistency
                    evaluation = self._validate_evaluation_logic(evaluation, pmid)
                    evaluations.append(evaluation)
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate PMID {pmid}: {e}")
                    evaluations.append(TripleEvaluation(
                        pmid=pmid,
                        is_supported=False,
                        supporting_sentence=None,
                        confidence=0.0,
                        reasoning=f"Evaluation failed: {str(e)}"
                    ))
        
        # Sort evaluations by PMID to maintain order
        pmid_order = {pmid: idx for idx, pmid in enumerate(pmids)}
        evaluations.sort(key=lambda x: pmid_order.get(x.pmid, float('inf')))
        
        return TripleEvaluationResult(triple=triple, evaluations=evaluations)
    
    def _validate_evaluation_logic(self, evaluation: "TripleEvaluation", pmid: str) -> "TripleEvaluation":
        """Apply validation rules to ensure logical consistency in evaluation results.
        
        Rules:
        1. If confidence > 0.8 AND supporting_sentence provided, then is_supported MUST be true
        2. If supporting_sentence cannot be provided (null/empty), set confidence to 0.0 and is_supported MUST be false
        
        Args:
            evaluation: The original evaluation result
            pmid: PMID for logging purposes
            
        Returns:
            Corrected evaluation result
        """
        # Rule 1: High confidence + supporting sentence = must be supported
        if (evaluation.confidence > 0.8 and 
            evaluation.supporting_sentence and 
            evaluation.supporting_sentence.strip() and
            not evaluation.is_supported):
            
            evaluation.is_supported = True
            evaluation.reasoning += " [Auto-corrected: High confidence with supporting evidence]"
        
        # Rule 2: Only force false when no supporting evidence
        if not evaluation.supporting_sentence or not evaluation.supporting_sentence.strip():
            
            evaluation.is_supported = False
            evaluation.supporting_sentence = None
            evaluation.confidence = 0.0
            evaluation.reasoning += " [Auto-corrected: No supporting evidence]"
        
        return evaluation
    