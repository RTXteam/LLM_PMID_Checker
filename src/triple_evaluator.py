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
            # Map evidence categories to display formats
            category_map = {
                "direct_evidence": "Direct evidence",
                "opposite_assertion": "Opposite Assertion",
                "related_not_direct": "Related (Not Direct)",
                "not_supported": "Not mentioned or not related",
                "dont_know": "Unknown"
            }
            
            category_display = category_map.get(eval_result.evidence_category, "Unknown")
            
            # Update supported logic: direct_evidence and related_not_direct are both supported
            is_supported = eval_result.evidence_category in ["direct_evidence", "related_not_direct"]
            
            # Build the main line with all key information
            supported_text = "Supported" if is_supported else "Not Supported"
            subject_mentioned = "Yes" if eval_result.subject_mentioned else "No"
            object_mentioned = "Yes" if eval_result.object_mentioned else "No"
            
            # Main output line with key information in requested order: PMID → Supported → Category → Subject → Object → Supporting Sentence
            main_line = f"PMID:{eval_result.pmid}, {supported_text}, {category_display}, Subject:{subject_mentioned}, Object:{object_mentioned}"
            
            # Add supporting sentence if available
            if eval_result.evidence_category in ["direct_evidence", "related_not_direct"] and eval_result.supporting_sentence:
                main_line += f", [{eval_result.supporting_sentence}]"
            
            lines.append(main_line)
            
            # Add detailed reasoning in verbose mode (without confidence)
            if verbose:
                lines.append(f"  Evidence Category: {eval_result.evidence_category}")
                lines.append(f"  Subject Mentioned: {'Yes' if eval_result.subject_mentioned else 'No'}")
                lines.append(f"  Object Mentioned: {'Yes' if eval_result.object_mentioned else 'No'}")
                lines.append(f"  Supporting sentence: {eval_result.supporting_sentence}")
                lines.append(f"  Reasoning: {eval_result.reasoning}")
                lines.append("")
        
        return "\n".join(lines)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        # Update supported logic: direct_evidence and related_not_direct are both supported
        supported_count = sum(1 for eval_result in self.evaluations 
                            if eval_result.evidence_category in ["direct_evidence", "related_not_direct"])
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
                                       pmids: List[str] = None,
                                       qualified_predicate: str = None,
                                       qualified_object_aspect: str = None,
                                       qualified_object_direction: str = None) -> TripleEvaluationResult:
        """Check a research triple with equivalent names against a list of PMIDs.
        
        Args:
            subject: Subject of the triple (e.g., 'SIX1')
            predicate: Predicate/relation (e.g., 'affects') 
            object_: Object of the triple (e.g., 'Cell Proliferation')
            subject_names: List of equivalent names for the subject
            object_names: List of equivalent names for the object
            pmids: List of PubMed identifiers
            qualified_predicate: Required qualified predicate (e.g., 'causes') if any qualifier is used
            qualified_object_aspect: Optional aspect qualifier (e.g., 'activity_or_abundance') if any qualifier is used
            qualified_object_direction: Optional direction qualifier (e.g., 'increased') if any qualifier is used
            
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
            object_names=object_names or [object_],
            qualified_predicate=qualified_predicate,
            qualified_object_aspect=qualified_object_aspect,
            qualified_object_direction=qualified_object_direction
        )
        
        # Log equivalent names
        logger.info(f"Subject equivalent names: {triple.subject_names}")
        logger.info(f"Object equivalent names: {triple.object_names}")
        if triple.has_qualifiers():
            logger.info(f"Qualifiers - predicate: {triple.qualified_predicate}, aspect: {triple.qualified_object_aspect}, direction: {triple.qualified_object_direction}")
        
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
                    evidence_category="not_supported",
                    supporting_sentence=None,
                    reasoning="PMID not found in results",
                    subject_mentioned=False,
                    object_mentioned=False
                ))
                continue
                
            if data.error:
                # Error extracting abstract
                logger.warning(f"Error for PMID {pmid}: {data.error}")
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    evidence_category="dont_know",
                    supporting_sentence=None,
                    reasoning=f"Error: {data.error}",
                    subject_mentioned=False,
                    object_mentioned=False
                ))
                continue
            
            if not data.abstract.strip():
                # No abstract available
                evaluations.append(TripleEvaluation(
                    pmid=pmid,
                    is_supported=False,
                    evidence_category="not_supported",
                    supporting_sentence=None,
                    reasoning="No abstract available",
                    subject_mentioned=False,
                    object_mentioned=False
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
                        evidence_category="dont_know",
                        supporting_sentence=None,
                        reasoning=f"Evaluation failed: {str(e)}",
                        subject_mentioned=False,
                        object_mentioned=False
                    ))
        
        # Sort evaluations by PMID to maintain order
        pmid_order = {pmid: idx for idx, pmid in enumerate(pmids)}
        evaluations.sort(key=lambda x: pmid_order.get(x.pmid, float('inf')))
        
        return TripleEvaluationResult(triple=triple, evaluations=evaluations)
    
    def _validate_evaluation_logic(self, evaluation: "TripleEvaluation", pmid: str) -> "TripleEvaluation":
        """Apply validation rules to ensure logical consistency in evaluation results.
        
        Rules:
        1. If evidence_category is "not_supported", then supporting_sentence MUST be None
        2. If no supporting_sentence for categories that should have one, adjust reasoning
        
        Args:
            evaluation: The original evaluation result
            pmid: PMID for logging purposes
            
        Returns:
            Corrected evaluation result
        """
        # Rule 1: Direct evidence and related_not_direct must be supported
        if (evaluation.evidence_category in ["direct_evidence", "related_not_direct"] and not evaluation.is_supported):
            evaluation.is_supported = True
            evaluation.reasoning += " [Auto-corrected: Direct evidence and related evidence are both supported]"
        
        # Rule 2: Categories that shouldn't have supporting sentences
        if evaluation.evidence_category in ["not_supported", "dont_know", "opposite_assertion"]:
            if evaluation.supporting_sentence:
                evaluation.supporting_sentence = None
                evaluation.reasoning += " [Auto-corrected: Category doesn't require supporting sentence]"
        
        # Rule 3: Ensure consistency for unsupported cases
        if not evaluation.supporting_sentence or not evaluation.supporting_sentence.strip():
            if evaluation.evidence_category in ["direct_evidence", "related_not_direct"]:
                evaluation.supporting_sentence = None
                evaluation.reasoning += " [Auto-corrected: No supporting evidence]"
        
        return evaluation
    