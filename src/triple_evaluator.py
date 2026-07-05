"""Main triple evaluation system for evaluating triples against supporting text."""
import logging
import asyncio
from typing import List, Dict, Any
from .evaluation_agent import EvaluationAgent, TripleData, TripleEvaluation
from .llm_factory import create_llm_client
from .config import settings

logger = logging.getLogger(__name__)


class TripleEvaluationResult:
    """Container for the complete evaluation result."""

    def __init__(self, triple: TripleData, evaluations: List[TripleEvaluation]):
        self.triple = triple
        self.evaluations = evaluations

    def format_output(self, verbose: bool = False) -> str:
        """Format the evaluation results for display."""
        lines = []
        for ev in self.evaluations:
            support_text = ev.support.capitalize()
            subject_mentioned = "Yes" if ev.subject_mentioned else "No"
            object_mentioned = "Yes" if ev.object_mentioned else "No"

            main_line = (
                f"PMID:{ev.pmid}, {support_text}, "
                f"Subject:{subject_mentioned}, Object:{object_mentioned}"
            )

            if ev.supporting_sentences:
                sents = " | ".join(ev.supporting_sentences)
                main_line += f", [{sents}]"

            lines.append(main_line)

            if verbose:
                lines.append(f"  Support: {ev.support}")
                lines.append(f"  Subject Mentioned: {subject_mentioned}")
                lines.append(f"  Object Mentioned: {object_mentioned}")
                if ev.supporting_sentences:
                    lines.append("  Supporting Sentences:")
                    for s in ev.supporting_sentences:
                        lines.append(f"    - {s}")
                lines.append(f"  Reasoning: {ev.reasoning}")
                lines.append("")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        total = len(self.evaluations)
        yes_count = sum(1 for ev in self.evaluations if ev.support == "yes")
        maybe_count = sum(1 for ev in self.evaluations if ev.support == "maybe")
        no_count = sum(1 for ev in self.evaluations if ev.support == "no")

        yes_pct = (yes_count / total * 100) if total > 0 else 0.0
        maybe_pct = (maybe_count / total * 100) if total > 0 else 0.0
        no_pct = (no_count / total * 100) if total > 0 else 0.0

        return {
            "total_pmids": total,
            "yes_count": yes_count,
            "maybe_count": maybe_count,
            "no_count": no_count,
            "yes_percentage": round(yes_pct, 1),
            "maybe_percentage": round(maybe_pct, 1),
            "no_percentage": round(no_pct, 1),
            # Backward-compatible aliases
            "supported_pmids": yes_count,
            "unsupported_pmids": no_count + maybe_count,
            "supported_percentage": round(yes_pct, 1),
            "unsupported_percentage": round((no_pct + maybe_pct), 1),
        }


class TripleEvaluatorSystem:
    """System for evaluating triples against supporting text using LLMs."""

    def __init__(self, llm_provider, round2_model=None):
        """Initialize the triple evaluation system.

        Args:
            llm_provider: LLM provider for Round 1 (e.g., 'gpt-oss-20b-vllm', model name).
            round2_model: Optional model for Round 2 re-evaluation. If None, Round 2 is skipped.
        """
        llm_client = create_llm_client(llm_provider)
        round2_client = create_llm_client(round2_model) if round2_model else None

        self.evaluation_agent = EvaluationAgent(
            llm_client=llm_client,
            round2_client=round2_client,
        )
        self.round2_model = round2_model
        self.use_round2 = round2_model is not None

    async def evaluate_triple_with_text(
        self,
        subject: str,
        predicate: str,
        object_: str,
        supporting_text: str,
        text_id: str = "",
        subject_names: List[str] = None,
        object_names: List[str] = None,
        subject_info: dict = None,
        object_info: dict = None,
    ) -> TripleEvaluationResult:
        """Evaluate a research triple directly against a supporting text.

        Args:
            subject: Subject entity name
            predicate: Predicate/relation
            object_: Object entity name
            supporting_text: The text to classify
            text_id: Identifier for the text (e.g. supporting_text_id)
            subject_names: Equivalent names for the subject
            object_names: Equivalent names for the object
            subject_info: Optional node info dict
            object_info: Optional node info dict

        Returns:
            TripleEvaluationResult with a single evaluation
        """
        logger.info(
            f"Evaluating triple ['{subject}' {predicate} '{object_}'] "
            f"against text {text_id}"
        )

        triple = TripleData(
            subject=subject,
            predicate=predicate,
            object=object_,
            subject_names=subject_names or [subject],
            object_names=object_names or [object_],
            subject_info=subject_info,
            object_info=object_info,
        )

        evaluation = await self.evaluation_agent.evaluate_triple_against_text(
            triple=triple,
            supporting_text=supporting_text,
            text_id=text_id,
            use_round2=self.use_round2,
        )

        return TripleEvaluationResult(triple=triple, evaluations=[evaluation])
