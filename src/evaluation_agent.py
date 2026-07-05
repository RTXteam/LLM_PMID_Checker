"""LLM agent for evaluating triples against supporting text."""
import logging
from typing import List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TripleEvaluation:
    """Result of triple evaluation against supporting text."""
    text_id: str
    support: str = "no"  # "yes", "no", "maybe"
    supporting_sentences: List[str] = field(default_factory=list)
    reasoning: str = ""
    subject_mentioned: bool = False
    object_mentioned: bool = False

    @property
    def is_supported(self) -> bool:
        """Backward-compatible property."""
        return self.support == "yes"


@dataclass
class TripleData:
    """Input triple data for evaluation."""
    subject: str
    predicate: str
    object: str
    subject_names: List[str] = None
    object_names: List[str] = None
    subject_info: Optional[dict] = None
    object_info: Optional[dict] = None

    def __post_init__(self):
        if self.subject_names is None:
            self.subject_names = [self.subject]
        if self.object_names is None:
            self.object_names = [self.object]

    def to_string(self) -> str:
        return f"'{self.subject}' {self.predicate} '{self.object}'"


class EvaluationAgent:
    """Agent for evaluating whether text supports research triples."""

    def __init__(self, llm_client, round2_client=None):
        """Initialize the evaluation agent.

        Args:
            llm_client: LLM client for Round 1 evaluation
            round2_client: Optional LLM client for Round 2 re-evaluation
        """
        self.llm_client = llm_client
        self.round2_client = round2_client
        logger.info("Evaluation agent initialized")

    async def evaluate_triple_against_text(
        self,
        triple: TripleData,
        supporting_text: str,
        text_id: str = "",
        use_round2: bool = False,
    ) -> TripleEvaluation:
        """Classify whether a supporting text supports a research triple.

        Args:
            triple: The research triple to evaluate
            supporting_text: The text snippet to classify
            text_id: Identifier for the text (e.g. supporting_text_id)
            use_round2: Whether to run Round 2 re-evaluation

        Returns:
            TripleEvaluation result
        """
        try:
            result = await self.llm_client.evaluate_text_support(
                triple=triple,
                supporting_text=supporting_text,
            )

            evaluation = self._parse_text_result(result, text_id)

            if (
                use_round2
                and self.round2_client
                and evaluation.support in ("yes", "maybe")
            ):
                logger.info(
                    f"Running Round 2 re-evaluation for {text_id} "
                    f"(Round 1: {evaluation.support})"
                )
                try:
                    r2_result = await self.round2_client.evaluate_text_support(
                        triple=triple,
                        supporting_text=supporting_text,
                    )
                    r2_eval = self._parse_text_result(r2_result, text_id)
                    r2_eval.reasoning = (
                        f"[Round2] {r2_eval.reasoning} "
                        f"[Round1 was: {evaluation.support}]"
                    )
                    evaluation = r2_eval
                except Exception as e:
                    logger.error(f"Round 2 failed for {text_id}: {e}")
                    evaluation.reasoning += f" [Round 2 failed: {str(e)}]"

            return evaluation

        except Exception as e:
            logger.error(f"Error evaluating text {text_id}: {e}")
            return TripleEvaluation(
                text_id=text_id,
                support="no",
                reasoning=f"Evaluation failed: {str(e)}",
            )

    def _parse_text_result(self, result: dict, text_id: str) -> TripleEvaluation:
        """Parse LLM result from text-mode evaluation."""
        support = result.get("support", "").lower().strip()
        if support not in ("yes", "no", "maybe"):
            if result.get("is_supported"):
                support = "yes"
            else:
                support = "no"

        return TripleEvaluation(
            text_id=text_id,
            support=support,
            supporting_sentences=[],
            reasoning=result.get("reasoning", ""),
            subject_mentioned=result.get("subject_mentioned", False),
            object_mentioned=result.get("object_mentioned", False),
        )
