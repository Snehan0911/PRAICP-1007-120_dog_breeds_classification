"""
CanineVision AI — Confidence Interpretation & Uncertainty Quantification.

Critical AI Engineering Principle:
    Softmax outputs are NOT mathematically calibrated probabilities.
    Under the closed-world assumption of a 120-class classifier,
    even if you feed an image of a bicycle, cat, or random noise,
    the 120 softmax values MUST sum to 1.0.

    Therefore, a naive system might report "92% Chihuahua" on a cat image.
    To prevent misleading users in enterprise systems, we implement:
    1. Tiered confidence interpretation based on empirical validation thresholds.
    2. Out-of-distribution / uncertainty warnings when top-1 probability is below safe margins.
    3. Top-1 vs Top-2 margin analysis (entropy proxy).
"""

from dataclasses import dataclass


@dataclass
class ConfidenceReport:
    """Structured confidence breakdown for a single inference call."""

    level: str  # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    score: float
    is_reliable: bool
    warning: str | None
    explanation: str


class ConfidenceInterpreter:
    """
    Translates raw softmax probabilities into human-interpretable confidence assessments.

    Thresholds:
        High: >= 0.70
        Medium: 0.40 - 0.69
        Low: 0.20 - 0.39
        Uncertain / OOD Warning: < 0.20
    """

    def __init__(
        self,
        high_threshold: float = 0.70,
        medium_threshold: float = 0.40,
        uncertain_threshold: float = 0.20,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.uncertain_threshold = uncertain_threshold

    def interpret(self, top_probabilities: list[float]) -> ConfidenceReport:
        """
        Analyzes the top prediction probabilities.

        Args:
            top_probabilities: List of descending float scores (e.g. [0.78, 0.12, 0.05]).

        Returns:
            ConfidenceReport dataclass.
        """
        if not top_probabilities:
            return ConfidenceReport(
                level="UNCERTAIN",
                score=0.0,
                is_reliable=False,
                warning="No prediction scores provided.",
                explanation="Model did not return probabilities.",
            )

        top_score = float(top_probabilities[0])
        margin = float(top_probabilities[0] - top_probabilities[1]) if len(top_probabilities) > 1 else top_score

        if top_score >= self.high_threshold:
            return ConfidenceReport(
                level="HIGH",
                score=top_score,
                is_reliable=True,
                warning=None,
                explanation=f"Strong pattern alignment ({top_score*100:.1f}%) with significant confidence margin.",
            )
        elif top_score >= self.medium_threshold:
            warning = "Secondary breed characteristics detected." if margin < 0.15 else None
            return ConfidenceReport(
                level="MEDIUM",
                score=top_score,
                is_reliable=True,
                warning=warning,
                explanation=f"Moderate match ({top_score*100:.1f}%). Possible mixed breed or visual ambiguity.",
            )
        elif top_score >= self.uncertain_threshold:
            return ConfidenceReport(
                level="LOW",
                score=top_score,
                is_reliable=False,
                warning="Low confidence prediction. Verify visually.",
                explanation=f"Low alignment ({top_score*100:.1f}%). Subject may be occluded, poorly lit, or a mixed breed.",
            )
        else:
            return ConfidenceReport(
                level="UNCERTAIN",
                score=top_score,
                is_reliable=False,
                warning="Potential Out-of-Distribution or Non-Dog image detected.",
                explanation=(
                    f"Confidence score ({top_score*100:.1f}%) is below reliable operating threshold. "
                    "The uploaded image may not clearly portray one of the 120 recognized dog breeds."
                ),
            )
