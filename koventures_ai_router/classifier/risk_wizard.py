"""Risk wizard: classify whether an AI feature is high-risk under the EU AI Act."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..governance.ai_governance import ModelRiskLevel as RiskTier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskAssessmentResult:
    """Result of a risk wizard assessment."""
    risk_tier: RiskTier
    score: float
    flags: List[str]
    rationale: str
    remediation: Optional[str] = None


class _Check:
    """Single risk check: a question, a weight, and a ['yes'/no/unknown'] answer."""

    def __init__(self, key: str, weight: float, description: str):
        self.key = key
        self.weight = weight
        self.description = description
        self.answer: Optional[bool] = None  # True = yes, False = no, None = unknown

    def ask(self, answer: str) -> None:
        a = answer.strip().lower()
        if a in {"yes", "y", "true", "1"}:
            self.answer = True
        elif a in {"no", "n", "false", "0"}:
            self.answer = False
        else:
            self.answer = None

    @property
    def contributes(self) -> float:
        if self.answer is True:
            return self.weight
        return 0.0


class RiskWizard:
    """EU AI Act high-risk feature classifier.

    Usage:
        wizard = RiskWizard()
        result = wizard.assess(
            feature_name="Student grade prediction",
            answers={
                "critical_infrastructure": "no",
                "education_admission": "yes",
                "access_to_essential_services": "no",
            },
        )
        print(result.risk_tier)
    """

    HIGH_RISK_THRESHOLD = 0.5

    def __init__(self):
        self._checks: List[_Check] = self._build_checks()

    def _build_checks(self) -> List[_Check]:
        checks = [
            _Check("critical_infrastructure", 1.0, "Does this feature operate in critical infrastructure (energy, transport, water, etc.)?"),
            _Check("education_access", 0.8, "Does it determine access to, or successful completion of, educational courses or institutions?"),
            _Check("employment_access", 0.8, "Does it determine access to employment, recruitment, or promotion?"),
            _Check("essential_services", 0.8, "Does it determine access to essential private services (credit scoring, insurance pricing, etc.)?"),
            _Check("law_enforcement", 1.0, "Is it used by or for law enforcement or migration/border control?"),
            _Check("administration_of_justice", 1.0, "Does it assist in the administration of justice or democratic processes?"),
            _Check("biometric_remote", 1.0, "Does it use remote biometric identification or categorisation based on sensitive attributes?"),
            _Check("emotion_recognition", 0.6, "Does it use emotion recognition or facial analysis in the workplace, education, or law enforcement?"),
            _Check("social_scoring", 1.0, "Does it perform social scoring based on social behaviour or predicted traits?"),
            _Check("sensitive_data", 0.4, "Does it process special-category data (health, biometric, race, religion, etc.)?"),
            _Check("vulnerable_groups", 0.6, "Is it used with or influences vulnerable groups (children, elderly, persons with disabilities)?"),
            _Check("high_automation", 0.4, "Does it make fully automated decisions with no meaningful human review?"),
        ]
        return checks

    @property
    def questions(self) -> List[str]:
        return [c.description for c in self._checks]

    def assess(
        self,
        *,
        feature_name: str,
        answers: Dict[str, str],
    ) -> RiskAssessmentResult:
        """Run the wizard for a given feature.

        Args:
            feature_name: Human-readable name of the feature.
            answers: Map of check_key -> 'yes'/'no'/'unknown'.

        Returns:
            RiskAssessmentResult indicating the assigned risk tier.
        """
        flags: List[str] = []
        score = 0.0

        for check in self._checks:
            raw = answers.get(check.key, "unknown")
            check.ask(raw)
            score += check.contributes
            if check.answer is True:
                flags.append(check.key)

        # Map score to tier
        if any(k in flags for k in {
            "critical_infrastructure",
            "law_enforcement",
            "administration_of_justice",
            "biometric_remote",
            "social_scoring",
        }):
            tier = RiskTier.PROHIBITED if "biometric_remote" in flags and score >= self.HIGH_RISK_THRESHOLD else RiskTier.HIGH
        elif score >= self.HIGH_RISK_THRESHOLD:
            tier = RiskTier.HIGH
        elif score >= 0.25:
            tier = RiskTier.LIMITED
        else:
            tier = RiskTier.MINIMAL

        rationales = {
            RiskTier.PROHIBITED: (
                "One or more prohibited practices (Annex III) were identified. "
                "High-risk deployment is not permitted without a qualified exception."
            ),
            RiskTier.HIGH: (
                "The feature falls within Annex III high-risk categories and/or "
                "the aggregate score exceeds the threshold."
            ),
            RiskTier.LIMITED: (
                "Some risk factors were flagged but the feature does not clearly "
                "qualify as high-risk. Limited-risk transparency obligations likely apply."
            ),
            RiskTier.MINIMAL: (
                "No significant risk factors were identified."
            ),
        }

        remediation = None
        if tier in {RiskTier.HIGH, RiskTier.PROHIBITED}:
            remediation = (
                "Conduct a conformity assessment, prepare technical documentation, "
                "implement human-in-the-loop review, and register the system where required."
            )
        elif tier == RiskTier.LIMITED:
            remediation = (
                "Ensure users are informed that they are interacting with an AI system."
            )

        logger.info(
            "RiskWizard assessed %s: tier=%s score=%.2f flags=%s",
            feature_name,
            tier.value,
            score,
            flags,
        )

        return RiskAssessmentResult(
            risk_tier=tier,
            score=score,
            flags=flags,
            rationale=rationales[tier],
            remediation=remediation,
        )
