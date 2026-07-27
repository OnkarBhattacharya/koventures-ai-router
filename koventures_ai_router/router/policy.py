"""Routing policy for the KOVentures AI router.

This module is the single place a compliance officer (not a developer)
should need to edit when a new feature is added or a regulator's
guidance changes. It maps a declared *use-case* to:

  - an EU AI Act risk classification (reusing `ModelRiskLevel` from
    `governance.ai_governance`, so routing and governance share one
    vocabulary instead of two)
  - a decision-type under GDPR Art. 22 / AI Act Art. 14 terms
    (`AIDecisionType`), which determines whether human oversight is
    mandatory before an output is acted on
  - the set of model tiers permitted to serve that use-case

Nothing in this file talks to a specific vertical (education, HR,
insurance, ...). Verticals declare their own `UseCase` entries and
policies; they do not modify the router or governance modules.

Fails closed: an undeclared use-case raises rather than silently
routing to a default model. Every new feature must be explicitly
classified before it can send traffic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from ..governance.ai_governance import AIDecisionType, ModelRiskLevel


class UseCase(Enum):
    """Named use-cases a caller declares when requesting a completion.

    Extend per-vertical by adding entries here (or, once this is a
    standalone package, in a vertical-specific policy overlay) — never
    by branching inside the router itself.
    """

    OPEN_CHAT = "open_chat"                    # e.g. Socratic tutoring, general Q&A
    CONTENT_GENERATION = "content_generation"   # e.g. lesson material, draft copy
    ASSESSMENT_SCORING = "assessment_scoring"   # e.g. grading, progress flags, screening scores
    ADMISSION_SCREENING = "admission_screening"  # e.g. applicant/candidate evaluation
    PROCTORING_MONITORING = "proctoring_monitoring"  # e.g. exam integrity, behavioural flags


class DataSensitivity(Enum):
    NONE = "none"
    PSEUDONYMISED = "pseudonymised"
    IDENTIFIABLE = "identifiable"


@dataclass(frozen=True)
class UseCasePolicy:
    risk_level: ModelRiskLevel
    decision_type: AIDecisionType
    human_review_required: bool
    allowed_model_tiers: List[str]
    max_data_sensitivity: DataSensitivity


# ---------------------------------------------------------------------
# Use-case -> classification table.
#
# Anything that only assists a student/candidate/user who remains the
# actual decision-maker (FULLY_HUMAN) stays limited-risk. Anything that
# produces a score, flag, or recommendation that a teacher/reviewer acts
# on (HUMAN_OVERSIGHT) is treated as high-risk regardless of how good
# the underlying model is — the classification follows the *use*, not
# the model's capability.
# ---------------------------------------------------------------------
USE_CASE_POLICY: Dict[UseCase, UseCasePolicy] = {
    UseCase.OPEN_CHAT: UseCasePolicy(
        risk_level=ModelRiskLevel.LIMITED,
        decision_type=AIDecisionType.FULLY_HUMAN,
        human_review_required=False,
        allowed_model_tiers=["tier0", "tier1", "tier2", "tier3", "tier_r", "open_frontier"],
        max_data_sensitivity=DataSensitivity.PSEUDONYMISED,
    ),
    UseCase.CONTENT_GENERATION: UseCasePolicy(
        risk_level=ModelRiskLevel.LIMITED,
        decision_type=AIDecisionType.HUMAN_INVOLVED,
        human_review_required=False,
        allowed_model_tiers=["tier1", "tier2", "tier3", "open_frontier"],
        max_data_sensitivity=DataSensitivity.PSEUDONYMISED,
    ),
    UseCase.ASSESSMENT_SCORING: UseCasePolicy(
        risk_level=ModelRiskLevel.HIGH,
        decision_type=AIDecisionType.HUMAN_OVERSIGHT,
        human_review_required=True,
        allowed_model_tiers=["tier2", "tier3", "open_frontier"],
        max_data_sensitivity=DataSensitivity.IDENTIFIABLE,
    ),
    UseCase.ADMISSION_SCREENING: UseCasePolicy(
        risk_level=ModelRiskLevel.HIGH,
        decision_type=AIDecisionType.HUMAN_OVERSIGHT,
        human_review_required=True,
        allowed_model_tiers=["tier3", "open_frontier"],
        max_data_sensitivity=DataSensitivity.IDENTIFIABLE,
    ),
    UseCase.PROCTORING_MONITORING: UseCasePolicy(
        risk_level=ModelRiskLevel.HIGH,
        decision_type=AIDecisionType.HUMAN_OVERSIGHT,
        human_review_required=True,
        allowed_model_tiers=["tier3", "open_frontier"],
        max_data_sensitivity=DataSensitivity.IDENTIFIABLE,
    ),
}


# ---------------------------------------------------------------------
# Model catalog: tier -> candidate models.
#
# "open_frontier" is a new tier alongside the existing tier0-tier3/tier_r
# ladder from model_security.py. It groups MIT-licensed open-weight
# models, which under AI Act Art. 53(2) carry lighter GPAI provider
# obligations unless designated systemic-risk — routing preferentially
# into this tier is a deliberate compliance-cost lever, not just a
# cost/latency one.
#
# tier0-tier2 and part of tier3 are unchanged from the existing
# model_security.py allowlist and are omitted here for brevity; merge
# at load time rather than duplicating that list.
# ---------------------------------------------------------------------
MODEL_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "open_frontier": [
        {"id": "glm-5.2", "adapter": "glm_5_2", "licence": "MIT"},
        {"id": "deepseek-v4-pro", "adapter": "deepseek_v4", "licence": "MIT"},
        {"id": "deepseek-v4-flash", "adapter": "deepseek_v4", "licence": "MIT"},
        {"id": "minimax-m3", "adapter": "minimax_m3", "licence": "MIT"},
    ],
    "tier3": [
        {"id": "deepseek-ai/DeepSeek-V3", "adapter": "hf_hosted", "licence": "MIT"},
        {"id": "meta-llama/Llama-3.1-405B", "adapter": "hf_hosted", "licence": "Llama 3.1 Community"},
    ],
}


def resolve(use_case: UseCase) -> UseCasePolicy:
    """Look up the policy for a use-case.

    Raises on anything undeclared — fail closed. A new feature must be
    run through the risk classification step and given an explicit
    entry above before it can route any traffic.
    """
    try:
        return USE_CASE_POLICY[use_case]
    except KeyError as exc:
        raise ValueError(
            f"No policy declared for {use_case!r}. Classify this use-case "
            "(risk level, decision type, human oversight requirement) "
            "before routing traffic for it."
        ) from exc