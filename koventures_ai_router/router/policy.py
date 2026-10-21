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

import os
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
# ONE MODEL AUTHORITY (plan task 2.3, F5): the allowlist in
# chat-app/backend/services/model_security.py is the single source of
# truth. MODEL_CATALOG is *generated* from that allowlist at import time
# by applying the tier mapping below (tier0 / tier3 / open_frontier) —
# the two structures cannot drift apart because there is only one list.
#
# "open_frontier" is a tier alongside the existing tier0-tier3/tier_r
# ladder from model_security.py. It groups MIT/Apache-licensed
# open-weight models, which under AI Act Art. 53(2) carry lighter GPAI
# provider obligations unless designated systemic-risk — routing
# preferentially into this tier is a deliberate compliance-cost lever,
# not just a cost/latency one.
#
# The MODEL_ALLOWLIST / MODEL_ALLOWLIST_FILE environment variables remain
# supported (Q5: not deprecated) — an operator-added allowlist entry is
# automatically picked up here and lands in open_frontier with the
# offline Ollama adapter as the safe default.
# ---------------------------------------------------------------------


def _load_allowlist_ids() -> set:
    """Load the canonical allowlist from chat-app model_security at import time.

    model_security.py is stdlib-only, so loading it from its file path is
    safe from inside the router package (the chat-app directory name
    contains a hyphen and cannot be imported as a module directly).

    Works whether this package is imported from the repo root or from the
    nested ``chat-app/koventures_ai_router`` copy (parent walk would otherwise
    resolve to ``chat-app/chat-app/...``).
    """
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    # Walk up looking for chat-app/backend/services/model_security.py
    cur = here
    for _ in range(6):
        candidates.append(
            os.path.join(cur, "chat-app", "backend", "services", "model_security.py")
        )
        candidates.append(
            os.path.join(cur, "backend", "services", "model_security.py")
        )
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        tried = ", ".join(candidates[:6])
        raise RuntimeError(
            "Single model authority missing: cannot load the allowlist from any of "
            f"[{tried}]. The router refuses to run without the canonical "
            "model_security allowlist (plan task 2.3)."
        )
    spec = importlib.util.spec_from_file_location("_model_security_allowlist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module._ALLOWLIST)


def _adapter_for(model_id: str) -> str:
    """Map an allowlisted model id to a registered adapter."""
    if model_id == "gemini-2.0-flash":
        return "gemini_flash"
    if model_id.startswith("groq/") or model_id.endswith(":free"):
        # OpenRouter-hosted models: legacy groq/ ids and all :free ids
        # (e.g. google/gemma-4-26b-a4b-it:free). Without this rule :free
        # ids would fall through to the ollama adapter and hit a local
        # daemon instead of OpenRouter.
        return "openrouter"
    # Everything else (Ollama-served open-weight models) uses the offline
    # backstop adapter.
    return "ollama"


def _licence_for(model_id: str) -> str:
    if model_id == "gemini-2.0-flash" or model_id.startswith("groq/"):
        return "Proprietary"
    if model_id.startswith("meta-llama/"):
        return "Llama-3.1"
    if model_id.startswith("google/gemma"):
        return "gemma"
    if model_id.startswith("deepseek-ai/"):
        return "MIT"
    # Qwen, Microsoft Phi, Mistral families are Apache-2.0
    return "Apache-2.0"


# High-capability models permitted for high-risk use-cases (tier3).
_TIER3_IDS = {
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B",
    "mistralai/Mistral-Large",
    "mistralai/Mixtral-8x22B",
    "deepseek-ai/DeepSeek-V3",
}

# Always-available cheap tier (tier0): fast primary + offline backstop.
# NOTE: the production online chain (gemma-4-26b-a4b-it:free,
# llama-3.3-70b-instruct:free, nemotron-3-ultra) intentionally stays in
# open_frontier — open_frontier is allowed by every use-case policy,
# while tier0 is excluded from ASSESSMENT_SCORING (homework), so placing
# the primary there would make homework unable to use it.
_TIER0_IDS = {
    "qwen2.5:3b",
}


def _catalog_entry(model_id: str) -> Dict[str, str]:
    return {
        "id": model_id,
        "adapter": _adapter_for(model_id),
        "licence": _licence_for(model_id),
    }


def build_model_catalog(allowlist: set) -> Dict[str, List[Dict[str, str]]]:
    """Generate the tier -> models catalog from the allowlist.

    Every allowlisted id appears in exactly one tier:
      - tier0: cheap, always-available models (Proprietary primary + offline)
      - tier3: high-capability models for high-risk use-cases
      - open_frontier: everything else (open-weight, lighter GPAI duties)

    Unknown/env-added ids land in open_frontier so the catalog always
    covers the full allowlist (catalog ids == allowlist, by construction).
    """
    catalog: Dict[str, List[Dict[str, str]]] = {
        "tier0": [],
        "tier3": [],
        "open_frontier": [],
    }
    for model_id in sorted(allowlist):
        if model_id in _TIER0_IDS:
            catalog["tier0"].append(_catalog_entry(model_id))
        elif model_id in _TIER3_IDS:
            catalog["tier3"].append(_catalog_entry(model_id))
        else:
            catalog["open_frontier"].append(_catalog_entry(model_id))
    return catalog


_ALLOWLIST_IDS = _load_allowlist_ids()

MODEL_CATALOG: Dict[str, List[Dict[str, str]]] = build_model_catalog(_ALLOWLIST_IDS)

# Fail closed at import time if the generated catalog ever drifts from the
# allowlist — this assertion is the single-authority invariant itself.
_catalog_ids = {m["id"] for tier in MODEL_CATALOG.values() for m in tier}
assert _catalog_ids == set(_ALLOWLIST_IDS), (
    "MODEL_CATALOG drifted from the model_security allowlist: "
    f"{_catalog_ids.symmetric_difference(set(_ALLOWLIST_IDS))}"
)


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