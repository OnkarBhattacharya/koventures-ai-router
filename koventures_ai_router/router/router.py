"""Model router.

Every generation request passes through `ModelRouter.route()` before it
reaches a provider adapter. That single choke point is what makes the
audit trail and governance log a byproduct of normal operation rather
than a step someone has to remember to add later — the same pattern
the existing `ai_governance.py` docstring already points at with its
"GDPR Article 22" and "AI Act" comments, just not yet wired to anything
upstream of it.

This module deliberately knows nothing about education, HR, or any
other vertical. It only knows: use-case in, policy out, model chosen,
decision logged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..governance.ai_governance import AIDecisionType, AIGovernanceManager
from ..governance.audit_logging import ComplianceAuditTrail
from .policy import MODEL_CATALOG, UseCase, UseCasePolicy, resolve

logger = logging.getLogger(__name__)


class NoEligibleModelError(RuntimeError):
    """Raised when policy allows no model for a use-case (e.g. every
    model in the allowed tiers has been disabled/deprecated)."""


class CircuitBreaker:
    """Track consecutive failures per model and block traffic while open.

    States
    ------
    CLOSED  — normal operation, failures are counted.
    OPEN    — too many recent failures; calls are rejected immediately.
    HALF_OPEN — cooldown elapsed; one trial call is allowed to probe
                whether the downstream has recovered.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
        half_open_max_probes: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_max_probes = half_open_max_probes
        self._states: Dict[str, str] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_failure_time: Dict[str, float] = {}
        self._half_open_probes: Dict[str, int] = {}

    def state(self, model_id: str) -> str:
        if model_id not in self._states:
            self._states[model_id] = self.CLOSED
        if self._states[model_id] == self.OPEN:
            last_fail = self._last_failure_time.get(model_id, 0)
            if time.time() - last_fail >= self.recovery_timeout:
                self._states[model_id] = self.HALF_OPEN
                self._half_open_probes[model_id] = 0
        return self._states[model_id]

    def allow_request(self, model_id: str) -> bool:
        s = self.state(model_id)
        if s == self.CLOSED:
            return True
        if s == self.HALF_OPEN:
            probes = self._half_open_probes.get(model_id, 0)
            if probes < self.half_open_max_probes:
                self._half_open_probes[model_id] = probes + 1
                return True
            return False
        return False

    def record_success(self, model_id: str) -> None:
        self._failure_counts.pop(model_id, None)
        self._states[model_id] = self.CLOSED
        self._half_open_probes.pop(model_id, None)

    def record_failure(self, model_id: str) -> None:
        s = self.state(model_id)
        if s == self.HALF_OPEN:
            self._states[model_id] = self.OPEN
            self._last_failure_time[model_id] = time.time()
            return
        count = self._failure_counts.get(model_id, 0) + 1
        self._failure_counts[model_id] = count
        if count >= self.failure_threshold:
            self._states[model_id] = self.OPEN
            self._last_failure_time[model_id] = time.time()

    def reset(self, model_id: str) -> None:
        self._failure_counts.pop(model_id, None)
        self._states.pop(model_id, None)
        self._last_failure_time.pop(model_id, None)
        self._half_open_probes.pop(model_id, None)


@dataclass
class RoutingDecision:
    use_case: UseCase
    model_id: str
    adapter: str
    decision_id: str
    human_review_required: bool
    risk_level: str


class ModelRouter:
    """Choke point for every generation request.

    Wires governance logging + audit trail as a byproduct of normal
    operation, so compliance is never an afterthought.
    """

    def __init__(
        self,
        governance: AIGovernanceManager,
        audit_trail: ComplianceAuditTrail,
        model_catalog: Optional[Dict[str, List[Dict[str, str]]]] = None,
        prefer_open_weight: bool = True,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.governance = governance
        self.audit_trail = audit_trail
        self.model_catalog = model_catalog or MODEL_CATALOG
        self.prefer_open_weight = prefer_open_weight
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def _candidates(self, policy: UseCasePolicy) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        for tier in policy.allowed_model_tiers:
            candidates.extend(self.model_catalog.get(tier, []))
        if self.prefer_open_weight:
            # MIT-licensed open-weight models first: lighter upstream GPAI
            # obligations (Art. 53(2)) and, usually, materially cheaper.
            candidates.sort(key=lambda m: 0 if m.get("licence") == "MIT" else 1)
        return candidates

    def route(
        self,
        *,
        use_case: UseCase,
        user_id: str,
        input_data: Dict[str, Any],
        override_model_id: Optional[str] = None,
    ) -> RoutingDecision:
        """Resolve policy, choose a model, and log the routing decision.

        ``output`` is not known yet at routing time — call
        ``record_output()`` after generation completes to attach it to
        the same decision record (needed for GDPR Art. 22 explanation
        rights and for the Annex IV evidence pack).
        """
        policy = resolve(use_case)
        candidates = self._candidates(policy)

        if override_model_id:
            candidates = [c for c in candidates if c["id"] == override_model_id]

        if not candidates:
            raise NoEligibleModelError(
                f"No model available for use-case={use_case.value!r} "
                f"under current policy (allowed tiers: {policy.allowed_model_tiers})"
            )

        chosen = candidates[0]

        decision_id = self.governance.log_ai_decision(
            user_id=user_id,
            decision_type=policy.decision_type,
            input_data=input_data,
            output=None,
            model_id=chosen["id"],
        )

        self.audit_trail.log_ai_act_compliance_action(
            action="model_routed",
            model_id=chosen["id"],
            details={
                "use_case": use_case.value,
                "risk_level": policy.risk_level.value,
                "human_review_required": policy.human_review_required,
                "candidates_considered": [c["id"] for c in candidates],
            },
        )

        return RoutingDecision(
            use_case=use_case,
            model_id=chosen["id"],
            adapter=chosen["adapter"],
            decision_id=decision_id,
            human_review_required=policy.human_review_required,
            risk_level=policy.risk_level.value,
        )

    def record_output(self, decision: RoutingDecision, output: Any) -> None:
        """Attach the generated output to the decision already logged
        by ``route()``. Keeps ``ai_governance.py`` as the single source of
        truth for what a model actually produced, without needing the
        router to duplicate storage."""
        for record in self.governance.decision_log.decisions:
            if record["id"] == decision.decision_id:
                record["output"] = output
                return

    def generate_with_fallback(
        self,
        *,
        use_case: UseCase,
        user_id: str,
        input_data: Dict[str, Any],
        system_prompt: str,
        max_length: int = 300,
        override_model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route, generate with circuit-breaker protection, and fall back.

        Returns a dict with:
          - text: the model response (or a degraded fallback string)
          - decision_id: governance decision log id (or None)
          - model_id: the model that produced the response
          - adapter: the adapter used
          - fallback_used: bool, True if we fell back from the first choice
        """
        policy = resolve(use_case)
        candidates = self._candidates(policy)
        if override_model_id:
            candidates = [c for c in candidates if c["id"] == override_model_id]

        if not candidates:
            raise NoEligibleModelError(
                f"No model available for use-case={use_case.value!r}"
            )

        last_error: Optional[Exception] = None
        fallback_used = False

        for candidate in candidates:
            model_id = candidate["id"]
            adapter_name = candidate["adapter"]

            if not self.circuit_breaker.allow_request(model_id):
                logger.warning(
                    "circuit_open", extra={"model_id": model_id, "adapter": adapter_name}
                )
                last_error = RuntimeError(f"Circuit open for {model_id}")
                continue

            decision = self.route(
                use_case=use_case,
                user_id=user_id,
                input_data=input_data,
                override_model_id=model_id,
            )

            try:
                adapter = _get_adapter(model_id)
                text = adapter.generate(
                    user_message=input_data.get("user_message", ""),
                    system_prompt=system_prompt,
                    max_length=max_length,
                    image_data=input_data.get("image_data"),
                )
                if text:
                    self.circuit_breaker.record_success(model_id)
                    self.record_output(decision, output=text)
                    return {
                        "text": text,
                        "decision_id": decision.decision_id,
                        "model_id": model_id,
                        "adapter": adapter_name,
                        "fallback_used": fallback_used,
                    }
                last_error = RuntimeError(f"Adapter {adapter_name} returned no text")
            except Exception as exc:
                last_error = exc
                logger.error(
                    "adapter_generation_failed",
                    extra={"model_id": model_id, "adapter": adapter_name, "error": str(exc)},
                )

            self.circuit_breaker.record_failure(model_id)
            fallback_used = True

        logger.error(
            "all_candidates_failed",
            extra={"use_case": use_case.value, "last_error": str(last_error)},
        )
        return {
            "text": (
                "I'm sorry, I couldn't generate a response right now. "
                "The AI service is temporarily unavailable."
            ),
            "decision_id": None,
            "model_id": None,
            "adapter": None,
            "fallback_used": True,
        }


# ---------------------------------------------------------------------------
# Adapter resolution (extracted from old policy.py)
# ---------------------------------------------------------------------------

_CATALOG_FLAT: Dict[str, str] = {}
for _tier, _models in MODEL_CATALOG.items():
    for _m in _models:
        _CATALOG_FLAT[_m["id"]] = _m["adapter"]


def _get_adapter(model_id: str):
    """Instantiate the correct adapter for a model ID."""
    adapter_name = _CATALOG_FLAT.get(model_id, "hf_hosted")
    if adapter_name == "hf_hosted":
        from .adapters.hf_hosted import HFHostedAdapter
        return HFHostedAdapter()
    if adapter_name == "glm_5_2":
        from .adapters.glm_5_2 import GLM52Client
        return GLM52Client()
    if adapter_name == "deepseek_v4":
        from .adapters.deepseek_v4 import DeepSeekV4Client
        return DeepSeekV4Client()
    if adapter_name == "minimax_m3":
        from .adapters.minimax_m3 import MiniMaxM3Client
        return MiniMaxM3Client()
    raise ValueError(f"No adapter registered for {adapter_name!r}")


# ---------------------------------------------------------------------------
# Module-level singletons for governance wiring (lazy to avoid circular imports).
# ---------------------------------------------------------------------------
_gov: Optional[AIGovernanceManager] = None
_audit: Optional[ComplianceAuditTrail] = None


def init_governance(
    gov: Optional[AIGovernanceManager] = None,
    audit: Optional[ComplianceAuditTrail] = None,
) -> None:
    global _gov, _audit
    _gov = gov
    _audit = audit


def route(
    *,
    model_id: str,
    use_case: UseCase,
    user_id: str,
    input_data: Dict[str, Any],
    system_prompt: str = (
        "You are a helpful educational AI assistant for school students aged 8-18. "
        "Provide clear, accurate, curriculum-appropriate answers. "
        "Explain concepts step by step in a way students can understand. "
        "Guide students to understanding rather than just giving answers. "
        "Keep responses concise and age-appropriate."
    ),
    max_length: int = 300,
    decision_type: AIDecisionType = AIDecisionType.HUMAN_INVOLVED,
) -> Dict[str, Any]:
    """Evaluate policy, run the right adapter, and return the generated text.

    .. deprecated::
        Prefer creating a ``ModelRouter`` instance and calling
        ``router.route()`` + ``router.record_output()`` separately.
        This function is a backward-compatible wrapper that does both
        in one call.

    The returned dict always contains:
      - text: the model response (or an error fallback)
      - decision_id: governance decision log id (or None)
    """
    if _gov is None or _audit is None:
        logger.warning(
            "Module-level governance not initialised. Call init_governance() "
            "before route() or use ModelRouter directly."
        )

    router = ModelRouter(
        governance=_gov or AIGovernanceManager(),
        audit_trail=_audit or ComplianceAuditTrail(  # type: ignore[arg-type]
            # AuditLogger is expected but will be a no-op singleton
            type("_NoOpAuditLogger", (), {"log_event": lambda *a, **kw: "", "events": []})()
        ),
    )

    decision = router.route(
        use_case=use_case,
        user_id=user_id,
        input_data=input_data,
        override_model_id=model_id,
    )

    text: Optional[str] = None
    try:
        adapter = _get_adapter(decision.model_id)
        text = adapter.generate(
            user_message=input_data.get("user_message", ""),
            system_prompt=system_prompt,
            max_length=max_length,
            image_data=input_data.get("image_data"),
        )
    except Exception as exc:
        logger.error("Router adapter %s failed: %s", decision.adapter, exc)

    if not text:
        text = (
            "I'm sorry, I couldn't generate a response right now. "
            "The AI service is temporarily unavailable."
        )

    router.record_output(decision, output=text)

    return {
        "text": text,
        "decision_id": decision.decision_id,
    }