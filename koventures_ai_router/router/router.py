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
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from headroom import CompressConfig, compress, CompressResult

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
    # Headroom fields (optional)
    headroom_tokens_saved: Optional[int] = None
    headroom_compression_ratio: Optional[float] = None
    # Output Headroom fields (optional)
    headroom_output_tokens_before: Optional[int] = None
    headroom_output_tokens_after: Optional[int] = None
    headroom_output_tokens_saved: Optional[int] = None
    headroom_output_compression_ratio: Optional[float] = None


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
        # Headroom integration
        self.headroom = None
        if os.environ.get("HEADROOM_ENABLED", "false").lower() == "true":
            # Hard-code beacon off as required
            os.environ["HEADROOM_BEACON"] = "off"
            # Initialize Headroom compression config with defaults
            # compress_user_messages=True so student/teacher questions get compressed
            self.headroom = CompressConfig(compress_user_messages=True)

    def _candidates(
        self,
        policy: UseCasePolicy,
        preferred_tier: Optional[str] = None,
        preferred_model_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        candidates: List[Dict[str, str]] = []
        for tier in policy.allowed_model_tiers:
            for model in self.model_catalog.get(tier, []):
                candidates.append({**model, "_policy_tier": tier})
        if self.prefer_open_weight:
            # MIT-licensed open-weight models first: lighter upstream GPAI
            # obligations (Art. 53(2)) and, usually, materially cheaper.
            candidates.sort(key=lambda m: 0 if m.get("licence") == "MIT" else 1)
        # Input constraints from the vertical's cost/routing layer (plan task
        # 1.1). These only REORDER policy-filtered candidates — policy stays
        # the authority on which models are eligible at all.
        if preferred_model_id:
            preferred = [c for c in candidates if c["id"] == preferred_model_id]
            rest = [c for c in candidates if c["id"] != preferred_model_id]
            candidates = preferred + rest
        elif preferred_tier:
            tier_first = [c for c in candidates if c.get("_policy_tier") == preferred_tier]
            rest = [c for c in candidates if c.get("_policy_tier") != preferred_tier]
            candidates = tier_first + rest
        return candidates

    def route(
        self,
        *,
        use_case: UseCase,
        user_id: str,
        input_data: Dict[str, Any],
        override_model_id: Optional[str] = None,
        preferred_tier: Optional[str] = None,
        preferred_model_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> RoutingDecision:
        """Resolve policy, choose a model, and log the routing decision.

        ``output`` is not known yet at routing time — call
        ``record_output()`` after generation completes to attach it to
        the same decision record (needed for GDPR Art. 22 explanation
        rights and for the Annex IV evidence pack).

        ``preferred_tier`` / ``preferred_model_id`` / ``intent`` are input
        constraints from the vertical's cost layer; they may only reorder
        the policy-filtered candidates.
        """
        policy = resolve(use_case)
        candidates = self._candidates(
            policy,
            preferred_tier=preferred_tier,
            preferred_model_id=preferred_model_id,
        )

        if override_model_id:
            candidates = [c for c in candidates if c["id"] == override_model_id]

        if not candidates:
            raise NoEligibleModelError(
                f"No model available for use-case={use_case.value!r} "
                f"under current policy (allowed tiers: {policy.allowed_model_tiers})"
            )

        chosen = candidates[0]

        log_input = dict(input_data)
        if intent:
            log_input.setdefault("intent", intent)

        decision_id = self.governance.log_ai_decision(
            user_id=user_id,
            decision_type=policy.decision_type,
            input_data=log_input,
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
                "intent": intent,
                "preferred_tier": preferred_tier,
                "preferred_model_id": preferred_model_id,
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
        changes: Dict[str, Any] = {"output": output}
        if decision.headroom_tokens_saved is not None:
            changes["headroom_tokens_saved"] = decision.headroom_tokens_saved
        if decision.headroom_compression_ratio is not None:
            changes["headroom_compression_ratio"] = decision.headroom_compression_ratio
        if decision.headroom_output_tokens_before is not None:
            changes["headroom_output_tokens_before"] = decision.headroom_output_tokens_before
        if decision.headroom_output_tokens_after is not None:
            changes["headroom_output_tokens_after"] = decision.headroom_output_tokens_after
        if decision.headroom_output_tokens_saved is not None:
            changes["headroom_output_tokens_saved"] = decision.headroom_output_tokens_saved
        if decision.headroom_output_compression_ratio is not None:
            changes["headroom_output_compression_ratio"] = decision.headroom_output_compression_ratio
        self.governance.decision_log.update_decision(decision.decision_id, changes)

    def generate_with_fallback(
        self,
        *,
        use_case: UseCase,
        user_id: str,
        input_data: Dict[str, Any],
        system_prompt: str,
        max_length: int = 300,
        override_model_id: Optional[str] = None,
        preferred_tier: Optional[str] = None,
        preferred_model_id: Optional[str] = None,
        intent: Optional[str] = None,
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
        candidates = self._candidates(
            policy,
            preferred_tier=preferred_tier,
            preferred_model_id=preferred_model_id,
        )
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
                intent=intent,
            )

            # Headroom compression
            compressed_input_data = input_data
            compressed_system_prompt = system_prompt
            original_tokens: Optional[int] = None
            compressed_tokens: Optional[int] = None
            if self.headroom is not None:
                try:
                    # Build messages for compression: system then user
                    messages = [
                        {"role": "system", "content": system_prompt}
                    ]
                    user_content = input_data.get("user_message", "")
                    # If there's image_data, we could handle it, but for now ignore
                    if "image_data" in input_data:
                        # For simplicity, we'll just note that image data is present
                        user_content += " [Image data present]"
                    messages.append({"role": "user", "content": user_content})
                    
                    compress_result: CompressResult = compress(messages, config=self.headroom)
                    original_tokens = compress_result.tokens_before
                    compressed_tokens = compress_result.tokens_after
                    
                    # Extract compressed content
                    # Find system and user messages in compressed result
                    compressed_system = ""
                    compressed_user = ""
                    for msg in compress_result.messages:
                        if msg.get("role") == "system":
                            compressed_system = msg.get("content", "")
                        elif msg.get("role") == "user":
                            compressed_user = msg.get("content", "")
                    
                    # If compression didn't change role structure, fallback to original
                    if compressed_system and compressed_user:
                        compressed_system_prompt = compressed_system
                        # Update input_data with compressed user message
                        compressed_input_data = dict(input_data)
                        compressed_input_data["user_message"] = compressed_user
                        # Preserve image_data if present
                        if "image_data" in input_data:
                            compressed_input_data["image_data"] = input_data["image_data"]
                    else:
                        # Fallback to original if compression didn't produce expected roles
                        logger.warning(
                            "Headroom compression did not produce expected system/user roles",
                            extra={"original_tokens": original_tokens, "compressed_tokens": compressed_tokens}
                        )
                except Exception as e:
                    logger.warning(
                        "Headroom compression failed, using original content",
                        extra={"error": str(e)}
                    )
                    # Use original content on failure (fail open)
                    compressed_system_prompt = system_prompt
                    compressed_input_data = input_data
                    # Keep original_tokens and compressed_tokens as None to indicate failure

            try:
                adapter = _get_adapter(model_id)
                text = adapter.generate(
                    user_message=compressed_input_data.get("user_message", ""),
                    system_prompt=compressed_system_prompt,
                    max_length=max_length,
                    image_data=compressed_input_data.get("image_data"),
                )
                # Headroom output compression (for metrics only - don't alter user-facing output)
                output_tokens_before = None
                output_tokens_after = None
                if self.headroom is not None:
                    try:
                        # Compress the output for token measurement
                        output_messages = [{"role": "assistant", "content": text}]
                        compress_result: CompressResult = compress(output_messages, config=self.headroom)
                        output_tokens_before = compress_result.tokens_before
                        output_tokens_after = compress_result.tokens_after
                    except Exception as e:
                        logger.warning(
                            "Headroom output compression failed",
                            extra={"error": str(e)}
                        )
                if text:
                    self.circuit_breaker.record_success(model_id)
                    # Set output Headroom metrics on decision for persistence
                    decision.headroom_output_tokens_before = output_tokens_before
                    decision.headroom_output_tokens_after = output_tokens_after
                    if output_tokens_before is not None and output_tokens_after is not None:
                        decision.headroom_output_tokens_saved = output_tokens_before - output_tokens_after
                        if output_tokens_before > 0:
                            decision.headroom_output_compression_ratio = output_tokens_after / output_tokens_before
                        else:
                            decision.headroom_output_compression_ratio = 0.0
                    
                    self.record_output(decision, output=text)
                    result = {
                        "text": text,
                        "decision_id": decision.decision_id,
                        "model_id": model_id,
                        "adapter": adapter_name,
                        "fallback_used": fallback_used,
                    }
                    # Thread token delta into result only if compression succeeded
                    if self.headroom is not None and original_tokens is not None and compressed_tokens is not None:
                        result["headroom_tokens_saved"] = original_tokens - compressed_tokens
                        if original_tokens > 0:
                            result["headroom_compression_ratio"] = compressed_tokens / original_tokens
                        else:
                            result["headroom_compression_ratio"] = 0.0
                    
                    # Thread output token metrics into result
                    if self.headroom is not None and output_tokens_before is not None and output_tokens_after is not None:
                        result["headroom_output_tokens_before"] = output_tokens_before
                        result["headroom_output_tokens_after"] = output_tokens_after
                        result["headroom_output_tokens_saved"] = output_tokens_before - output_tokens_after
                        if output_tokens_before > 0:
                            result["headroom_output_compression_ratio"] = output_tokens_after / output_tokens_before
                        else:
                            result["headroom_output_compression_ratio"] = 0.0
                    return result
                last_error = RuntimeError(f"Adapter {adapter_name} returned no text")
            except NoEligibleModelError:
                # Unknown model id: re-raise so route_and_generate maps it to
                # the safe error response (plan task 2.3, Q6) instead of
                # counting it as an ordinary adapter failure.
                raise
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


    def route_and_generate(
        self,
        *,
        use_case: UseCase,
        user_id: str,
        input_data: Dict[str, Any],
        system_prompt: str,
        max_length: int = 300,
        override_model_id: Optional[str] = None,
        preferred_tier: Optional[str] = None,
        preferred_model_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Single primitive for a generation request (plan task 1.1).

        ``route()`` → per eligible model in policy order:
        ``circuit_breaker.allow_request()`` → adapter ``generate()`` →
        on failure ``record_failure()`` and try the next candidate → on
        success ``record_output(decision, output)``.

        This is the entrypoint the live chat path must use; the vertical's
        cost/routing layer feeds ``preferred_tier`` / ``preferred_model_id`` /
        ``intent`` as input constraints — it must never bypass this method.
        """
        try:
            return self.generate_with_fallback(
                use_case=use_case,
                user_id=user_id,
                input_data=input_data,
                system_prompt=system_prompt,
                max_length=max_length,
                override_model_id=override_model_id,
                preferred_tier=preferred_tier,
                preferred_model_id=preferred_model_id,
                intent=intent,
            )
        except NoEligibleModelError as exc:
            # Q6 (plan task 2.3): an unknown / non-allowlisted model id is a
            # safe error response, same pattern as the all-candidates-fail
            # path — never an HTTP 400 (that is a Phase 6 blueprint concern).
            logger.error(
                "no_eligible_model",
                extra={"use_case": use_case.value, "error": str(exc)},
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
    """Instantiate the correct adapter for a model ID.

    Fails closed: an unknown model id raises NoEligibleModelError instead of
    silently defaulting to gemini_flash (plan task 2.3, F5) — a silent
    fallback would route traffic to a model the operator never allowlisted.
    """
    adapter_name = _CATALOG_FLAT.get(model_id)
    if adapter_name is None:
        raise NoEligibleModelError(
            f"Model {model_id!r} is not in the MODEL_CATALOG (which is generated "
            "from the model_security allowlist); refusing to fall back silently."
        )
    if adapter_name == "gemini_flash":
        from .adapters.gemini_flash import GeminiFlashClient
        return GeminiFlashClient()
    if adapter_name == "openrouter":
        from .adapters.openrouter import OpenRouterClient
        # Pass the chosen model id through so each chain link (gemma primary,
        # llama-3.3 fallback, nemotron last resort) actually hits OpenRouter
        # with the intended model instead of the adapter default.
        return OpenRouterClient(model=model_id)
    if adapter_name == "ollama":
        from .adapters.ollama import OllamaClient
        # Pass the chosen model id through so offline fallbacks (phi4-mini,
        # qwen3:4b, qwen2.5:3b, ...) actually hit the intended Ollama model
        # instead of silently defaulting to qwen2.5:3b. Respect OLLAMA_BASE_URL
        # so self-hosted/sidecar Ollama hosts work (not just localhost).
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=model_id,
        )
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
    """Install the app-wide governance singletons (plan task 2.2, F3).

    Idempotent for the same instances; raises if a *different* governance
    manager or audit trail is installed while one is already live — two
    live AIGovernanceManagers would mean two divergent decision logs and
    two audit trails, which is exactly the governance split we must not
    reintroduce. Use :func:`reset_governance` in tests to clear.
    """
    global _gov, _audit
    if (_gov is not None or _audit is not None) and (gov is not _gov or audit is not _audit):
        raise RuntimeError(
            "init_governance() already called with different instances; "
            "there must be exactly one app-wide AIGovernanceManager / "
            "ComplianceAuditTrail. Use reset_governance() if you really "
            "need to replace them (tests only)."
        )
    _gov = gov
    _audit = audit


def reset_governance() -> None:
    """Clear the module-level governance singletons (tests only)."""
    global _gov, _audit
    _gov = None
    _audit = None


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
