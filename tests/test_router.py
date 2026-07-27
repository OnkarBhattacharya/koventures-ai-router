"""Tests for koventures_ai_router.router"""

import pytest

from koventures_ai_router.governance import AIGovernanceManager, AuditLogger, ComplianceAuditTrail
from koventures_ai_router.router import ModelRouter, NoEligibleModelError, UseCase
from koventures_ai_router.router.policy import resolve
from koventures_ai_router.router.router import CircuitBreaker


def test_resolve_returns_policy_for_known_use_case():
    policy = resolve(UseCase.OPEN_CHAT)
    assert policy.risk_level is not None
    assert policy.allowed_model_tiers


def test_resolve_raises_for_unknown_use_case():
    with pytest.raises(ValueError, match="No policy declared"):
        resolve("unknown_use_case")  # type: ignore[arg-type]


def test_circuit_breaker_transitions_closed_to_open():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)
    assert cb.state("model-a") == CircuitBreaker.CLOSED
    assert cb.allow_request("model-a") is True
    cb.record_failure("model-a")
    assert cb.state("model-a") == CircuitBreaker.CLOSED
    cb.record_failure("model-a")
    assert cb.state("model-a") == CircuitBreaker.OPEN
    assert cb.allow_request("model-a") is False


def test_circuit_breaker_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.05)
    cb.record_failure("model-a")
    assert cb.state("model-a") == CircuitBreaker.OPEN
    import time
    time.sleep(0.1)
    assert cb.state("model-a") == CircuitBreaker.HALF_OPEN
    assert cb.allow_request("model-a") is True


def test_model_router_raises_when_no_candidates():
    gov = AIGovernanceManager()
    audit = ComplianceAuditTrail(AuditLogger())
    router = ModelRouter(governance=gov, audit_trail=audit)
    with pytest.raises(NoEligibleModelError):
        router.route(
            use_case=UseCase.OPEN_CHAT,
            user_id="user-1",
            input_data={},
            override_model_id="nonexistent-model",
        )
