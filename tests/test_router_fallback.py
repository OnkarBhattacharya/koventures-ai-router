"""Tests for ModelRouter.generate_with_fallback and record_output (adapters monkeypatched)."""

import pytest

from koventures_ai_router.governance import AIGovernanceManager, AuditLogger, ComplianceAuditTrail
from koventures_ai_router.router import ModelRouter, UseCase
from koventures_ai_router.router.router import CircuitBreaker


@pytest.fixture()
def router():
    gov = AIGovernanceManager()
    audit = ComplianceAuditTrail(AuditLogger())
    return ModelRouter(governance=gov, audit_trail=audit)


class _StubAdapter:
    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc
        self.calls = 0

    def generate(self, **_):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.text


def test_generate_with_fallback_first_choice_success(router, monkeypatch):
    stub = _StubAdapter(text="hello")
    monkeypatch.setattr(
        "koventures_ai_router.router.router._get_adapter", lambda model_id: stub
    )
    result = router.generate_with_fallback(
        use_case=UseCase.OPEN_CHAT,
        user_id="user-1",
        input_data={"user_message": "hi"},
        system_prompt="sys",
    )
    assert result["text"] == "hello"
    assert result["fallback_used"] is False
    assert result["decision_id"] is not None
    assert stub.calls == 1


def test_generate_with_fallback_falls_back_to_next_candidate(router, monkeypatch):
    state = {"first": True}

    def get_adapter(model_id):
        if state["first"]:
            state["first"] = False
            return _StubAdapter(text=None)  # returns no text -> counts as failure
        return _StubAdapter(text="from-fallback")

    monkeypatch.setattr("koventures_ai_router.router.router._get_adapter", get_adapter)
    result = router.generate_with_fallback(
        use_case=UseCase.OPEN_CHAT,
        user_id="user-1",
        input_data={"user_message": "hi"},
        system_prompt="sys",
    )
    assert result["text"] == "from-fallback"
    assert result["fallback_used"] is True
    assert state["first"] is False


def test_generate_with_fallback_skips_open_circuit(router, monkeypatch):
    router.circuit_breaker.record_failure("glm-5.2")
    router.circuit_breaker.record_failure("glm-5.2")
    router.circuit_breaker.record_failure("glm-5.2")  # threshold=3 -> OPEN

    seen = []

    def get_adapter(model_id):
        seen.append(model_id)
        return _StubAdapter(text="ok")

    monkeypatch.setattr("koventures_ai_router.router.router._get_adapter", get_adapter)
    result = router.generate_with_fallback(
        use_case=UseCase.OPEN_CHAT,
        user_id="user-1",
        input_data={"user_message": "hi"},
        system_prompt="sys",
    )
    assert "glm-5.2" not in seen
    assert result["text"] == "ok"
    assert result["model_id"] != "glm-5.2"


def test_record_output_attaches_text_to_decision(router, monkeypatch):
    stub = _StubAdapter(text="answer")
    monkeypatch.setattr(
        "koventures_ai_router.router.router._get_adapter", lambda model_id: stub
    )
    decision = router.route(
        use_case=UseCase.OPEN_CHAT,
        user_id="user-1",
        input_data={"user_message": "q"},
    )
    router.record_output(decision, output="answer")
    record = next(
        r
        for r in router.governance.decision_log.decisions
        if r["id"] == decision.decision_id
    )
    assert record["output"] == "answer"


def test_circuit_breaker_reopens_when_half_open_probe_fails():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.01)
    cb.record_failure("m")
    import time

    time.sleep(0.02)
    assert cb.state("m") == CircuitBreaker.HALF_OPEN
    assert cb.allow_request("m") is True
    cb.record_failure("m")
    assert cb.state("m") == CircuitBreaker.OPEN
