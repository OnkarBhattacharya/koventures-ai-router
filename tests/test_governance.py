"""Tests for accolaide_ai_router.governance"""

import pytest

from accolaide_ai_router.governance import (
    AIDecisionType,
    AIGovernanceManager,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    ComplianceAuditTrail,
    ModelRiskLevel,
)
from accolaide_ai_router.governance.api_governance import RateLimiter, RateLimitStrategy


def test_ai_governance_manager_logs_decisions():
    gov = AIGovernanceManager()
    decision_id = gov.log_ai_decision(
        user_id="user-1",
        decision_type=AIDecisionType.HUMAN_INVOLVED,
        input_data={"prompt": "test"},
        output="result",
        model_id="model-a",
    )
    assert decision_id is not None
    assert len(gov.decision_log.decisions) == 1
    assert gov.decision_log.decisions[0]["user_id"] == "user-1"


def test_audit_logger_records_events():
    logger = AuditLogger()
    event_id = logger.log_event(
        event_type=AuditEventType.MODEL_PREDICTION,
        user_id="user-1",
        resource="model-a",
        action="generated_prediction",
        details={"use_case": "open_chat"},
    )
    assert len(logger.events) == 1
    assert logger.events[0].event_type == AuditEventType.MODEL_PREDICTION
    assert isinstance(event_id, str)
    assert len(event_id) > 0


def test_compliance_audit_trail_logs_actions():
    logger = AuditLogger()
    trail = ComplianceAuditTrail(logger)
    trail.log_ai_act_compliance_action(
        action="model_routed",
        model_id="model-a",
        details={"risk_level": "minimal"},
    )
    assert len(logger.events) == 1
    assert logger.events[0].details["risk_level"] == "minimal"


def test_rate_limiter_token_bucket_allows_under_limit():
    limiter = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET)
    allowed, info = limiter.is_allowed("user-1", "default")
    assert allowed is True
    assert "remaining" in info


def test_rate_limiter_token_bucket_blocks_over_limit():
    limiter = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET)
    limiter.rate_limits["default"] = {"requests": 1, "window": 3600}
    allowed, _ = limiter.is_allowed("user-1", "default")
    assert allowed is True
    allowed, info = limiter.is_allowed("user-1", "default")
    assert allowed is False
    assert "retry_after" in info
