"""Tests for accolaide_ai_router.classifier"""

import pytest

from accolaide_ai_router.classifier import RiskAssessmentResult, RiskWizard
from accolaide_ai_router.governance import ModelRiskLevel


def test_risk_wizard_all_no_returns_low_risk():
    wizard = RiskWizard()
    result = wizard.assess(
        feature_name="Trivial feature",
        answers={
            "critical_infrastructure": "no",
            "education_access": "no",
            "employment_access": "no",
            "essential_services": "no",
            "law_enforcement": "no",
            "administration_of_justice": "no",
            "biometric_remote": "no",
            "emotion_recognition": "no",
            "social_scoring": "no",
        },
    )
    assert result.risk_tier.value == "minimal_risk"
    assert result.score == pytest.approx(0.0)
    assert result.rationale


def test_risk_wizard_high_risk_answers_return_high_risk():
    wizard = RiskWizard()
    result = wizard.assess(
        feature_name="Student grade prediction",
        answers={
            "critical_infrastructure": "no",
            "education_access": "yes",
            "employment_access": "no",
            "essential_services": "no",
            "law_enforcement": "no",
            "administration_of_justice": "no",
            "biometric_remote": "no",
            "emotion_recognition": "no",
            "social_scoring": "no",
        },
    )
    assert result.risk_tier.value == "high_risk"
    assert result.score > 0.5
    assert result.flags == ["education_access"]


def test_risk_assessment_result_optional_remediation():
    result = RiskAssessmentResult(
        risk_tier=ModelRiskLevel.MINIMAL,
        score=0.0,
        flags=[],
        rationale="test",
        remediation="fix it",
    )
    assert result.remediation == "fix it"
