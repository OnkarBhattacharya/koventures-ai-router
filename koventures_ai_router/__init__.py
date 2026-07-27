"""KOVentures AI Router — policy-aware model routing with governance."""

__version__ = "0.1.0"

from .classifier import RiskAssessmentResult, RiskWizard
from .docgen import AnnexIVGenerator, AnnexIVPack
from .governance import (
    AIDecisionLog,
    AIDecisionType,
    AIGovernanceManager,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    ComplianceAuditTrail,
    ExplainabilityProvider,
    ModelCard,
    ModelRiskLevel,
    SeverityLevel,
)
from .router import (
    DataSensitivity,
    DeepSeekV4Client,
    GLM52Client,
    HFHostedAdapter,
    MiniMaxM3Client,
    ModelRouter,
    NoEligibleModelError,
    RoutingDecision,
    UseCase,
    UseCasePolicy,
    init_governance,
    resolve,
    route,
)

__all__ = [
    "__version__",
    "AnnexIVGenerator",
    "AnnexIVPack",
    "AIDecisionLog",
    "AIDecisionType",
    "AIGovernanceManager",
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "ComplianceAuditTrail",
    "DataSensitivity",
    "DeepSeekV4Client",
    "ExplainabilityProvider",
    "GLM52Client",
    "HFHostedAdapter",
    "MiniMaxM3Client",
    "ModelCard",
    "ModelRiskLevel",
    "ModelRouter",
    "NoEligibleModelError",
    "RiskAssessmentResult",
    "RiskWizard",
    "RoutingDecision",
    "SeverityLevel",
    "UseCase",
    "UseCasePolicy",
    "init_governance",
    "resolve",
    "route",
]
