"""Router package: policy, adapters, and generation routing."""

from .adapters import (
    DeepSeekV4Client,
    GLM52Client,
    HFHostedAdapter,
    MiniMaxM3Client,
)
from .policy import (
    MODEL_CATALOG,
    DataSensitivity,
    UseCase,
    UseCasePolicy,
    resolve,
)
from .router import (
    ModelRouter,
    NoEligibleModelError,
    RoutingDecision,
    init_governance,
    route,
)

__all__ = [
    "DataSensitivity",
    "DeepSeekV4Client",
    "GLM52Client",
    "HFHostedAdapter",
    "MiniMaxM3Client",
    "ModelRouter",
    "NoEligibleModelError",
    "RoutingDecision",
    "UseCase",
    "UseCasePolicy",
    "init_governance",
    "resolve",
    "route",
    "MODEL_CATALOG",
]
