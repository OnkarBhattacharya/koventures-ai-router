"""Router package: policy, adapters, and generation routing."""

from .adapters import (
    GeminiFlashClient,
    OpenRouterClient,
    OllamaClient,
    DeepSeekV4Client,
    GLM52Client,
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
    reset_governance,
    route,
)

__all__ = [
    "DataSensitivity",
    "GeminiFlashClient",
    "OpenRouterClient",
    "OllamaClient",
    "DeepSeekV4Client",
    "GLM52Client",
    "MiniMaxM3Client",
    "ModelRouter",
    "NoEligibleModelError",
    "RoutingDecision",
    "UseCase",
    "UseCasePolicy",
    "init_governance",
    "reset_governance",
    "resolve",
    "route",
    "MODEL_CATALOG",
]
