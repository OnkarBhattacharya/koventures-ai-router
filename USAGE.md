# Standalone Usage Guide — `koventures-ai-router`

Use this guide to install `koventures-ai-router` 0.2.0 in another project. The package is proprietary to Accolaide and is not on PyPI. Runtime dependency: `headroom-ai`.

## Installation

### Option A — Install directly from GitHub (recommended for personal/private repos)

```bash
pip install "koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git"
```

In `requirements.txt`:

```text
koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git
```

This installs the package directly from the tagged source. No registry or publish step is required.

### Option B — Install from GitHub Packages (when available)

If GitHub Packages is enabled for the repo, add the index URL:

```text
--index-url https://python.pkg.github.com/OnkarBhattacharya/simple/
--extra-index-url https://pypi.org/simple/
koventures-ai-router==0.2.0
```

## Extras

```bash
# Flask API governance helpers
pip install "koventures-ai-router[governance-extra]"

# Local model inference adapters
pip install "koventures-ai-router[adapters]"
```

## Quick Start

```python
from koventures_ai_router import ModelRouter, UseCase
from koventures_ai_router.governance import AIGovernanceManager, AuditLogger, ComplianceAuditTrail

# 1. Initialise governance
gov = AIGovernanceManager()
audit = ComplianceAuditTrail(AuditLogger())

# 2. Create router
router = ModelRouter(governance=gov, audit_trail=audit)

# 3. Route a request
decision = router.route(
    use_case=UseCase.OPEN_CHAT,
    user_id="user-123",
    input_data={"user_message": "Explain photosynthesis"},
)

print(decision.model_id, decision.decision_id)
```

## Public API

The package re-exports its public surface from the top-level module:

```python
from koventures_ai_router import (
    # Router
    ModelRouter,
    UseCase,
    UseCasePolicy,
    DataSensitivity,
    RoutingDecision,
    NoEligibleModelError,
    init_governance,
    resolve,
    route,
    # Adapters
    DeepSeekV4Client,
    GeminiFlashClient,
    GLM52Client,
    MiniMaxM3Client,
    OpenRouterClient,
    OllamaClient,
    # Governance
    AIGovernanceManager,
    AuditLogger,
    ComplianceAuditTrail,
    AIDecisionLog,
    AIDecisionType,
    AuditEvent,
    AuditEventType,
    ModelCard,
    ModelRiskLevel,
    ExplainabilityProvider,
    SeverityLevel,
    # Classifier
    RiskWizard,
    RiskAssessmentResult,
    # Doc generation
    AnnexIVGenerator,
    AnnexIVPack,
)
```

## Core Concepts

### Use Cases

Declare what you are building so the router can apply the correct policy:

```python
class UseCase(Enum):
    OPEN_CHAT = "open_chat"
    CONTENT_GENERATION = "content_generation"
    ASSESSMENT_SCORING = "assessment_scoring"
    ADMISSION_SCREENING = "admission_screening"
    PROCTORING_MONITORING = "proctoring_monitoring"
```

### Governance

Every routed decision is logged automatically. Wire singletons once at startup:

```python
from koventures_ai_router.router import init_governance

init_governance(gov=gov, audit=audit)
```

### Risk Wizard

Classify your feature before routing:

```python
from koventures_ai_router.classifier import RiskWizard

wizard = RiskWizard()
result = wizard.assess(
    feature_name="Student grade prediction",
    answers={"education_access": "yes"},
)
print(result.risk_tier)  # ModelRiskLevel.HIGH
```

## Versioning

- Versions follow SemVer. The synced package is `0.2.0`.
- Pin to a tag in production:
  ```text
  koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git
  ```
- Track `main` for bleeding-edge development:
  ```text
  koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git@main
  ```

## Updating

```bash
# Update to latest tag
pip install --upgrade "koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git"

# Or pin to a newer tag
pip install "koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git@v0.2.0"
```

## Dependencies

- Core package: zero required dependencies
- Flask is optional (`[governance-extra]`)
- Torch/transformers are optional (`[adapters]`)

## Requirements

- Python >= 3.11

## Troubleshooting

**ImportError:** Ensure you installed the package and your `PYTHONPATH` is not shadowing it with an older local `koventures_ai_router/` directory.

**Git install fails:** Check network access to GitHub. In restricted environments, vendor the package or use an internal mirror.

**Flask missing:** `APIGovernanceManager` is importable without Flask, but `create_api_protection_decorator` requires Flask at runtime. Install `[governance-extra]` if you need it.

## Repository

- Source: https://github.com/OnkarBhattacharya/koventures-ai-router
- Issues: https://github.com/OnkarBhattacharya/koventures-ai-router/issues
