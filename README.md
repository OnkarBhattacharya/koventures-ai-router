# KOVentures AI Router

**Version:** 0.2.0
**Python:** 3.11+
**License:** Proprietary — © 2026 Accolaide. All rights reserved.

Policy-aware AI model router with governance, audit logging, and EU AI Act compliance.

This tree is synced from the `koventures_ai_router` package in the OMsolutions repository. Package version 0.2.0 is the installable release of that code. It is a governance and routing layer. It is not a forecasting model and it is not a grid optimiser.

The router loads its model allowlist from `koventures_ai_router/backend/services/model_security.py`, a copy of the OMsolutions chat-app allowlist, so the package imports outside that monorepo. Inside OMsolutions the same loader still prefers `chat-app/backend/services/model_security.py` when that file is on the walk.

## Features

- Policy-enforced routing against EU AI Act risk tiers, model allowlists, and decision-type constraints
- Provider adapters: Gemini Flash, Ollama, OpenRouter, GLM-5.2, DeepSeek V4, MiniMax M3
- Circuit breaker and tier fallback
- Governance logging on every routing decision
- Annex IV evidence generation and an Annex III risk classifier
- Optional headroom context compression via `headroom-ai` when `HEADROOM_ENABLED=true`

## Installation

The package is not published on PyPI. Install from this GitHub repository:

```bash
pip install "koventures-ai-router @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git"
```

`headroom-ai` is a required dependency. Install the PyPI project `headroom-ai`, not the unrelated project named `headroom`.

Extras:

```bash
pip install "koventures-ai-router[governance-extra] @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git"
pip install "koventures-ai-router[adapters] @ git+https://github.com/OnkarBhattacharya/koventures-ai-router.git"
```

See [USAGE.md](USAGE.md) for embedding the package in another project.

## Quick start

```python
from koventures_ai_router import ModelRouter, UseCase
from koventures_ai_router.governance import AIGovernanceManager, AuditLogger, ComplianceAuditTrail

gov = AIGovernanceManager()
audit = ComplianceAuditTrail(AuditLogger())
router = ModelRouter(governance=gov, audit_trail=audit)

decision = router.route(
    use_case=UseCase.OPEN_CHAT,
    user_id="user-1",
    input_data={"user_message": "Hello!"},
)
```

Headroom compression stays off unless `HEADROOM_ENABLED=true`. When it is on, the router forces `HEADROOM_BEACON=off`.

## Layout

```
koventures_ai_router/
├── governance/
│   ├── ai_governance.py
│   ├── audit_logging.py
│   └── api_governance.py
├── router/
│   ├── policy.py
│   ├── router.py
│   └── adapters/
│       ├── gemini_flash.py
│       ├── ollama.py
│       ├── openrouter.py
│       ├── glm_5_2.py
│       ├── deepseek_v4.py
│       └── minimax_m3.py
├── docgen/annex_iv.py
└── classifier/risk_wizard.py
```

## Tests

```bash
python -m pip install -e ".[dev]"
pytest
```
