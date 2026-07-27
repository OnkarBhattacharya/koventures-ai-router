# KOVentures AI Router

Policy-aware AI model router with governance, audit logging, and EU AI Act compliance.

## Installation

```bash
pip install koventures-ai-router
```

## Extras

```bash
pip install koventures-ai-router[governance-extra]  # Flask integration
pip install koventures-ai-router[adapters]           # Local model inference
```

## Standalone Usage in Other Projects

See **[USAGE.md](USAGE.md)** for a complete guide on installing and using this package in your own projects via pip from GitHub.

## Quick Start

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

## License

MIT
