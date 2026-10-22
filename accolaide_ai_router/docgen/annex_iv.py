"""Annex IV generator: produce draft EU AI Act Annex IV compliance evidence packs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..router.policy import MODEL_CATALOG

logger = logging.getLogger(__name__)


class AnnexIVPack:
    """Container for a generated Annex IV compliance pack."""

    def __init__(
        self,
        generated_at: str,
        sections: Dict[str, Any],
        governance_report: Dict[str, Any],
        audit_trail: List[Dict[str, Any]],
    ):
        self.generated_at = generated_at
        self.sections = sections
        self.governance_report = governance_report
        self.audit_trail = audit_trail

    def to_markdown(self) -> str:
        """Render the pack as Markdown (human-readable)."""
        lines = [
            "# Annex IV Compliance Evidence Pack",
            "",
            "Generated: {}".format(self.generated_at),
            "",
            "---",
            "",
        ]

        for section_name, section_body in self.sections.items():
            lines.append("## {}".format(section_name))
            lines.append("")
            if isinstance(section_body, list):
                for item in section_body:
                    lines.append("- {}".format(item))
            elif isinstance(section_body, dict):
                for k, v in section_body.items():
                    lines.append("- **{}**: {}".format(k, v))
            else:
                lines.append(str(section_body))
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Render the pack as a plain dict (machine-readable)."""
        return {
            "pack_version": "1.0",
            "generated_at": self.generated_at,
            "sections": self.sections,
            "governance_report": self.governance_report,
            "audit_trail": self.audit_trail,
        }


class AnnexIVGenerator:
    """Generate Annex IV compliance evidence from governance and audit data."""

    def generate(
        self,
        *,
        governance_report: Dict[str, Any],
        audit_trail: List[Dict[str, Any]],
        provider_name: str = "Provider",
        system_name: str = "AI Chat System",
        intended_purposes: Optional[List[str]] = None,
        geographic_market: str = "EU",
        exceptions_claimed: Optional[List[str]] = None,
    ) -> AnnexIVPack:
        """Create a draft Annex IV evidence pack.

        Args:
            governance_report: Output of AIGovernanceManager.generate_governance_report()
            audit_trail: Output of ComplianceAuditTrail.get_compliance_audit_trail()
            provider_name: Legal name of the provider.
            system_name: Name/version string for the AI system.
            intended_purposes: List of intended use descriptions.
            geographic_market: Target market (e.g., 'EU').
            exceptions_claimed: Any transparency exceptions the provider relies on.

        Returns:
            AnnexIVPack ready for markdown or JSON export.
        """
        now = datetime.now(timezone.utc).isoformat()
        intended_purposes = intended_purposes or [
            "Interactive educational assistance and learning support."
        ]
        exceptions_claimed = exceptions_claimed or []

        models_registered = governance_report.get("models_registered", 0)
        decisions_logged = governance_report.get("decisions_logged", 0)
        disputes_filed = governance_report.get("disputes_filed", 0)
        disputes_pending = governance_report.get("disputes_pending_human_review", 0)
        models_with_bias_analysis = governance_report.get(
            "models_with_bias_analysis", 0
        )
        audit_entries = governance_report.get("audit_trail_entries", len(audit_trail))

        sections = {
            "Provider and system identification": {
                "provider": provider_name,
                "system_name": system_name,
                "geographic_market": geographic_market,
                "generation_date": now,
            },
            "Intended purpose": {
                "description": "; ".join(intended_purposes),
                "domains": "Education / tutoring",
                "user_groups": "Students, educators",
            },
            "Model inventory": {
                "models_registered": models_registered,
                "models_with_bias_analysis": models_with_bias_analysis,
            },
            "Decision logging": {
                "decisions_logged": decisions_logged,
                "disputes_filed": disputes_filed,
                "disputes_pending_human_review": disputes_pending,
            },
            "Audit trail": {
                "total_entries": audit_entries,
                "compliance_events": len(audit_trail),
                "retention_period_days": 180,
            },
            "Transparency and exceptions": {
                "exceptions_claimed": exceptions_claimed,
                "exception_justification": (
                    "None — full transparency for EU deploy."
                    if not exceptions_claimed
                    else "; ".join(exceptions_claimed)
                ),
            },
            "Data quality and governance": {
                "allowlist_controlled": True,
                "model_allowlist_size": len(MODEL_CATALOG),
            },
        }

        regulation_counts: Dict[str, int] = {}
        for event in audit_trail:
            regulation = event.get("regulation")
            if regulation:
                regulation_counts[regulation] = regulation_counts.get(regulation, 0) + 1

        sections["Regulatory activity summary"] = {
            "regulations": list(regulation_counts.keys()),
            "events_by_regulation": regulation_counts,
        }

        return AnnexIVPack(
            generated_at=now,
            sections=sections,
            governance_report=governance_report,
            audit_trail=audit_trail,
        )
