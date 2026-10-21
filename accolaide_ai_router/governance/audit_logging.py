"""Comprehensive Audit Logging System for Compliance

Logs:
- Data access events
- Model decisions and predictions
- User data operations (CRUD)
- Consent management
- System security events
- Regulatory compliance actions
"""

import hashlib
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)

# Cross-process/cross-module lock: hash + persist event + persist tip must be
# atomic. Module-level threading.Lock alone is insufficient when the package is
# imported twice (repo root vs chat-app/ duplicate) or across processes.
_CHAIN_LOCK = threading.Lock()

# Shared lock path is derived from the DB path so every module copy
# (repo root vs chat-app/ duplicate) targets the same lock file.

def _chain_lock_path(db=None) -> str:
    """Resolve a lock file shared by every module copy writing to this DB."""
    db_path = None
    if db is not None:
        db_path = getattr(db, "_db_path", None) or getattr(db, "db_path", None)
    if db_path:
        return f"{db_path}.audit_chain.lock"
    # Fallback: one process-wide lock when no DB path is available.
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".audit_chain.lock",
    )


def _acquire_chain_lock(db=None):
    """Best-effort file lock so duplicate module copies cannot branch the chain."""
    lock_path = _chain_lock_path(db)
    try:
        import fcntl

        fh = open(lock_path, "a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        return fh
    except Exception:
        return None


def _release_chain_lock(fh) -> None:
    if fh is None:
        return
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


class AuditEventType(Enum):
    """Types of auditable events"""
    # Data Access
    DATA_ACCESS = "data_access"
    DATA_CREATED = "data_created"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    
    # User Operations
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_REGISTERED = "user_registered"
    USER_DELETED = "user_deleted"
    
    # AI/Model Operations
    MODEL_PREDICTION = "model_prediction"
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYED = "model_deployed"
    
    # Consent and Privacy
    CONSENT_GRANTED = "consent_granted"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    DATA_EXPORT = "data_export"
    DATA_ERASURE = "data_erasure"
    
    # Security
    SECURITY_ALERT = "security_alert"
    ACCESS_DENIED = "access_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # Compliance
    AUDIT_LOG_ACCESSED = "audit_log_accessed"
    COMPLIANCE_CHECK = "compliance_check"
    REPORT_GENERATED = "report_generated"


class SeverityLevel(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AuditEvent:
    """Represents a single audit event"""
    
    def __init__(self, event_type: AuditEventType, user_id: str, 
                 resource: str, action: str, severity: SeverityLevel = SeverityLevel.INFO):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc)
        self.event_type = event_type
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.severity = severity
        self.details: Dict = {}
        self.result: str = "success"
        self.ip_address: Optional[str] = None
        self.user_agent: Optional[str] = None
        self.hash: Optional[str] = None
    
    def add_details(self, details: Dict) -> None:
        """Add additional event details"""
        self.details.update(details)
    
    def set_result(self, result: str) -> None:
        """Set operation result"""
        self.result = result
    
    def compute_hash(self, prev_hash: str = "") -> str:
        """
        Compute hash for audit log integrity verification.

        Includes the previous event's hash so the log forms a tamper-evident
        chain: altering any entry invalidates every subsequent hash.

        The timestamp is serialized with ``isoformat()`` so the hash computed
        in-memory matches the hash recomputed from the persisted record (which
        stores ``timestamp`` as ``isoformat()``). This keeps
        ``verify_chain_from_db`` consistent with the live chain.
        """
        data_to_hash = (
            f"{prev_hash}{self.id}{self.timestamp.isoformat()}{self.user_id}"
            f"{self.event_type.value}{self.action}"
        )
        self.hash = hashlib.sha256(data_to_hash.encode()).hexdigest()
        return self.hash
    
    def to_dict(self) -> Dict:
        """Serialize audit event"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "hash": self.hash
        }


class AuditLogger:
    """Central audit logging system"""
    
    def __init__(self, max_logs: int = 100000, db=None):
        self.events: List[AuditEvent] = []
        self.max_logs = max_logs
        self.log_file: Optional[str] = None
        self.db = db  # Firestore client or LocalStorage instance
        self.enable_persistence = bool(os.getenv('AUDIT_LOG_PERSISTENCE', 'false').lower() == 'true')
        self._last_hash = ""
        # Continue the hash chain across restarts so tamper-evidence survives
        # process restarts (the chain tip is persisted on every log_event).
        if self.db:
            try:
                snap = self.db.collection("audit_log_meta").document("chain").get()
                if snap and getattr(snap, "exists", False):
                    self._last_hash = (snap.to_dict() or {}).get("last_hash", "") or ""
            except Exception as exc:
                logger.warning("Could not load audit chain tip: %s", exc)
    
    def _persist_to_db(self, event: AuditEvent) -> None:
        """Persist audit event to the database if available."""
        if not self.db:
            return
        try:
            self.db.collection("audit_log").add(event.to_dict())
        except Exception as exc:
            logger.error("Failed to persist audit event to database: %s", exc)

    def _persist_chain_tip(self) -> None:
        """Persist the running hash-chain tip so it survives restarts."""
        if not self.db:
            return
        try:
            self.db.collection("audit_log_meta").document("chain").set({"last_hash": self._last_hash})
        except Exception as exc:
            logger.error("Failed to persist audit chain tip: %s", exc)
    
    def _load_chain_tip(self) -> str:
        """Read the persisted chain tip (empty string if none/unavailable)."""
        if not self.db:
            return ""
        try:
            snap = self.db.collection("audit_log_meta").document("chain").get()
            if snap and getattr(snap, "exists", False):
                return (snap.to_dict() or {}).get("last_hash", "") or ""
        except Exception as exc:
            logger.warning("Could not load audit chain tip: %s", exc)
        return ""

    def log_event(self, event_type: AuditEventType, user_id: str,
                  resource: str, action: str,
                  severity: SeverityLevel = SeverityLevel.INFO,
                  details: Dict = None,
                  ip_address: str = None,
                  user_agent: str = None) -> str:
        """
        Log an audit event.
        
        Returns:
            Event ID
        """
        with _CHAIN_LOCK:
            file_fh = _acquire_chain_lock(self.db)
            try:
                # Always continue from the persisted tip: other AuditLogger
                # instances (app reloads, tests, multi-worker) may have appended
                # since this object last wrote. An empty tip after a chain reset
                # must clear the in-memory hash so the next event starts fresh.
                if self.db:
                    self._last_hash = self._load_chain_tip() or ""

                event = AuditEvent(event_type, user_id, resource, action, severity)

                if details:
                    event.add_details(details)

                if ip_address:
                    event.ip_address = ip_address

                if user_agent:
                    event.user_agent = user_agent

                event.compute_hash(self._last_hash)

                self.events.append(event)
                self._last_hash = event.hash

                # Enforce size limit with rotation
                if len(self.events) > self.max_logs:
                    self.events = self.events[-self.max_logs:]

                # Persist to database (survives restarts)
                self._persist_to_db(event)

                # Persist the chain tip so the tamper-evident chain continues after a restart
                self._persist_chain_tip()
            finally:
                _release_chain_lock(file_fh)

        # Emit Prometheus metrics for this event

        # Optionally persist to file as well
        if self.enable_persistence:
            self._persist_event(event)

        logger.debug(f"Audit event logged: {event_type.value} - {resource}")

        return event.id
    
    def log_data_access(self, user_id: str, data_type: str, 
                       record_id: str, ip_address: str = None) -> str:
        """Log data access event"""
        return self.log_event(
            AuditEventType.DATA_ACCESS,
            user_id,
            resource=data_type,
            action=f"accessed_{record_id}",
            details={"record_id": record_id},
            ip_address=ip_address
        )
    
    def log_model_prediction(self, user_id: str, model_id: str,
                            prediction_id: str, input_hash: str = None,
                            confidence: float = None) -> str:
        """Log model prediction event"""
        details = {"prediction_id": prediction_id, "model_id": model_id}
        if input_hash:
            details["input_hash"] = input_hash
        if confidence is not None:
            details["confidence"] = confidence
        
        return self.log_event(
            AuditEventType.MODEL_PREDICTION,
            user_id,
            resource=f"model_{model_id}",
            action="generated_prediction",
            details=details
        )
    
    def log_consent_change(self, user_id: str, consent_type: str,
                          granted: bool, details: Dict = None) -> str:
        """Log consent granted or withdrawn"""
        event_type = AuditEventType.CONSENT_GRANTED if granted else AuditEventType.CONSENT_WITHDRAWN
        
        return self.log_event(
            event_type,
            user_id,
            resource="consent",
            action=f"{consent_type}_{'granted' if granted else 'withdrawn'}",
            details=details or {}
        )
    
    def log_data_export(self, user_id: str, data_type: str,
                       export_format: str, record_count: int) -> str:
        """Log data export event (GDPR Article 20)"""
        return self.log_event(
            AuditEventType.DATA_EXPORT,
            user_id,
            resource=data_type,
            action="exported",
            details={
                "export_format": export_format,
                "record_count": record_count
            }
        )
    
    def log_data_erasure(self, user_id: str, data_type: str,
                        record_count: int, reason: str = None) -> str:
        """Log data erasure event (GDPR Article 17)"""
        return self.log_event(
            AuditEventType.DATA_ERASURE,
            user_id,
            resource=data_type,
            action="erased",
            severity=SeverityLevel.WARNING,
            details={
                "record_count": record_count,
                "reason": reason
            }
        )
    
    def log_security_alert(self, alert_type: str, description: str,
                          user_id: str = "system", affected_resources: List[str] = None) -> str:
        """Log security alert"""
        return self.log_event(
            AuditEventType.SECURITY_ALERT,
            user_id,
            resource="security",
            action=alert_type,
            severity=SeverityLevel.CRITICAL,
            details={
                "description": description,
                "affected_resources": affected_resources or []
            }
        )
    
    def get_events(self, user_id: str = None, event_type: AuditEventType = None,
                   days: int = 90, limit: int = 1000) -> List[Dict]:
        """
        Retrieve audit events with filtering.
        
        Args:
            user_id: Filter by user
            event_type: Filter by event type
            days: Only include events from last N days
            limit: Maximum events to return
        
        Returns:
            List of audit events
        """
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        filtered_events = [
            event for event in self.events
            if event.timestamp > start_date
        ]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        # Sort by timestamp descending
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [e.to_dict() for e in filtered_events[:limit]]
    
    def verify_log_integrity(self, event_id: str = None, expected_hash: str = None) -> bool:
        """Verify the in-memory hash chain and optionally a specific entry."""
        prev = ""
        for event in self.events:
            if event.compute_hash(prev) != event.hash:
                return False
            prev = event.hash
        if event_id and expected_hash:
            for event in self.events:
                if event.id == event_id:
                    return event.hash == expected_hash
        return True

    def verify_chain_from_db(self) -> Dict:
        """Verify the full tamper-evident chain across all persisted events."""
        if not self.db:
            return {"ok": False, "reason": "no database configured"}
        try:
            events = []
            for doc in self.db.collection("audit_log").stream():
                data = doc.to_dict()
                if not data:
                    continue
                # Some backends (e.g. the local SQLite store) persist the event
                # ``id`` as the document id rather than inside the payload. The
                # id is part of the hash pre-image, so restore it from the doc
                # reference when it is missing from the payload.
                if not data.get("id"):
                    data = {**data, "id": getattr(doc, "id", "")}
                events.append(data)
        except Exception as exc:
            return {"ok": False, "reason": str(exc)}
        events.sort(key=lambda e: e.get("timestamp", ""))
        prev = ""
        broken_at = None
        chain_ok = True
        for e in events:
            seed = (
                f"{prev}{e.get('id', '')}{e.get('timestamp', '')}{e.get('user_id', '')}"
                f"{e.get('event_type', '')}{e.get('action', '')}"
            )
            computed = hashlib.sha256(seed.encode()).hexdigest()
            if computed != e.get("hash"):
                broken_at = e.get("id")
                chain_ok = False
                break
            prev = e.get("hash", "")
        if not chain_ok:
            logger.error("Audit hash chain broken at event %s", broken_at)
        return {"ok": chain_ok, "event_count": len(events), "broken_at": broken_at}
    
    def generate_audit_report(self, user_id: str = None, days: int = 30) -> Dict:
        """Generate comprehensive audit report"""
        events = self.get_events(user_id=user_id, days=days)
        
        event_type_counts = {}
        severity_counts = {}
        
        for event in events:
            event_type = event["event_type"]
            severity = event["severity"]
            
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "report_period": f"Last {days} days",
            "user_id": user_id or "all_users",
            "total_events": len(events),
            "event_type_distribution": event_type_counts,
            "severity_distribution": severity_counts,
            "critical_events": sum(1 for e in events if e["severity"] == "critical"),
            "warning_events": sum(1 for e in events if e["severity"] == "warning"),
            "recent_events": events[:20]
        }
    
    def _persist_event(self, event: AuditEvent) -> None:
        """Persist event to file for long-term storage"""
        try:
            log_dir = os.getenv('AUDIT_LOG_DIR', '/tmp/audit_logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl")
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        
        except Exception as e:
            logger.error(f"Failed to persist audit event: {str(e)}")


def assert_audit_sink_db_backed(logger: "AuditLogger") -> None:
    """Fail closed when the audit sink is not DB-backed (plan task 1.2).

    Router AI-Act events are only durable when they flow into the DB-backed
    hash chain; an in-memory logger means "logged" events evaporate on
    restart. Production must refuse to boot without a DB-backed sink.
    """
    if logger is None or getattr(logger, "db", None) is None:
        raise RuntimeError(
            "AUDIT sink is not DB-backed: production requires the hash-chained "
            "audit trail (AuditLogger(db=...)); refusing to boot."
        )


class ComplianceAuditTrail:
    """Specific audit trail for regulatory compliance events"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.compliance_events: List[Dict] = []
    
    def log_gdpr_compliance_action(self, action: str, user_id: str,
                                  details: Dict = None) -> str:
        """Log GDPR-specific compliance action"""
        event_id = self.audit_logger.log_event(
            AuditEventType.COMPLIANCE_CHECK,
            user_id,
            resource="gdpr",
            action=action,
            details=details or {}
        )
        
        self.compliance_events.append({
            "event_id": event_id,
            "action": action,
            "regulation": "GDPR",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return event_id
    
    def log_ai_act_compliance_action(self, action: str, model_id: str,
                                    details: Dict = None) -> str:
        """Log EU AI Act compliance action"""
        event_id = self.audit_logger.log_event(
            AuditEventType.COMPLIANCE_CHECK,
            "system",
            resource=f"model_{model_id}",
            action=action,
            details=details or {}
        )
        
        self.compliance_events.append({
            "event_id": event_id,
            "action": action,
            "regulation": "EU_AI_ACT",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return event_id
    
    def get_compliance_audit_trail(self, regulation: str = None,
                                  days: int = 180) -> List[Dict]:
        """Retrieve compliance-specific audit trail"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        trail = [
            event for event in self.compliance_events
            if datetime.fromisoformat(event["timestamp"]) > start_date
        ]
        
        if regulation:
            trail = [e for e in trail if e["regulation"] == regulation]
        
        return sorted(trail, key=lambda x: x["timestamp"], reverse=True)
