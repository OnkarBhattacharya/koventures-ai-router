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
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
    
    def compute_hash(self) -> str:
        """
        Compute hash for audit log integrity verification.
        
        Ensures audit logs haven't been tampered with.
        """
        data_to_hash = f"{self.id}{self.timestamp}{self.user_id}{self.event_type.value}"
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
    
    def __init__(self, max_logs: int = 100000):
        self.events: List[AuditEvent] = []
        self.max_logs = max_logs
        self.log_file: Optional[str] = None
        self.enable_persistence = bool(os.getenv('AUDIT_LOG_PERSISTENCE', 'false').lower() == 'true')
    
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
        event = AuditEvent(event_type, user_id, resource, action, severity)
        
        if details:
            event.add_details(details)
        
        if ip_address:
            event.ip_address = ip_address
        
        if user_agent:
            event.user_agent = user_agent
        
        event.compute_hash()
        
        self.events.append(event)
        
        # Enforce size limit with rotation
        if len(self.events) > self.max_logs:
            self.events = self.events[-self.max_logs:]
        
        # Optionally persist to file
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
    
    def verify_log_integrity(self, event_id: str, expected_hash: str) -> bool:
        """Verify integrity of a specific log entry"""
        for event in self.events:
            if event.id == event_id:
                return event.hash == expected_hash
        
        return False
    
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
