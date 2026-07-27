"""AI Governance and Regulation Compliance Module

Implements compliance with:
- EU AI Act (when applicable)
- GDPR Article 22 (Automated Decision-Making)
- Responsible AI principles
- Model transparency and explainability
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelRiskLevel(Enum):
    """AI Act classification of model risk"""
    PROHIBITED = "prohibited"
    HIGH = "high_risk"
    LIMITED = "limited_risk"
    MINIMAL = "minimal_risk"


class AIDecisionType(Enum):
    """Types of decisions made by AI systems"""
    AUTOMATED_DECISION = "automated_decision"
    HUMAN_INVOLVED = "human_in_loop"
    HUMAN_OVERSIGHT = "human_oversight"
    FULLY_HUMAN = "fully_human_decision"


class ModelCard:
    """
    Model Card - Documentation of model characteristics, performance, and limitations.
    Based on: https://arxiv.org/abs/1810.03993
    """
    
    def __init__(self, model_name: str, version: str, developer: str):
        self.id = str(uuid.uuid4())
        self.model_name = model_name
        self.version = version
        self.developer = developer
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        
        # Model Details
        self.description = ""
        self.intended_use = []
        self.constraints = []
        self.limitations = []
        
        # Performance Metrics
        self.performance_metrics: Dict[str, float] = {}
        self.performance_disaggregated: Dict[str, Dict[str, float]] = {}  # metrics by demographic group
        
        # Evaluation Data
        self.training_data_description = ""
        self.evaluation_data_description = ""
        
        # Bias and Fairness
        self.bias_analysis: List[Dict] = []
        self.fairness_considerations: List[str] = []
        
        # Transparency and Explainability
        self.explainability_method = ""
        self.transparency_notes = ""
        
        # Environmental Impact
        self.training_compute: Optional[float] = None  # in kWh
        self.inference_compute: Optional[float] = None  # in kWh
        
        # Recommendations
        self.recommendations: List[str] = []
    
    def add_performance_metric(self, metric_name: str, value: float, 
                               demographic_group: str = None) -> None:
        """Record performance metric"""
        if demographic_group:
            if demographic_group not in self.performance_disaggregated:
                self.performance_disaggregated[demographic_group] = {}
            self.performance_disaggregated[demographic_group][metric_name] = value
        else:
            self.performance_metrics[metric_name] = value
        
        self.last_updated = datetime.now(timezone.utc)
    
    def add_bias_analysis(self, bias_type: str, description: str, 
                          mitigation: str = None) -> None:
        """Record bias analysis"""
        analysis = {
            "type": bias_type,
            "description": description,
            "mitigation": mitigation,
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }
        self.bias_analysis.append(analysis)
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict:
        """Serialize model card"""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "version": self.version,
            "developer": self.developer,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "description": self.description,
            "intended_use": self.intended_use,
            "constraints": self.constraints,
            "limitations": self.limitations,
            "performance_metrics": self.performance_metrics,
            "performance_disaggregated": self.performance_disaggregated,
            "bias_analysis": self.bias_analysis,
            "fairness_considerations": self.fairness_considerations,
            "explainability_method": self.explainability_method,
            "environmental_impact": {
                "training_compute_kwh": self.training_compute,
                "inference_compute_kwh": self.inference_compute
            }
        }


class ExplainabilityProvider:
    """Provides model explainability and interpretability"""
    
    def __init__(self):
        self.explanations: List[Dict] = []
    
    def add_explanation(self, prediction_id: str, explanation_type: str,
                       explanation_data: Dict, confidence: float = None,
                       reasoning_trace: List[str] = None) -> str:
        """
        Record an explanation for a model prediction.
        
        explanation_type: 'feature_importance', 'lime', 'shap', 'attention', 'rule_based'
        """
        explanation_id = str(uuid.uuid4())
        explanation_record = {
            "id": explanation_id,
            "prediction_id": prediction_id,
            "type": explanation_type,
            "data": explanation_data,
            "confidence": confidence,
            "reasoning_trace": reasoning_trace or [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.explanations.append(explanation_record)
        
        return explanation_id
    
    def get_explanation(self, explanation_id: str) -> Optional[Dict]:
        """Retrieve an explanation"""
        for exp in self.explanations:
            if exp["id"] == explanation_id:
                return exp
        return None


class AIDecisionLog:
    """Log AI decision-making for transparency and accountability"""
    
    def __init__(self, db=None):
        self.db = db
        # Refactored to support a repository pattern for persistence
        self.decisions: List[Dict] = [] 
        self.disputes: List[Dict] = []  
        self.storage_enabled = os.getenv("PERSISTENCE_ENABLED", "false").lower() == "true"
    
    def log_decision(self, user_id: str, decision_type: AIDecisionType,
                     input_data: Dict, output: Any, model_version: str,
                     explanation_id: str = None, human_review_required: bool = False) -> str:
        """
        Log an AI decision.
        
        GDPR Article 22: Users have the right not to be subject to automated decisions.
        This logging ensures we can provide the right to explanation and human review.
        """
        decision_id = str(uuid.uuid4())
        
        decision_record = {
            "id": decision_id,
            "user_id": user_id,
            "decision_type": decision_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "explanation_id": explanation_id,
            "human_review_required": human_review_required,
            "human_review_completed": False,
            "output": output,
            # Don't store raw input data to minimize personal data retention
            "input_summary": self._create_input_summary(input_data)
        }
        
        self.decisions.append(decision_record)
        
        if self.storage_enabled:
            if self.db:
                try:
                    self.db.collection("ai_decisions").add(decision_record)
                except Exception as e:
                    logger.error(f"Failed to persist AI decision: {e}")

        if human_review_required:
            logger.info(f"Decision marked for human review: {decision_id}")
        
        return decision_id
    
    def log_dispute(self, decision_id: str, reason: str, user_comment: str = None) -> str:
        """
        Log a user dispute of an AI decision.
        
        Supports GDPR Article 22(3) - user's right to explanation and human review.
        """
        dispute_id = str(uuid.uuid4())
        
        dispute_record = {
            "id": dispute_id,
            "decision_id": decision_id,
            "reason": reason,
            "user_comment": user_comment,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "human_review_status": "pending"
        }
        
        self.disputes.append(dispute_record)
        
        # Mark decision for review
        for decision in self.decisions:
            if decision["id"] == decision_id:
                decision["human_review_required"] = True
                logger.info(f"Dispute logged for decision {decision_id}: {reason}")
        
        return dispute_id
    
    def complete_human_review(self, decision_id: str, review_notes: str, 
                             overridden: bool = False) -> bool:
        """Mark human review as completed"""
        for decision in self.decisions:
            if decision["id"] == decision_id:
                decision["human_review_completed"] = True
                decision["human_review_notes"] = review_notes
                decision["human_override"] = overridden
                decision["review_completed_at"] = datetime.now(timezone.utc).isoformat()
                
                logger.info(f"Human review completed for decision {decision_id}")
                return True
        
        return False
    
    @staticmethod
    def _create_input_summary(input_data: Dict) -> Dict:
        """Create a summary of input data without storing sensitive details"""
        return {
            "keys": list(input_data.keys()),
            "data_types": {k: type(v).__name__ for k, v in input_data.items()},
            "summary": "Input data summary for compliance tracking"
        }


class AIGovernanceManager:
    """Central management for AI governance and compliance"""
    
    def __init__(self, db=None):
        self.db = db
        self.model_cards: Dict[str, ModelCard] = {} 
        self.decision_log = AIDecisionLog(db=db)
        self.explainability_provider = ExplainabilityProvider()
        self.audit_trail: List[Dict] = [] # Suggest moving to a dedicated database collection
        self.model_monitoring: Dict[str, Dict] = {}
    
    def register_model(self, model_name: str, version: str, developer: str) -> ModelCard:
        """Register a model with documentation"""
        model_id = f"{model_name}_{version}"
        
        model_card = ModelCard(model_name, version, developer)
        self.model_cards[model_id] = model_card
        
        self._audit("model_registered", model_id, {"model_name": model_name})
        logger.info(f"Model registered: {model_name} v{version}")
        
        return model_card
    
    def get_model_card(self, model_id: str) -> Optional[ModelCard]:
        """Retrieve model card"""
        return self.model_cards.get(model_id)
    
    def log_ai_decision(self, user_id: str, decision_type: AIDecisionType,
                        input_data: Dict, output: Any, model_id: str,
                        explanation_id: str = None) -> str:
        """Log an AI decision with explanation support"""
        decision_id = self.decision_log.log_decision(
            user_id, decision_type, input_data, output, model_id, explanation_id
        )
        
        self._audit("decision_logged", decision_id, {"user_id": user_id, "model_id": model_id})
        return decision_id
    
    def add_explanation(self, prediction_id: str, explanation_type: str,
                       explanation_data: Dict, confidence: float = None) -> str:
        """Add explanation for a prediction"""
        explanation_id = self.explainability_provider.add_explanation(
            prediction_id, explanation_type, explanation_data, confidence
        )
        
        self._audit("explanation_added", explanation_id, 
                   {"explanation_type": explanation_type})
        return explanation_id
    
    def record_model_performance(self, model_id: str, metrics: Dict,
                                disaggregated_metrics: Dict = None) -> bool:
        """
        Record model performance metrics including disaggregated performance.
        
        Important for fairness: disaggregated metrics reveal potential biases.
        """
        model_card = self.get_model_card(model_id)
        if not model_card:
            logger.error(f"Model not found: {model_id}")
            return False
        
        for metric_name, value in metrics.items():
            model_card.add_performance_metric(metric_name, value)
        
        if disaggregated_metrics:
            for group, metrics_dict in disaggregated_metrics.items():
                for metric_name, value in metrics_dict.items():
                    model_card.add_performance_metric(metric_name, value, group)
        
        self._audit("performance_recorded", model_id, {"metrics_count": len(metrics)})
        logger.info(f"Performance metrics recorded for {model_id}")
        
        return True
    
    def record_bias_analysis(self, model_id: str, bias_type: str,
                            description: str, mitigation: str = None) -> bool:
        """Record bias analysis for a model"""
        model_card = self.get_model_card(model_id)
        if not model_card:
            logger.error(f"Model not found: {model_id}")
            return False
        
        model_card.add_bias_analysis(bias_type, description, mitigation)
        
        self._audit("bias_analysis", model_id, {"bias_type": bias_type})
        logger.info(f"Bias analysis recorded for {model_id}: {bias_type}")
        
        return True
    
    def log_decision_dispute(self, decision_id: str, reason: str,
                            user_comment: str = None) -> str:
        """Handle user dispute of AI decision"""
        dispute_id = self.decision_log.log_dispute(decision_id, reason, user_comment)
        
        self._audit("dispute_logged", dispute_id, {"decision_id": decision_id})
        logger.info(f"Decision dispute logged: {dispute_id}")
        
        return dispute_id
    
    def get_audit_trail(self, days: int = 90) -> List[Dict]:
        """Retrieve audit trail for compliance verification"""
        from datetime import timedelta
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        filtered_trail = [
            entry for entry in self.audit_trail
            if datetime.fromisoformat(entry["timestamp"]) > start_date
        ]
        
        return sorted(filtered_trail, key=lambda x: x["timestamp"], reverse=True)
    
    def _audit(self, action: str, subject: str, details: Dict = None) -> None:
        """Internal audit logging"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "subject": subject,
            "details": details or {}
        }
        self.audit_trail.append(audit_entry)
    
    def generate_governance_report(self) -> Dict:
        """Generate AI governance compliance report"""
        return {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "models_registered": len(self.model_cards),
            "decisions_logged": len(self.decision_log.decisions),
            "disputes_filed": len(self.decision_log.disputes),
            "disputes_pending_human_review": sum(
                1 for d in self.decision_log.disputes if d["status"] == "pending"
            ),
            "models_with_bias_analysis": sum(
                1 for mc in self.model_cards.values() if mc.bias_analysis
            ),
            "audit_trail_entries": len(self.audit_trail)
        }
