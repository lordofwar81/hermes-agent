"""
Performance Monitoring Module

Continuous monitoring of model performance with drift detection,
alerting, and automated retriggering capabilities.

Features:
- Real-time performance tracking
- Statistical drift detection (PSI, KS test, Chi-squared, ADWIN)
- Configurable alert channels
- Automated retraining triggers
- Performance degradation detection
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class DriftDetector:
    """Statistical drift detection algorithms."""
    
    def __init__(self, method: str = "psi", threshold: float = 0.15):
        self.method = method
        self.threshold = threshold
    
    def detect_drift(self, baseline_data: List[float], current_data: List[float]) -> Dict[str, Any]:
        """Detect if drift has occurred between baseline and current data."""
        if not baseline_data or not current_data:
            return {"has_drift": False, "reason": "insufficient_data"}
        
        baseline_array = np.array(baseline_data)
        current_array = np.array(current_data)
        
        if self.method == "psi":
            return self._psi_drift(baseline_array, current_array)
        elif self.method == "ks_test":
            return self._ks_drift(baseline_array, current_array)
        elif self.method == "chi_squared":
            return self._chi_squared_drift(baseline_array, current_array)
        elif self.method == "adwin":
            return self._adwin_drift(baseline_array, current_array)
        else:
            raise ValueError(f"Unknown drift detection method: {self.method}")
    
    def _psi_drift(self, baseline: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Population Stability Index (PSI) drift detection."""
        # Create bins
        all_values = np.concatenate([baseline, current])
        min_val, max_val = np.min(all_values), np.max(all_values)
        bins = np.linspace(min_val, max_val, 11)  # 10 bins
        
        # Calculate distributions
        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)
        
        # Avoid zero division
        baseline_pct = baseline_hist / len(baseline)
        current_pct = current_hist / len(current)
        
        # Calculate PSI
        psi = np.sum(
            (current_pct - baseline_pct) * np.log(current_pct / (baseline_pct + 1e-10) + 1e-10)
        )
        
        has_drift = psi > self.threshold
        
        return {
            "has_drift": has_drift,
            "psi": float(psi),
            "threshold": self.threshold,
            "reason": f"PSI = {psi:.4f} > {self.threshold:.3f}" if has_drift else f"PSI = {psi:.4f} <= {self.threshold:.3f}",
        }
    
    def _ks_drift(self, baseline: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test drift detection."""
        from scipy import stats
        
        statistic, p_value = stats.ks_2samp(baseline, current)
        has_drift = p_value < self.threshold
        
        return {
            "has_drift": has_drift,
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": self.threshold,
            "reason": f"p-value = {p_value:.4f} < {self.threshold:.3f}" if has_drift else f"p-value = {p_value:.4f} >= {self.threshold:.3f}",
        }
    
    def _chi_squared_drift(self, baseline: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """Chi-squared test drift detection."""
        from scipy import stats
        
        # Create contingency table
        bins = np.linspace(min(baseline.min(), current.min()), 
                          max(baseline.max(), current.max()), 5)
        baseline_hist, _ = np.histogram(baseline, bins=bins)
        current_hist, _ = np.histogram(current, bins=bins)
        
        contingency_table = np.array([baseline_hist, current_hist])
        statistic, p_value, _, _ = stats.chi2_contingency(contingency_table)
        has_drift = p_value < self.threshold
        
        return {
            "has_drift": has_drift,
            "chi2_statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": self.threshold,
            "reason": f"p-value = {p_value:.4f} < {self.threshold:.3f}" if has_drift else f"p-value = {p_value:.4f} >= {self.threshold:.3f}",
        }
    
    def _adwin_drift(self, baseline: np.ndarray, current: np.ndarray) -> Dict[str, Any]:
        """ADaptive WINdowing (ADWIN) drift detection."""
        # Simplified ADWIN implementation
        # In production, use river.drift.ADWIN
        mean_baseline = np.mean(baseline)
        mean_current = np.mean(current)
        
        std_baseline = np.std(baseline)
        std_current = np.std(current)
        
        # Combined standard error
        combined_std = np.sqrt(std_baseline**2/len(baseline) + std_current**2/len(current))
        z_score = abs(mean_current - mean_baseline) / (combined_std + 1e-10)
        
        has_drift = z_score > self.threshold
        
        return {
            "has_drift": has_drift,
            "z_score": float(z_score),
            "baseline_mean": float(mean_baseline),
            "current_mean": float(mean_current),
            "threshold": self.threshold,
            "reason": f"Z-score = {z_score:.4f} > {self.threshold:.3f}" if has_drift else f"Z-score = {z_score:.4f} <= {self.threshold:.3f}",
        }


class PerformanceMonitor:
    """Continuous model performance monitoring system."""
    
    def __init__(self, config, artifacts_dir: str = "./pipeline/artifacts"):
        """
        Args:
            config: MonitoringConfig instance
            artifacts_dir: Directory for storing monitoring artifacts
        """
        self.config = config
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.drift_detector = DriftDetector(
            method=config.drift_detection_method,
            threshold=config.drift_threshold
        )
        
        self._performance_history = []
        self._baseline_metrics = None
        self._setup_alert_channels()
    
    def _setup_alert_channels(self):
        """Setup alert notification channels."""
        self.alert_channels = []
        
        if "log" in self.config.alert_channels:
            self.alert_channels.append("log")
        
        if "webhook" in self.config.alert_channels and self.config.webhook_url:
            self.alert_channels.append("webhook")
        
        if "slack" in self.config.alert_channels and self.config.slack_webhook:
            self.alert_channels.append("slack")
        
        if "email" in self.config.alert_channels and self.config.email_recipients:
            self.alert_channels.append("email")
    
    def setup_model_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring for a deployed model."""
        logger.info("Setting up model monitoring")
        
        try:
            # Initialize baseline metrics
            self._initialize_baseline()
            
            # Setup monitoring schedule
            schedule_info = {
                "check_interval_minutes": self.config.check_interval_minutes,
                "next_check": self._get_next_check_time(),
                "alert_channels": self.alert_channels,
            }
            
            # Save monitoring configuration
            config_path = self.artifacts_dir / "monitoring_config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "config": self.config.to_dict() if hasattr(self.config, "to_dict") else self.config.__dict__,
                    "schedule": schedule_info,
                    "setup_at": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2, default=str)
            
            logger.info("Monitoring setup complete. Next check at %s", schedule_info["next_check"])
            return {
                "success": True,
                "schedule": schedule_info,
                "config_path": str(config_path),
            }
            
        except Exception as e:
            logger.error("Monitoring setup failed: %s", e)
            return {
                "success": False,
                "error": str(e),
            }
    
    def _initialize_baseline(self):
        """Initialize baseline performance metrics."""
        logger.info("Initializing baseline performance metrics")
        
        # In production, this would query the deployed model
        # For simulation, generate realistic baseline metrics
        self._baseline_metrics = {
            "accuracy": 0.85 + np.random.normal(0, 0.02),
            "precision": 0.83 + np.random.normal(0, 0.02),
            "recall": 0.82 + np.random.normal(0, 0.02),
            "f1_score": 0.82 + np.random.normal(0, 0.02),
            "latency_ms": 120 + np.random.normal(0, 15),
            "error_rate": 0.02 + np.random.normal(0, 0.005),
        }
        
        # Ensure values are in valid ranges
        self._baseline_metrics = {
            k: max(0, min(1 if k != "latency_ms" else 1000, v))
            for k, v in self._baseline_metrics.items()
        }
        
        logger.info("Baseline metrics: %s", self._baseline_metrics)
    
    def record_performance(self, metrics: Dict[str, float], metadata: Optional[Dict] = None):
        """Record performance metrics for monitoring."""
        timestamp = datetime.now(timezone.utc)
        record = {
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "metadata": metadata or {},
        }
        
        self._performance_history.append(record)
        
        # Maintain history size
        cutoff_time = timestamp - timedelta(days=self.config.retention_days)
        self._performance_history = [
            r for r in self._performance_history 
            if datetime.fromisoformat(r["timestamp"]) >= cutoff_time
        ]
        
        # Save to disk
        self._save_performance_history()
    
    def check_performance_drift(self) -> Dict[str, Any]:
        """Check for performance drift and degradation."""
        if not self._baseline_metrics or len(self._performance_history) < 10:
            return {"has_drift": False, "reason": "insufficient_data"}
        
        # Get recent performance data
        recent_metrics = [
            r["metrics"] for r in self._performance_history[-10:]
        ]
        
        # Check for degradation
        degradation_results = []
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            if metric_name in self._baseline_metrics:
                baseline_val = self._baseline_metrics[metric_name]
                recent_vals = [m[metric_name] for m in recent_metrics if metric_name in m]
                recent_avg = np.mean(recent_vals) if recent_vals else 0
                
                degradation = baseline_val - recent_avg
                if degradation > self.config.performance_degradation_threshold:
                    degradation_results.append({
                        "metric": metric_name,
                        "baseline": baseline_val,
                        "current": recent_avg,
                        "degradation": degradation,
                        "threshold": self.config.performance_degradation_threshold,
                    })
        
        has_degradation = len(degradation_results) > 0
        
        # Check for drift using all recent metrics
        all_metrics = []
        for metrics in recent_metrics:
            all_metrics.extend(list(metrics.values()))
        
        if all_metrics:
            baseline_vals = list(self._baseline_metrics.values())
            drift_result = self.drift_detector.detect_drift(baseline_vals, all_metrics)
        else:
            drift_result = {"has_drift": False, "reason": "insufficient_data"}
        
        # Combined alert
        has_alert = has_degradation or drift_result["has_drift"]
        
        alert_info = {
            "has_alert": has_alert,
            "performance_degradation": degradation_results,
            "statistical_drift": drift_result,
            "alert_severity": "high" if has_degradation else "medium",
            "recommendation": "retrain_model" if has_alert else "no_action",
        }
        
        # Send alerts if needed
        if has_alert:
            self._send_alert(alert_info)
        
        return alert_info
    
    def _send_alert(self, alert_info: Dict[str, Any]):
        """Send alerts through configured channels."""
        alert_message = self._format_alert_message(alert_info)
        
        for channel in self.alert_channels:
            try:
                if channel == "log":
                    logger.warning("PERFORMANCE ALERT: %s", alert_message)
                elif channel == "webhook":
                    self._send_webhook_alert(alert_message)
                elif channel == "slack":
                    self._send_slack_alert(alert_message)
                elif channel == "email":
                    self._send_email_alert(alert_message)
                    
            except Exception as e:
                logger.error("Failed to send alert via %s: %s", channel, e)
    
    def _format_alert_message(self, alert_info: Dict[str, Any]) -> str:
        """Format alert message for notification."""
        parts = ["Model performance alert"]
        
        if alert_info["performance_degradation"]:
            parts.append("Performance degradation detected:")
            for deg in alert_info["performance_degradation"]:
                parts.append(f"  - {deg['metric']}: {deg['degradation']:.4f} degradation (baseline: {deg['baseline']:.3f})")
        
        if alert_info["statistical_drift"]["has_drift"]:
            parts.append(f"Statistical drift: {alert_info['statistical_drift']['reason']}")
        
        parts.append("Recommendation: " + alert_info["recommendation"])
        parts.append(f"Severity: {alert_info['alert_severity']}")
        
        return "\n".join(parts)
    
    def _send_webhook_alert(self, message: str):
        """Send alert via webhook."""
        import requests
        
        payload = {
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": "warning",
        }
        
        try:
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info("Webhook alert sent successfully")
        except Exception as e:
            logger.error("Webhook alert failed: %s", e)
    
    def _send_slack_alert(self, message: str):
        """Send alert via Slack webhook."""
        import requests
        
        payload = {
            "text": f"⚠️ Performance Alert\n{message}",
            "channel": "#alerts" if self.config.slack_webhook.endswith("/alerts") else "#general",
        }
        
        try:
            response = requests.post(
                self.config.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Slack alert sent successfully")
        except Exception as e:
            logger.error("Slack alert failed: %s", e)
    
    def _send_email_alert(self, message: str):
        """Send alert via email."""
        # This would use an email service like SendGrid, AWS SES, etc.
        # For simulation, just log it
        logger.info("Email alert would be sent to: %s", self.config.email_recipients)
        logger.info("Email content: %s", message)
    
    def _save_performance_history(self):
        """Save performance history to disk."""
        history_path = self.artifacts_dir / "performance_history.json"
        
        # Convert to serializable format
        serializable_history = []
        for record in self._performance_history:
            serializable = record.copy()
            if isinstance(serializable["timestamp"], datetime):
                serializable["timestamp"] = serializable["timestamp"].isoformat()
            serializable_history.append(serializable)
        
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2, default=str)
    
    def _get_next_check_time(self) -> str:
        """Get the next scheduled check time."""
        next_check = datetime.now(timezone.utc) + timedelta(minutes=self.config.check_interval_minutes)
        return next_check.isoformat()
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "is_active": len(self._performance_history) > 0,
            "last_check": self._performance_history[-1]["timestamp"] if self._performance_history else None,
            "total_checks": len(self._performance_history),
            "baseline_metrics": self._baseline_metrics,
            "next_check": self._get_next_check_time(),
            "alert_channels": self.alert_channels,
        }
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get performance trends for the last N days."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        recent_data = [
            r for r in self._performance_history
            if datetime.fromisoformat(r["timestamp"]) >= cutoff_time
        ]
        
        if not recent_data:
            return {"error": "No data available"}
        
        trends = {}
        for metric_name in ["accuracy", "precision", "recall", "f1_score"]:
            values = [r["metrics"].get(metric_name, 0) for r in recent_data if metric_name in r["metrics"]]
            if values:
                trends[metric_name] = {
                    "values": values,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": "improving" if values[-1] > values[0] else "declining",
                }
        
        return {
            "period_days": days,
            "data_points": len(recent_data),
            "trends": trends,
        }