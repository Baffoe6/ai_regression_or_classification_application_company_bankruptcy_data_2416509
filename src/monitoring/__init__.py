"""
Monitoring and alerting configurations for enterprise deployment.
"""

import os
import yaml
from typing import Dict, Any, List
import json


class MonitoringConfig:
    """Monitoring configuration class."""

    def __init__(self):
        self.prometheus_rules = self._get_prometheus_rules()
        self.grafana_dashboards = self._get_grafana_dashboards()
        self.alert_rules = self._get_alert_rules()
        self.sla_metrics = self._get_sla_metrics()

    def _get_prometheus_rules(self) -> Dict[str, Any]:
        """Get Prometheus recording and alerting rules."""
        return {
            "groups": [
                {
                    "name": "bankruptcy_prediction_api",
                    "interval": "30s",
                    "rules": [
                        {
                            "record": "api:request_duration_seconds:mean5m",
                            "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
                        },
                        {
                            "record": "api:request_rate:5m",
                            "expr": "rate(http_requests_total[5m])",
                        },
                        {
                            "record": "api:error_rate:5m",
                            "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])',
                        },
                        {
                            "alert": "HighErrorRate",
                            "expr": "api:error_rate:5m > 0.05",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "bankruptcy-prediction-api",
                            },
                            "annotations": {
                                "summary": "High error rate detected",
                                "description": "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes",
                            },
                        },
                        {
                            "alert": "HighLatency",
                            "expr": "api:request_duration_seconds:mean5m > 2.0",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "bankruptcy-prediction-api",
                            },
                            "annotations": {
                                "summary": "High latency detected",
                                "description": "Average response time is {{ $value }}s for the last 5 minutes",
                            },
                        },
                        {
                            "alert": "APIDown",
                            "expr": 'up{job="bankruptcy-prediction-api"} == 0',
                            "for": "1m",
                            "labels": {
                                "severity": "critical",
                                "service": "bankruptcy-prediction-api",
                            },
                            "annotations": {
                                "summary": "API is down",
                                "description": "Bankruptcy prediction API has been down for more than 1 minute",
                            },
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "process_resident_memory_bytes / 1024 / 1024 > 1000",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "bankruptcy-prediction-api",
                            },
                            "annotations": {
                                "summary": "High memory usage",
                                "description": "Memory usage is {{ $value }}MB",
                            },
                        },
                        {
                            "alert": "ModelLoadFailure",
                            "expr": "models_loaded_total == 0",
                            "for": "1m",
                            "labels": {
                                "severity": "critical",
                                "service": "bankruptcy-prediction-api",
                            },
                            "annotations": {
                                "summary": "No models loaded",
                                "description": "No ML models are currently loaded in the API",
                            },
                        },
                    ],
                },
                {
                    "name": "kubernetes_resources",
                    "interval": "30s",
                    "rules": [
                        {
                            "alert": "PodCrashLooping",
                            "expr": "rate(kube_pod_container_status_restarts_total[15m]) > 0",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Pod is crash looping",
                                "description": "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is restarting frequently",
                            },
                        },
                        {
                            "alert": "PodNotReady",
                            "expr": 'kube_pod_status_ready{condition="false"} == 1',
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Pod not ready",
                                "description": "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} has been not ready for more than 5 minutes",
                            },
                        },
                    ],
                },
            ]
        }

    def _get_grafana_dashboards(self) -> Dict[str, Any]:
        """Get Grafana dashboard configurations."""
        return {
            "bankruptcy_prediction_api_dashboard": {
                "dashboard": {
                    "id": None,
                    "title": "Bankruptcy Prediction API",
                    "tags": ["api", "ml", "bankruptcy"],
                    "timezone": "browser",
                    "panels": [
                        {
                            "id": 1,
                            "title": "Request Rate",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "rate(http_requests_total[5m])",
                                    "legendFormat": "{{method}} {{status}}",
                                }
                            ],
                            "yAxes": [{"label": "Requests/sec", "min": 0}],
                        },
                        {
                            "id": 2,
                            "title": "Response Time",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                                    "legendFormat": "50th percentile",
                                },
                                {
                                    "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                    "legendFormat": "95th percentile",
                                },
                                {
                                    "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
                                    "legendFormat": "99th percentile",
                                },
                            ],
                            "yAxes": [{"label": "Seconds", "min": 0}],
                        },
                        {
                            "id": 3,
                            "title": "Error Rate",
                            "type": "singlestat",
                            "targets": [
                                {
                                    "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])',
                                    "legendFormat": "Error Rate",
                                }
                            ],
                            "valueMaps": [{"value": "null", "text": "0%"}],
                            "colorBackground": True,
                            "thresholds": "0.01,0.05",
                        },
                        {
                            "id": 4,
                            "title": "Memory Usage",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "process_resident_memory_bytes / 1024 / 1024",
                                    "legendFormat": "Memory Usage (MB)",
                                }
                            ],
                            "yAxes": [{"label": "MB", "min": 0}],
                        },
                        {
                            "id": 5,
                            "title": "CPU Usage",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "rate(process_cpu_seconds_total[5m]) * 100",
                                    "legendFormat": "CPU Usage (%)",
                                }
                            ],
                            "yAxes": [{"label": "Percent", "min": 0, "max": 100}],
                        },
                        {
                            "id": 6,
                            "title": "Models Loaded",
                            "type": "singlestat",
                            "targets": [
                                {
                                    "expr": "models_loaded_total",
                                    "legendFormat": "Models",
                                }
                            ],
                        },
                        {
                            "id": 7,
                            "title": "Prediction Accuracy",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "prediction_accuracy",
                                    "legendFormat": "{{model_name}}",
                                }
                            ],
                            "yAxes": [{"label": "Accuracy", "min": 0, "max": 1}],
                        },
                        {
                            "id": 8,
                            "title": "Active Connections",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "nginx_connections_active",
                                    "legendFormat": "Active Connections",
                                }
                            ],
                        },
                    ],
                    "time": {"from": "now-1h", "to": "now"},
                    "refresh": "30s",
                }
            }
        }

    def _get_alert_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get alerting rules configuration."""
        return {
            "critical_alerts": [
                {
                    "name": "API_DOWN",
                    "condition": 'up{job="bankruptcy-prediction-api"} == 0',
                    "duration": "1m",
                    "message": "Bankruptcy Prediction API is down",
                    "channels": ["slack", "email", "pagerduty"],
                },
                {
                    "name": "HIGH_ERROR_RATE",
                    "condition": "api:error_rate:5m > 0.1",
                    "duration": "5m",
                    "message": "High error rate detected: {{ $value | humanizePercentage }}",
                    "channels": ["slack", "email"],
                },
                {
                    "name": "NO_MODELS_LOADED",
                    "condition": "models_loaded_total == 0",
                    "duration": "1m",
                    "message": "No ML models are loaded",
                    "channels": ["slack", "email", "pagerduty"],
                },
            ],
            "warning_alerts": [
                {
                    "name": "HIGH_LATENCY",
                    "condition": "api:request_duration_seconds:mean5m > 2.0",
                    "duration": "5m",
                    "message": "High latency detected: {{ $value }}s",
                    "channels": ["slack"],
                },
                {
                    "name": "HIGH_MEMORY_USAGE",
                    "condition": "process_resident_memory_bytes / 1024 / 1024 > 1000",
                    "duration": "5m",
                    "message": "High memory usage: {{ $value }}MB",
                    "channels": ["slack"],
                },
                {
                    "name": "LOW_PREDICTION_ACCURACY",
                    "condition": "prediction_accuracy < 0.85",
                    "duration": "10m",
                    "message": "Model accuracy below threshold: {{ $value }}",
                    "channels": ["slack", "email"],
                },
            ],
        }

    def _get_sla_metrics(self) -> Dict[str, Any]:
        """Get SLA metrics and targets."""
        return {
            "availability": {
                "target": 99.9,  # 99.9% uptime
                "measurement": 'up{job="bankruptcy-prediction-api"}',
            },
            "latency": {
                "target": 1.0,  # 1 second 95th percentile
                "measurement": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            },
            "error_rate": {
                "target": 0.01,  # 1% error rate
                "measurement": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])',
            },
            "throughput": {
                "target": 100,  # 100 requests per second
                "measurement": "rate(http_requests_total[5m])",
            },
        }

    def export_prometheus_rules(self, output_dir: str) -> str:
        """Export Prometheus rules to YAML file."""
        rules_file = os.path.join(output_dir, "prometheus-rules.yml")
        with open(rules_file, "w") as f:
            yaml.dump(self.prometheus_rules, f, default_flow_style=False)
        return rules_file

    def export_grafana_dashboards(self, output_dir: str) -> List[str]:
        """Export Grafana dashboards to JSON files."""
        dashboard_files = []
        for name, dashboard in self.grafana_dashboards.items():
            dashboard_file = os.path.join(output_dir, f"{name}.json")
            with open(dashboard_file, "w") as f:
                json.dump(dashboard, f, indent=2)
            dashboard_files.append(dashboard_file)
        return dashboard_files

    def export_alert_rules(self, output_dir: str) -> str:
        """Export alert rules to YAML file."""
        alerts_file = os.path.join(output_dir, "alert-rules.yml")
        with open(alerts_file, "w") as f:
            yaml.dump(self.alert_rules, f, default_flow_style=False)
        return alerts_file


class LoggingConfig:
    """Logging configuration for enterprise deployment."""

    def __init__(self):
        self.fluentd_config = self._get_fluentd_config()
        self.elasticsearch_config = self._get_elasticsearch_config()
        self.log_retention_policy = self._get_log_retention_policy()

    def _get_fluentd_config(self) -> Dict[str, Any]:
        """Get Fluentd configuration for log aggregation."""
        return {
            "sources": [
                {
                    "type": "tail",
                    "path": "/var/log/containers/*bankruptcy-prediction*.log",
                    "pos_file": "/var/log/fluentd-containers.log.pos",
                    "tag": "kubernetes.*",
                    "format": "json",
                    "time_key": "time",
                    "time_format": "%Y-%m-%dT%H:%M:%S.%NZ",
                },
                {"type": "forward", "port": 24224, "bind": "0.0.0.0"},
            ],
            "filters": [
                {
                    "type": "kubernetes_metadata",
                    "match": "kubernetes.**",
                    "kubernetes_url": "https://kubernetes.default.svc:443",
                    "verify_ssl": True,
                    "ca_file": "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
                },
                {
                    "type": "parser",
                    "match": "kubernetes.**",
                    "key_name": "log",
                    "parser": {
                        "type": "json",
                        "time_key": "timestamp",
                        "time_format": "%Y-%m-%dT%H:%M:%S.%f",
                    },
                },
            ],
            "matches": [
                {
                    "type": "elasticsearch",
                    "match": "kubernetes.**",
                    "host": "elasticsearch.logging.svc.cluster.local",
                    "port": 9200,
                    "index_name": "bankruptcy-prediction-logs",
                    "type_name": "_doc",
                    "include_tag_key": True,
                    "tag_key": "@log_name",
                    "flush_interval": "1s",
                    "buffer": {
                        "type": "file",
                        "path": "/var/log/fluentd-buffers/kubernetes.system.buffer",
                        "flush_mode": "interval",
                        "retry_type": "exponential_backoff",
                        "flush_thread_count": 2,
                        "flush_interval": "5s",
                        "retry_forever": True,
                        "retry_max_interval": "30s",
                        "chunk_limit_size": "2M",
                        "queue_limit_length": "8",
                        "overflow_action": "block",
                    },
                }
            ],
        }

    def _get_elasticsearch_config(self) -> Dict[str, Any]:
        """Get Elasticsearch configuration for log storage."""
        return {
            "cluster": {
                "name": "bankruptcy-prediction-logs",
                "initial_master_nodes": ["es-master-0", "es-master-1", "es-master-2"],
            },
            "network": {"host": "0.0.0.0"},
            "discovery": {"seed_hosts": ["es-master-0", "es-master-1", "es-master-2"]},
            "indices": {
                "template": {
                    "bankruptcy-prediction-logs": {
                        "index_patterns": ["bankruptcy-prediction-logs-*"],
                        "settings": {
                            "number_of_shards": 3,
                            "number_of_replicas": 1,
                            "refresh_interval": "5s",
                        },
                        "mappings": {
                            "properties": {
                                "@timestamp": {"type": "date"},
                                "level": {"type": "keyword"},
                                "message": {"type": "text"},
                                "logger": {"type": "keyword"},
                                "kubernetes": {
                                    "properties": {
                                        "namespace_name": {"type": "keyword"},
                                        "pod_name": {"type": "keyword"},
                                        "container_name": {"type": "keyword"},
                                    }
                                },
                            }
                        },
                    }
                }
            },
        }

    def _get_log_retention_policy(self) -> Dict[str, Any]:
        """Get log retention policies."""
        return {
            "policies": [
                {
                    "name": "delete_old_logs",
                    "phases": {
                        "hot": {
                            "actions": {
                                "rollover": {"max_size": "5GB", "max_age": "1d"}
                            }
                        },
                        "warm": {
                            "min_age": "7d",
                            "actions": {"allocate": {"number_of_replicas": 0}},
                        },
                        "delete": {"min_age": "30d"},
                    },
                }
            ]
        }


def create_monitoring_files(output_dir: str) -> Dict[str, List[str]]:
    """Create all monitoring and alerting configuration files."""
    os.makedirs(output_dir, exist_ok=True)

    monitoring_config = MonitoringConfig()
    logging_config = LoggingConfig()

    created_files = {
        "prometheus": [monitoring_config.export_prometheus_rules(output_dir)],
        "grafana": monitoring_config.export_grafana_dashboards(output_dir),
        "alerts": [monitoring_config.export_alert_rules(output_dir)],
    }

    # Create Fluentd configuration
    fluentd_file = os.path.join(output_dir, "fluentd-config.yml")
    with open(fluentd_file, "w") as f:
        yaml.dump(logging_config.fluentd_config, f, default_flow_style=False)
    created_files["logging"] = [fluentd_file]

    # Create Elasticsearch configuration
    es_file = os.path.join(output_dir, "elasticsearch-config.yml")
    with open(es_file, "w") as f:
        yaml.dump(logging_config.elasticsearch_config, f, default_flow_style=False)
    created_files["logging"].append(es_file)

    return created_files


if __name__ == "__main__":
    # Example usage
    output_dir = "./monitoring_configs"
    files = create_monitoring_files(output_dir)
    print("Created monitoring configuration files:")
    for category, file_list in files.items():
        print(f"  {category}: {file_list}")
