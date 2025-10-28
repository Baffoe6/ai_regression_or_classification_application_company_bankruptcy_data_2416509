# Deployment Guide

## Production Deployment Options

### Option 1: AWS ECS (Recommended)

```yaml
# ecs-task-definition.json
{
  "family": "bankruptcy-prediction",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::your-account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::your-account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "bankruptcy-prediction-api",
      "image": "your-registry/bankruptcy-prediction:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "API_HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "API_PORT",
          "value": "8000"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/bankruptcy-prediction",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Option 2: Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bankruptcy-prediction
  labels:
    app: bankruptcy-prediction
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bankruptcy-prediction
  template:
    metadata:
      labels:
        app: bankruptcy-prediction
    spec:
      containers:
      - name: api
        image: your-registry/bankruptcy-prediction:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: bankruptcy-prediction-service
spec:
  selector:
    app: bankruptcy-prediction
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bankruptcy-prediction-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.yourcompany.com
    secretName: bankruptcy-prediction-tls
  rules:
  - host: api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bankruptcy-prediction-service
            port:
              number: 80
```

### Option 3: Google Cloud Run

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: bankruptcy-prediction
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 100
      containers:
      - image: gcr.io/your-project/bankruptcy-prediction:latest
        ports:
        - containerPort: 8000
        env:
        - name: API_HOST
          value: "0.0.0.0"
        - name: API_PORT
          value: "8000"
        resources:
          limits:
            memory: 1Gi
            cpu: 1000m
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

## Environment Configuration

### Production Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_TYPE=ensemble
MAX_FEATURES=30
ENABLE_HYPERPARAMETER_TUNING=false
ENABLE_FEATURE_SELECTION=true

# Security
CORS_ORIGINS=["https://yourapp.com", "https://api.yourcompany.com"]
API_KEY_REQUIRED=true
JWT_SECRET_KEY=your-secret-key-here

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO

# Database (if applicable)
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0

# Cloud Storage
AWS_S3_BUCKET=your-model-bucket
AWS_REGION=us-east-1
```

### SSL/TLS Configuration

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.yourcompany.com;
    
    ssl_certificate /etc/ssl/certs/bankruptcy-prediction.crt;
    ssl_certificate_key /etc/ssl/private/bankruptcy-prediction.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://bankruptcy-prediction-service:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /health {
        proxy_pass http://bankruptcy-prediction-service:8000/health;
        access_log off;
    }
}
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'bankruptcy-prediction'
    static_configs:
      - targets: ['bankruptcy-prediction-service:9090']
    metrics_path: /metrics
    scrape_interval: 5s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Bankruptcy Prediction API",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "model_accuracy_score"
          }
        ]
      }
    ]
  }
}
```

## Security Hardening

### Docker Security

```dockerfile
# Multi-stage build for security
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.9-slim
# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copy from builder stage
COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app . .

# Security: Remove unnecessary packages
RUN apt-get update && apt-get remove -y wget curl && rm -rf /var/lib/apt/lists/*

# Set secure environment
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/home/app

EXPOSE 8000
CMD ["python", "simple_api.py"]
```

### API Security Headers

```python
# Add to FastAPI app
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.yourcompany.com", "*.yourcompany.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

## Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bankruptcy-prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bankruptcy-prediction
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Testing

```python
# load_test.py
import asyncio
import aiohttp
import time
import random

async def make_request(session, url, data):
    try:
        async with session.post(url, json=data) as response:
            return await response.json()
    except Exception as e:
        return {"error": str(e)}

async def load_test(num_requests=1000, concurrent=50):
    url = "http://localhost:8000/predict"
    
    # Sample data
    sample_data = {
        "features": {f"feature_{i}": random.random() for i in range(30)}
    }
    
    connector = aiohttp.TCPConnector(limit=concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        start_time = time.time()
        
        for _ in range(num_requests):
            task = make_request(session, url, sample_data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        successful_requests = sum(1 for r in results if "error" not in r)
        total_time = end_time - start_time
        rps = num_requests / total_time
        
        print(f"Completed {num_requests} requests in {total_time:.2f}s")
        print(f"Success rate: {successful_requests/num_requests*100:.1f}%")
        print(f"Requests per second: {rps:.1f}")

if __name__ == "__main__":
    asyncio.run(load_test())
```

## Backup and Recovery

### Model Backup Strategy

```bash
#!/bin/bash
# backup_models.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/models"
S3_BUCKET="your-model-backups"

# Create backup directory
mkdir -p $BACKUP_DIR/$DATE

# Backup trained models
cp -r models/ $BACKUP_DIR/$DATE/
cp src/config.py $BACKUP_DIR/$DATE/
cp requirements.txt $BACKUP_DIR/$DATE/

# Compress backup
tar -czf $BACKUP_DIR/bankruptcy_prediction_$DATE.tar.gz -C $BACKUP_DIR $DATE

# Upload to S3
aws s3 cp $BACKUP_DIR/bankruptcy_prediction_$DATE.tar.gz s3://$S3_BUCKET/

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: bankruptcy_prediction_$DATE.tar.gz"
```

### Database Backup (if applicable)

```sql
-- PostgreSQL backup
pg_dump -h localhost -U username -d bankruptcy_db > backup_$(date +%Y%m%d).sql

-- Restore
psql -h localhost -U username -d bankruptcy_db < backup_20241201.sql
```

## Troubleshooting Guide

### Common Production Issues

#### High Memory Usage
```bash
# Monitor memory usage
kubectl top pods
docker stats

# Solution: Increase memory limits or optimize model
resources:
  limits:
    memory: "2Gi"
```

#### Slow Response Times
```bash
# Check API performance
curl -w "@curl-format.txt" -s -o /dev/null http://api.yourcompany.com/health

# Solution: Enable caching or add more replicas
```

#### Model Loading Errors
```bash
# Check logs
kubectl logs deployment/bankruptcy-prediction
docker logs container_name

# Solution: Verify model file integrity
```

### Health Check Endpoints

```python
# Extended health check
@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "model_loaded": model is not None,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent(),
        "disk_usage": psutil.disk_usage('/').percent
    }
```

This deployment guide provides comprehensive instructions for production deployment across multiple cloud platforms with proper security, monitoring, and scaling configurations.