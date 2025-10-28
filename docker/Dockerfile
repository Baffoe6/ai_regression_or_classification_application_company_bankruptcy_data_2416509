# Multi-stage Docker build for bankruptcy prediction application
# Stage 1: Base image with dependencies
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 jupyter

# Copy all source code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port for API
EXPOSE 8000

# Command for development (with hot reload)
CMD ["python", "-m", "uvicorn", "simple_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production image
FROM base as production

# Copy only necessary files for production
COPY src/ ./src/
COPY CompanyBankruptcyData.csv ./
COPY simple_api.py ./
COPY run_simple.py ./

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p outputs/models outputs/plots outputs/reports logs && \
    chown -R app:app /app
USER app

# Create directories for outputs
RUN mkdir -p /app/outputs/models /app/outputs/plots /app/outputs/reports

# Expose port for API
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["python", "simple_api.py"]

# Stage 4: Testing image
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Run tests during build (optional)
RUN python -m pytest tests/ -v --tb=short || true

# Command for testing
CMD ["python", "-m", "pytest", "tests/", "-v"]