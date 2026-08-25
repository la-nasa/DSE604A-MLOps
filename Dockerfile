FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/

# Create non-root user
RUN useradd -m -u 1000 mlflow && \
    mkdir -p /app/mlruns && \
    chown -R mlflow:mlflow /app

USER mlflow

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

# Health check hits the Flask app's own lightweight /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run the prediction API (Lab 8/9 packaging)
CMD ["python", "src/app.py"]
