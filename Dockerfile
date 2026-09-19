# Multi-Stage Production Dockerfile for AI-ECG Clinical Platform
# Conforms to IEC 62304 / ISO 27799 Container Security Guidelines

FROM python:3.11-slim AS runtime

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (OpenCV headless & image support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root medical software service user
RUN groupadd -g 1001 ecggroup && \
    useradd -u 1001 -g ecggroup -m -s /bin/bash ecguser

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Ensure data and reports directories exist with appropriate permissions
RUN mkdir -p /app/data /app/reports && \
    chown -R ecguser:ecggroup /app

# Switch to non-root user
USER ecguser

# Expose Streamlit default telemetry port
EXPOSE 8501

# Container Healthcheck verifying Streamlit server status
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
