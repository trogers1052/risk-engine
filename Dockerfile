FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY risk_engine/ ./risk_engine/
COPY config/ ./config/

# Set Python path
ENV PYTHONPATH=/app

# Default environment variables
ENV RISK_REDIS_HOST=localhost
ENV RISK_REDIS_PORT=6379
ENV RISK_TIMESCALE_HOST=localhost
ENV RISK_TIMESCALE_PORT=5432
ENV RISK_LOG_LEVEL=INFO

# The risk engine is primarily used as a library imported by decision-engine
# This CMD is for testing/debugging only
CMD ["python", "-c", "from risk_engine import RiskAdapter; print('Risk Engine loaded successfully')"]
