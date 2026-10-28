# CodeLens AI - Production Dockerfile for Web Publishing
FROM python:3.11-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONSAFEPATH=1 \
    PORT=8000 \
    CODELENS_PROVIDER=auto

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose port
EXPOSE 8000

# Start FastAPI server via uvicorn
CMD ["sh", "-c", "uvicorn codelens.web.app:app --host 0.0.0.0 --port ${PORT}"]
