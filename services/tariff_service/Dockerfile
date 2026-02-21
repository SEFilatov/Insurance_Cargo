FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY app /app/app
COPY config /app/config

ENV PYTHONUNBUFFERED=1 \
    TARIFF_CONFIG_PATH=/app/config/tariff_config.json

# Yandex Serverless Containers provides PORT env var (8080 by default)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
