FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY config /app/config

# Yandex Serverless Containers expects the app to listen on $PORT
ENV PORT=8080
ENV HOST=0.0.0.0

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
