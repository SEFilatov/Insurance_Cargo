# tariff_service

FastAPI service for tariff-related AI platform logic.

## Local run

```bash
cd services/tariff_service
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Docker run

```bash
docker build -t tariff-service:local .
docker run --rm -p 8080:8080 tariff-service:local
```
