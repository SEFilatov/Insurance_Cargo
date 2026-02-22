# Dialog Service (Yandex Serverless Containers)

This service provides `/chat` endpoint for a web widget.
It uses Yandex AI Studio (OpenAI-compatible API) to:
- extract/confirm inputs for insurance quote
- call your Tariff Engine `/quote`
- return a short dialog response to the widget

## Env vars

Required:
- YANDEX_FOLDER_ID=<your folder id>
- YANDEX_API_KEY=<AI Studio API key>
- TARIFF_URL=https://<tariff-container>.containers.yandexcloud.net/quote

Optional:
- YANDEX_BASE_URL=https://llm.api.cloud.yandex.net/v1
- YANDEX_MODEL_URI=gpt://<folder_id>/yandexgpt-lite
- ALLOW_ORIGINS=https://<your-gh-pages-domain> (comma-separated). Default: *
- TARIFF_BEARER=<IAM token> (only if tariff container requires Authorization Bearer)

## Local run

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export YANDEX_FOLDER_ID=...
export YANDEX_API_KEY=...
export TARIFF_URL=...
uvicorn app.main:app --reload --port 8080

## Test

curl http://localhost:8080/health

curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"session_id":"test","message":"Хочу застраховать перевозку. Везу микроволновки на 2 млн, б/у, франшиза 50 тыс, рефа нет, зона РФ."}'
