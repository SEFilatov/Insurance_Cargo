# Cargo Insurance Tariff Engine (Yandex Serverless Containers)

Это небольшой детерминированный сервис для расчёта страховой премии и авто-андеррайтинга.

## Основные идеи

- Тарифы/коэффициенты/формула **не должны попадать в LLM**.
- LLM/виджет передаёт только входные параметры (без секретов) и получает обратно только итоговую премию и публичные признаки.
- Секретные ставки и коэффициенты хранятся внутри сервиса (конфиг) и не возвращаются наружу.

## Эндпоинты

- `GET /health` — проверка здоровья.
- `POST /quote` — оценка (AUTO_OK/REFER/DECLINE) + расчёт премии для AUTO_OK.

## Конфигурация тарифа

По умолчанию сервис читает конфиг из `TARIFF_CONFIG_PATH` (в образе это `/app/config/tariff_config.json`).

Формат конфига: см. `config/tariff_config.json`.

## Локальный запуск

```bash
cd tariff_service
docker build -t tariff-engine:local .
docker run --rm -p 8080:8080 -e PORT=8080 tariff-engine:local

curl http://localhost:8080/health
```

Пример запроса:

```bash
curl -X POST http://localhost:8080/quote \
  -H 'Content-Type: application/json' \
  -d '{
    "cargo_class_id": "CARGO003",
    "sum_insured_rub": 10000000,
    "condition": "NEW",
    "franchise_rub": 50000,
    "is_reefer": false,
    "route_zone": "РФ"
  }'
```

## Деплой в Yandex Serverless Containers

Yandex Serverless Containers требует, чтобы приложение слушало порт из переменной окружения `PORT` (по умолчанию 8080). См. quickstart по сервису. 

### 1) Собрать образ и запушить в Yandex Container Registry

1. Создайте registry (Container Registry) и репозиторий.
2. Авторизуйте docker:

```bash
yc container registry configure-docker
```

3. Соберите и запушьте образ:

```bash
export REGISTRY_ID=<your_registry_id>
export IMAGE=cr.yandex/${REGISTRY_ID}/tariff-engine:2026-02-18_v1

docker build -t ${IMAGE} .
docker push ${IMAGE}
```

### 2) Создать Serverless Container и revision

```bash
yc serverless container create --name tariff-engine

# Деплой ревизии
# Важно: если registry приватный, укажите service account с ролью container-registry.images.puller

yc serverless container revision deploy \
  --container-name tariff-engine \
  --image ${IMAGE} \
  --cores 1 \
  --memory 512MB \
  --concurrency 4 \
  --execution-timeout 10s \
  --service-account-id <service_account_ID>
```

(Команды/параметры приведены в официальном quickstart Serverless Containers.)

### 3) Вызов контейнера

```bash
# Узнайте url контейнера (или возьмите из вывода create):
yc serverless container get --name tariff-engine

# Пример вызова (нужен IAM токен):
curl \
  --header "Authorization: Bearer $(yc iam create-token)" \
  https://<container-id>.containers.yandexcloud.net/health
```

## Что дальше

- В проде лучше не включать тарифный JSON в образ, а хранить версионно в Object Storage и подгружать при старте (или через безопасный reload).
- Для полной цепочки (виджет → LLM → tool-calling → quote) рекомендуется ставить API Gateway перед публичным backend, а tariff-engine держать в приватной сети.
