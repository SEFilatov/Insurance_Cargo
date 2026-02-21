# Insurance Cargo AI Platform Monorepo (MVP)

## Monorepo layout

```text
services/            # Deployable services
  tariff_service/    # FastAPI service
libs/                # Shared libraries (schemas, clients, utils)
docs/                # Platform docs and runbooks
infra/yandex-cloud/  # IaC/deploy scaffolding for YC
.github/             # Workflows and collaboration templates
```

## Quick start (tariff_service)

```bash
cd services/tariff_service
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

## CI/CD overview

Workflow: `.github/workflows/tariff_service_ci_cd.yml`

- Runs lint and docker build for `services/tariff_service`.
- Uses Docker build context `services/tariff_service` (keeps monorepo out of image context).
- On `main`, builds and pushes image to Yandex Container Registry.
- Deploys a new revision to Yandex Serverless Container (template command, customize container name/resources).

## Required GitHub Secrets for deploy

- `YC_SA_KEY_JSON`
- `YC_CLOUD_ID`
- `YC_FOLDER_ID`
- `YC_REGISTRY_ID`

## Yandex Cloud deployment plan (summary)

1. Create Container Registry and grant push rights to CI service account.
2. Create Serverless Container target (`tariff-service`).
3. Configure GitHub Actions secrets listed above.
4. Run workflow on `main` push to build, push, and deploy new revision.
5. Assign minimal roles to CI service account (registry push + serverless deploy).

See detailed notes in `docs/yandex-cloud-deploy-plan.md`.
