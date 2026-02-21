# Monorepo architecture (MVP)

## Goals
- Keep service boundaries clear.
- Share reusable code under `libs/`.
- Separate product code from infrastructure and platform automation.

## Directory conventions
- `services/`: deployable services.
- `libs/`: shared internal libraries.
- `infra/yandex-cloud/`: infrastructure blueprints and future Terraform modules.
- `.github/`: CI/CD and contribution templates.
- `docs/`: architecture, runbooks, and operations notes.
