# Yandex Cloud deploy plan (MVP)

## Required GitHub secrets
- `YC_SA_KEY_JSON`
- `YC_CLOUD_ID`
- `YC_FOLDER_ID`
- `YC_REGISTRY_ID`

## High-level flow
1. Build and lint `services/tariff_service`.
2. Push image to Yandex Container Registry.
3. Deploy new revision to Yandex Serverless Container.

## IAM notes
- Create a service account for GitHub Actions.
- Grant minimal roles:
  - `container-registry.images.pusher`
  - `serverless-containers.editor`
  - `viewer` (if needed for metadata lookups)
