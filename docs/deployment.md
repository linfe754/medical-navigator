# Production deployment

The production architecture is Docker → Azure Container Registry → Azure Container Apps → the `findhealthcare.au` custom domain. This summary omits resource identifiers and credentials.

```text
Docker image → Azure Container Registry → Azure Container Apps → findhealthcare.au
```

The application runs as a container in Azure Container Apps. Its image is stored in Azure Container Registry. The `Dockerfile` in this repository defines the image: Python 3.12, locked production dependencies installed with `uv`, and Uvicorn serving FastAPI on port 8000.

The repository does not include Azure resource definitions or a deployment workflow. The application exposes `/health` for an HTTP health check.
