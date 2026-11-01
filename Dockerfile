FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY app ./app
COPY data/chunks ./data/chunks
COPY data/cleaned ./data/cleaned
COPY data/vector/milvus.db ./data/vector/milvus.db
COPY data/sources.json ./data/sources.json
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
