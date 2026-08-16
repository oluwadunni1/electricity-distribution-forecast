# syntax=docker/dockerfile:1

# --- Build stage: resolve + install only the `api` dependency group -------
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies first, before copying source — this layer only
# rebuilds when pyproject.toml/uv.lock change, not on every code edit.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra api --no-dev

COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra api --no-dev

# --- Runtime stage: copy only the built venv + source, nothing else -------
FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

EXPOSE 8000

# No model baked into the image — pulled live from MLflow/DagsHub at startup
# (see src/api/main.py lifespan). Credentials passed via --env-file at run time.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]