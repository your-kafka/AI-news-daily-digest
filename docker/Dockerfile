# syntax=docker/dockerfile:1.7
# One image for every app service: the API, the Celery workers, beat and the migration job.

FROM ghcr.io/astral-sh/uv:0.12.21 AS uv

# ---- build: resolve and install dependencies into /app/.venv ----
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first, so editing source code doesn't reinstall them.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ---- runtime: just Python, the virtualenv and the code ----
FROM python:3.12-slim-bookworm AS runtime
RUN groupadd --system --gid 10001 lodestar \
    && useradd --system --uid 10001 --gid lodestar --home-dir /app --no-create-home lodestar \
    && mkdir -p /models \
    && chown lodestar:lodestar /models

WORKDIR /app
COPY --from=builder --chown=lodestar:lodestar /app/.venv ./.venv
COPY --chown=lodestar:lodestar src ./src
COPY --chown=lodestar:lodestar migrations ./migrations
COPY --chown=lodestar:lodestar config ./config
COPY --chown=lodestar:lodestar pyproject.toml ./

# Models (fastembed, Hugging Face tokenizer, tiktoken) download once into the /models volume.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_CACHE_DIR=/models/fastembed \
    HF_HOME=/models/huggingface \
    TIKTOKEN_CACHE_DIR=/models/tiktoken

USER lodestar
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status != 200)"]

CMD ["uvicorn", "ai_news.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
