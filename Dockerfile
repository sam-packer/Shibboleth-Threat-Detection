FROM python:3.14 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=python3.14

COPY pyproject.toml uv.lock ./

ENV TORCH_SELECTOR=torch-cpu
RUN uv sync --frozen --no-dev --extra $TORCH_SELECTOR --no-install-project

COPY . .

RUN uv sync --frozen --no-dev --extra $TORCH_SELECTOR

FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY --from=builder /app/.venv /app/.venv

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON=python3.14

EXPOSE 8000
CMD ["uv", "run", "api-prod"]
