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

# Copy CPU-only requirements
COPY requirements-docker.txt .

# Install ONLY from requirements-docker.txt (no pyproject deps)
RUN uv venv && uv pip install -r requirements-docker.txt

# Copy all project files
COPY . .

# Install your project itself (editable install)
RUN uv pip install -e .

FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON=python3.14

EXPOSE 8000
CMD ["uv", "run", "api-prod"]
