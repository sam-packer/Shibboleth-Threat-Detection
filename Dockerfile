# Use Python 3.9 Slim (Consider upgrading to 3.12-slim if pyproject.toml requires newer python)
FROM python:3.11-slim

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Explicitly set PORT to 8000 so run_api.py aligns with Caddy/Docker Compose
ENV PORT=8000

# --- ADD THIS HERE ---
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PYTHON=python3.11

# Copy project definition files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Sync the project
RUN uv sync --frozen --no-dev

# Expose the port
EXPOSE 8000

# Run the production entry point defined in pyproject.toml
CMD ["uv", "run", "api-prod"]