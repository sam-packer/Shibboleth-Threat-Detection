# Use Python 3.11 Slim (Matches your project requirement)
FROM python:3.11-slim

# Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PORT=8000
ENV UV_PYTHON_DOWNLOADS=never
ENV UV_PYTHON=python3.11

# --- Environment Variables Defaults (Will be overridden by docker-compose) ---
ENV MAXMIND_LICENSE_KEY=""
ENV POSTGRES_CONNECTION_STRING=""
ENV PASSTHROUGH_MODE="false"
ENV ENABLE_MLFLOW="false"
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_REGISTRY_URI=""
ENV DATABRICKS_HOST=""
ENV DATABRICKS_TOKEN=""
ENV MLFLOW_EXPERIMENT_PATH=""
ENV UC_CATALOG=""
ENV UC_SCHEMA=""
ENV UC_MODEL_NAME=""
# ----------------------------------------------------------------------------

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