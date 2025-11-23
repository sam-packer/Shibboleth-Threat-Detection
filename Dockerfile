FROM python:3.11-slim

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# uv runtime settings
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=python3.11

# Copy project definition
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
