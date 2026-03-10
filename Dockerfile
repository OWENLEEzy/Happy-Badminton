FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies (locked, skip dev, skip project install to avoid stale scripts)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code and data
COPY src/ src/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY data/ data/
COPY config.yaml ./
COPY main.py ./

# Train models at build time (avoids storing binary files in git)
RUN mkdir -p models && \
    uv run python scripts/train_simplified.py && \
    uv run python scripts/train_set_count.py

# HuggingFace Spaces requires port 7860
EXPOSE 7860
ENV PORT=7860

# Run with gunicorn (1 worker — free tier CPU limit)
CMD ["uv", "run", "gunicorn", "frontend.app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--timeout", "120"]
