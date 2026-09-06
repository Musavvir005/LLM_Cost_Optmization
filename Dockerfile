# CPU-only AD-BoN gateway image.  Build context must be the repository root.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Pin runtime dependencies so an image rebuild remains reproducible.  The PyPI
# Linux torch wheel is CPU-compatible; no CUDA runtime is included in this image.
RUN python -m pip install --upgrade pip==24.3.1 \
    && python -m pip install \
        torch==2.5.1 \
        transformers==4.48.3 \
        numpy==1.26.4 \
        requests==2.32.3 \
        openai==1.61.0 \
        google-generativeai==0.8.4 \
        fastapi==0.115.6 \
        "uvicorn[standard]==0.34.0" \
        flask==3.1.0 \
        pydantic==2.10.5 \
        pillow==11.1.0 \
        pypdf==5.1.0 \
        joblib==1.4.2

COPY . /app

RUN mkdir -p /app/.cache/huggingface /app/logs /app/data/cache \
    && test -f /app/semantic_cache.json || printf '[]\n' > /app/semantic_cache.json

EXPOSE 8000

# The FastAPI service is the long-running gateway.  Use docker compose run for
# one-off jobs such as verify_docker_stack.py.
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
