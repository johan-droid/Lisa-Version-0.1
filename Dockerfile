FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=127.0.0.1 \
    LISA_ALLOW_REMOTE_BIND=false \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        g++ \
        git \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md main.py config.yaml ./
COPY brain ./brain
COPY conductor ./conductor
COPY evolution ./evolution
COPY interfaces ./interfaces
COPY lisa ./lisa
COPY memory ./memory
COPY personal ./personal
COPY safety ./safety
COPY tools ./tools
COPY utils ./utils
COPY mcp_servers.json ./mcp_servers.json
COPY skills_manifest.json ./skills_manifest.json

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "main.py"]
