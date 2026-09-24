# Backend image for the FastAPI RAG server.
# Built for Hugging Face Docker Spaces (port 7860, non-root uid 1000) but runs
# anywhere:  docker build -t support-api . && docker run -p 7860:7860 --env-file .env support-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# poppler-utils: pdf2image (scanned-PDF OCR); libgl/libglib: image libs used by parsers
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only torch first so FlagEmbedding / sentence-transformers don't pull CUDA wheels (~5 GB)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
RUN pip install --no-deps -e . && chown -R user:user /home/user

USER user
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/supported-formats')"
CMD ["sh", "-c", "uvicorn file_processor.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
