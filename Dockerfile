# Backend image for the FastAPI RAG server.
# Deployed to Google Cloud Run (see DEPLOYMENT.md) but runs anywhere:
#   docker build -t support-api . && docker run -p 8080:8080 --env-file .env support-api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

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

# Bake BGE-M3 (~2.3 GB) into the image so cold starts don't re-download it.
# Downloaded to a plain directory (not the HF cache) and selected via EMBEDDING_MODEL, because
# FlagEmbedding re-runs snapshot_download for hub names and would fetch the ONNX weights too.
ENV EMBEDDING_MODEL=/home/user/models/bge-m3
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', local_dir='$EMBEDDING_MODEL', ignore_patterns=['onnx/*', '*.onnx*', 'imgs/*', '*.jpg'])" \
    && rm -rf "$EMBEDDING_MODEL/.cache" && chown -R user:user /home/user/models

COPY --chown=user . .
RUN pip install --no-deps -e . && chown -R user:user /home/user/app

USER user
EXPOSE 8080
CMD ["sh", "-c", "uvicorn file_processor.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
