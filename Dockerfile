# syntax=docker/dockerfile:1
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt gunicorn
# Copy application code only; never include .env, .venv or local databases.
COPY pharma/*.py ./pharma/
COPY frontend/ ./frontend/
COPY knowledge/ ./knowledge/
COPY evaluation/ ./evaluation/
# Adapt the existing local-only origin check for Railway's HTTPS domain.
# This keeps the Dockerfile usable without another source-file download.
RUN python - <<'PY'
from pathlib import Path
path = Path('/app/pharma/web.py')
text = path.read_text(encoding='utf-8')
marker = "f'http://localhost:{PORT}',"
replacement = marker + "\n        os.getenv('APP_ORIGIN') or ('https://' + os.getenv('RAILWAY_PUBLIC_DOMAIN', '')),"
assert text.count(marker) == 1, 'Unsupported web.py: review the origin check.'
path.write_text(text.replace(marker, replacement), encoding='utf-8')
PY
# Attach a Railway Volume at /data for persistent uploads and ChromaDB.
RUN mkdir -p /data/chroma_db /data/uploads \
    && ln -s /data/chroma_db /app/chroma_db \
    && ln -s /data/uploads /app/uploads
EXPOSE 8000
# First launch builds the sample index using GEMINI_API_KEY from Railway Variables.
# Later launches reuse the index on the persistent volume.
CMD ["sh", "-c", "set -e; mkdir -p /data/chroma_db /data/uploads; if [ ! -f /data/chroma_db/index.json ]; then python -m pharma index; fi; exec gunicorn pharma.web:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 240 --access-logfile - --error-logfile -"]