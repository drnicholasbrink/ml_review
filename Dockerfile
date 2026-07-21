FROM node:22-alpine AS atlas-build

WORKDIR /build
COPY package.json package-lock.json vite.config.js ./
RUN npm ci
COPY frontend ./frontend
RUN npm run build:atlas

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ML_REVIEW_RUNTIME_DIR=/app/runtime

WORKDIR /app
COPY requirements_.txt ./
RUN pip install --no-cache-dir -r requirements_.txt
COPY . .
COPY --from=atlas-build /build/ml_review_app/static/atlas ./ml_review_app/static/atlas
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
