FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ML_REVIEW_RUNTIME_DIR=/app/runtime

WORKDIR /app
COPY requirements_.txt ./
RUN pip install --no-cache-dir -r requirements_.txt
COPY . .
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"]
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
