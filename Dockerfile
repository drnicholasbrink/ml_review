FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ML_REVIEW_RUNTIME_DIR=/app/runtime

WORKDIR /app
COPY requirements_.txt requirements_app.lock ./
RUN pip install --no-cache-dir -r requirements_app.lock
COPY . .
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/ready', timeout=3)"]
# Runtime artifacts are intentionally filesystem-backed. One worker avoids concurrent
# writes to a project while preserving the supported single-reviewer deployment model.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "1", "--timeout", "3600", "--access-logfile", "-", "--error-logfile", "-", "wsgi:app"]
