FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ML_REVIEW_RUNTIME_DIR=/app/runtime

WORKDIR /app
COPY requirements_.txt ./
RUN pip install --no-cache-dir -r requirements_.txt
COPY . .
EXPOSE 5000
CMD ["flask", "--app", "wsgi:app", "run", "--host", "0.0.0.0", "--port", "5000"]
