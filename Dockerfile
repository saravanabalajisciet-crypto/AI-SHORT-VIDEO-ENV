FROM python:3.11-slim-bullseye

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies with retry-safe flags
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Copy application files
COPY app.py .
COPY environment.py .
COPY models.py .
COPY client.py .
COPY inference.py .
COPY strategy_engine.py .
COPY gradio_ui.py .
COPY test_environment.py .
COPY openenv.yaml .
COPY video_dataset.json .
COPY server/ ./server/

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
