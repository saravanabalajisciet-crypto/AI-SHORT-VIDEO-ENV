FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY app.py .
COPY environment.py .
COPY models.py .
COPY client.py .
COPY inference.py .
COPY openenv.yaml .

# HF Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
