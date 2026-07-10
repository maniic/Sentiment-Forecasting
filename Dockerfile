# Container image for the live demo — Fly.io, Hugging Face Spaces (Docker
# SDK), Railway, or any Docker host. Deliberately built on
# requirements-deploy.txt (no torch/transformers) to stay small and fast to
# cold-start; the app's lexicon fallback handles sentiment scoring instead.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY server.py .
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY design/ ./design/

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
