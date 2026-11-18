# Contexte de build : racine du dépôt (docker compose s'en charge)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# torch CPU (image ~4x plus légère que la variante CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ml /app/ml
RUN pip install --no-cache-dir -e /app/ml
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend /app/backend

WORKDIR /app/backend
ENV AEROVISION_MODELS_DIR=/app/models \
    AEROVISION_STORAGE_DIR=/app/data/storage \
    AEROVISION_DATABASE_URL=sqlite:////app/data/aerovision.db \
    TORCH_HOME=/app/models/.torch
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
