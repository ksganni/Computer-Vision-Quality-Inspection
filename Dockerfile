FROM python:3.11-slim

WORKDIR /app

# System libs needed by OpenCV / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download YOLO weights into the image (makes first request faster)
RUN python scripts/download_model.py || true

EXPOSE 8000 8501

# Default: FastAPI. Override in docker-compose for Streamlit.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
