# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pipeline.py query.py postprocess.py extract.py ./

# volumes for data, model weights, and outputs
# mount at runtime — not baked into image
VOLUME ["/data", "/weights", "/chroma_db", "/query_results"]

# default to API server
EXPOSE 8000
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]