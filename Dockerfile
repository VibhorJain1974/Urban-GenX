# Urban-GenX | Dockerfile (CPU-only, 12GB RAM safe)
# Build:  docker build -t urban-genx .
# Run:    docker run -p 8501:8501 -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints urban-genx

FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY models/ ./models/
COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p data/raw checkpoints

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default: launch Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
