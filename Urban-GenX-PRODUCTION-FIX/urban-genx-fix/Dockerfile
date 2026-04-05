# ============================================================
# Urban-GenX | Dockerfile (CPU-only, 12GB RAM safe)
# ============================================================
# Build:  docker build -t urban-genx .
# Run:    docker run -p 8501:8501 \
#           -v $(pwd)/data:/app/data \
#           -v $(pwd)/checkpoints:/app/checkpoints \
#           urban-genx
# ============================================================

FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Work directory
WORKDIR /app

# Install Python deps (no GPU)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch dashboard
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
