# ─────────────────────────────────────────────────────────────────────
# 🍄 Mushroom Classifier - Dockerfile
# Containerized deployment for production-ready AI application.
# ─────────────────────────────────────────────────────────────────────

# ── Base Image ─────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Set Working Directory ──────────────────────────────────────────
WORKDIR /app

# ── Environment Variables ──────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLEWEBSOCKETCOMPRESSION=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# ── Install System Dependencies ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy Requirements ──────────────────────────────────────────────
COPY requirements.txt .

# ── Install Python Dependencies ────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ──────────────────────────────────────────
COPY . .

# ── Create Necessary Directories ───────────────────────────────────
RUN mkdir -p data reports assets

# ── Expose Streamlit Port ──────────────────────────────────────────
EXPOSE 8501

# ── Health Check ───────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Run Application ────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
