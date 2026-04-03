FROM python:3.12-slim

WORKDIR /app

# System deps for paper2video (ffmpeg for video assembly, poppler for PDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install researcher deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install paper2video deps (separate requirements, shares some with researcher)
COPY paper2video/requirements.txt paper2video/requirements.txt
RUN pip install --no-cache-dir -r paper2video/requirements.txt

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy source
COPY src/ src/
COPY paper2video/ paper2video/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create data + result directories with correct ownership
RUN mkdir -p data/weeks paper2video/result && chown -R appuser:appuser /app

USER appuser

EXPOSE 8888 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888/api/status')"

ENTRYPOINT ["./entrypoint.sh"]
