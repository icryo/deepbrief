FROM python:3.12-slim

WORKDIR /app

# System deps (ffmpeg for video assembly, fonts for subtitles)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install researcher deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install paper2video deps
COPY paper2video/requirements.txt paper2video/requirements.txt
RUN pip install --no-cache-dir -r paper2video/requirements.txt

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy source
COPY src/ src/
COPY paper2video/ paper2video/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create runtime directories
RUN mkdir -p data/weeks paper2video/result && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["./entrypoint.sh"]
