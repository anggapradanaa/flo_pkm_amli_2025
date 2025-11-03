FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
ENV PORT=8080
EXPOSE 8080

# Run application
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --max-requests 5000 --max-requests-jitter 100 --worker-class sync --threads 4
