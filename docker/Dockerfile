FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/external checkpoints/best_model checkpoints/logs

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "-m", "layers.cli.main_cli", "--help"]