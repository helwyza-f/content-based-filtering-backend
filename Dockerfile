FROM python:3.10-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirement file & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port for Flask
EXPOSE 5000

# Run app with host 0.0.0.0 (so it's accessible from outside the container)
CMD ["python", "app.py"]