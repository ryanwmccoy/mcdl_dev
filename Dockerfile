FROM python:3.11-slim

# Install Java (required by PySpark)
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jdk-headless curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=${JAVA_HOME}/bin:${PATH}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY models /app/models
ENV PYTHONPATH=/app/src

# Default command is a fetcher; override per service in docker-compose.yml
CMD ["python", "/app/src/fetcher.py"]
