FROM apache/airflow:2.10.3-python3.11

USER root
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir mlflow==2.17.2