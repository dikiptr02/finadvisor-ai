from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# Nama container mengikuti konvensi Docker Compose: <nama-project>-<nama-service>-1
# Sesuaikan <nama-project> dengan nama folder project kamu jika beda (cek dengan
# `docker ps` untuk konfirmasi nama container aktual di mesin kamu).
RAG_CONTAINER = "infra-rag-service-1"

default_args = {
    "owner": "finadvisor-ai",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="daily_ingest",
    description="Ingest dokumen/data baru dan update index RAG",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["ingest", "rag"],
) as dag:

    ingest_dummy_articles = BashOperator(
        task_id="ingest_dummy_articles",
        bash_command=f"docker exec {RAG_CONTAINER} python scripts/ingest_dummy.py",
    )

    # Placeholder untuk langkah masa depan begitu ada sumber data transaksi real:
    # task tarik data baru dari dashboard existing -> chunking -> index, akan
    # ditambahkan di sini sebagai task terpisah dengan dependency eksplisit.

    ingest_dummy_articles