from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

ML_CONTAINER = "infra-ml-service-1"
FORECAST_CONTAINER = "infra-forecast-service-1"

default_args = {
    "owner": "finadvisor-ai",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="weekly_eval_retrain",
    description="Retrain semua model ML, registrasi ke MLflow, jalankan CI gate",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["mlops", "retrain"],
) as dag:

    retrain_classifier = BashOperator(
        task_id="retrain_classifier",
        bash_command=f"docker exec {ML_CONTAINER} python scripts/train_classifier.py",
    )

    retrain_anomaly = BashOperator(
        task_id="retrain_anomaly",
        bash_command=f"docker exec {ML_CONTAINER} python scripts/train_anomaly.py",
    )

    retrain_clustering = BashOperator(
        task_id="retrain_clustering",
        bash_command=f"docker exec {ML_CONTAINER} python scripts/train_clustering.py",
    )

    retrain_forecast = BashOperator(
        task_id="retrain_forecast",
        bash_command=f"docker exec {FORECAST_CONTAINER} python scripts/run_comparison.py",
    )

    check_classifier_quality_gate = BashOperator(
        task_id="check_classifier_quality_gate",
        bash_command=f"docker exec {ML_CONTAINER} python /mlops/ci-gate/check_model_quality.py",
    )

    # Matikan paralelisme untuk mengurangi beban CPU (jalan berurutan)
    retrain_classifier >> check_classifier_quality_gate
    retrain_classifier >> retrain_anomaly >> retrain_clustering >> retrain_forecast