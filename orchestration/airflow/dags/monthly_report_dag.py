from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import mlflow


def generate_report(**context):
    mlflow.set_tracking_uri("http://mlflow:5000")

    experiments = ["transaction-classifier", "anomaly-detection", "user-segmentation", "expense-forecasting"]
    report_lines = [f"=== Laporan Bulanan FinAdvisor AI — {context['ds']} ==="]

    for exp_name in experiments:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            report_lines.append(f"\n{exp_name}: belum ada eksperimen tercatat")
            continue

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs.empty:
            report_lines.append(f"\n{exp_name}: belum ada run tercatat")
            continue

        latest_run = runs.iloc[0]
        report_lines.append(f"\n{exp_name} (run terbaru: {latest_run['run_id'][:8]}):")
        metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
        for col in metric_cols:
            metric_name = col.replace("metrics.", "")
            report_lines.append(f"  - {metric_name}: {latest_run[col]:.4f}")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(f"/opt/airflow/dags/../monthly_report_{context['ds']}.txt", "w") as f:
        f.write(report_text)


default_args = {
    "owner": "finadvisor-ai",
    "retries": 1,
}

with DAG(
    dag_id="monthly_report",
    description="Agregasi metric dari semua eksperimen MLflow jadi laporan bulanan",
    default_args=default_args,
    schedule_interval="@monthly",
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["reporting"],
) as dag:

    generate_monthly_report = PythonOperator(
        task_id="generate_monthly_report",
        python_callable=generate_report,
    )