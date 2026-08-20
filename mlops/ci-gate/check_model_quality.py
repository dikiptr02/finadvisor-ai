import sys

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_NAME = "transaction-classifier"
METRIC_KEY = "accuracy_held_out"
MINIMUM_ACCEPTABLE_SCORE = 0.15  # threshold absolut minimum, terlepas dari model produksi


def get_latest_run_metric(client: MlflowClient, run_id: str, metric_key: str) -> float | None:
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_key)


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Ambil versi model TERBARU yang baru saja di-log (belum tentu "Production")
    latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not latest_versions:
        print("Tidak ada model terdaftar sama sekali. Gate lolos otomatis (first run).")
        sys.exit(0)

    latest_version = max(latest_versions, key=lambda v: int(v.version))
    candidate_score = get_latest_run_metric(client, latest_version.run_id, METRIC_KEY)

    if candidate_score is None:
        print(f"GATE GAGAL: metric '{METRIC_KEY}' tidak ditemukan di run terbaru.")
        sys.exit(1)

    print(f"Candidate model (version {latest_version.version}): {METRIC_KEY} = {candidate_score:.4f}")

    # Cari model yang SAAT INI berstatus "Production"
    production_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not production_versions:
        # Belum ada model production sama sekali -- cukup cek threshold absolut minimum
        if candidate_score >= MINIMUM_ACCEPTABLE_SCORE:
            print(f"Belum ada model Production. Candidate lolos threshold minimum ({MINIMUM_ACCEPTABLE_SCORE}).")
            sys.exit(0)
        else:
            print(f"GATE GAGAL: candidate ({candidate_score:.4f}) di bawah threshold minimum ({MINIMUM_ACCEPTABLE_SCORE}).")
            sys.exit(1)

    production_version = production_versions[0]
    production_score = get_latest_run_metric(client, production_version.run_id, METRIC_KEY)

    print(f"Production model (version {production_version.version}): {METRIC_KEY} = {production_score:.4f}")

    if candidate_score >= production_score:
        print("GATE LOLOS: candidate model >= model production saat ini.")
        sys.exit(0)
    else:
        print(f"GATE GAGAL: candidate ({candidate_score:.4f}) lebih buruk dari production ({production_score:.4f}).")
        sys.exit(1)


if __name__ == "__main__":
    main()