import mlflow
import torch
import torch.nn as nn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("transaction-classifier")

# Model dummy 1 layer -- isinya tidak penting, cuma supaya log_model tidak error
dummy_model = nn.Linear(1, 1)

with mlflow.start_run(run_name="deliberately_bad_for_gate_testing"):
    mlflow.set_tag("purpose", "TESTING ONLY - bukan model asli, untuk validasi CI gate")
    mlflow.log_metric("accuracy_held_out", 0.01)  # sengaja jauh lebih buruk dari production (0.1779)
    mlflow.pytorch.log_model(dummy_model, artifact_path="model", registered_model_name="transaction-classifier")

print("Fake run dengan accuracy_held_out=0.01 berhasil di-log & registrasi sebagai versi baru.")