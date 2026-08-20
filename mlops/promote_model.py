import sys
from mlflow.tracking import MlflowClient

# Gunakan localhost jika dijalankan di mesin host, gunakan mlflow jika di dalam container
MLFLOW_TRACKING_URI = "http://mlflow:5000"

MODEL_NAME = "transaction-classifier"
VERSION_TO_PROMOTE = "1"

def main():
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=VERSION_TO_PROMOTE,
        stage="Production",
    )
    print(f"{MODEL_NAME} versi {VERSION_TO_PROMOTE} sekarang berstatus Production")

if __name__ == "__main__":
    main()
