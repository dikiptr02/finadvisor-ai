import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.clustering.features import build_user_features

N_CLUSTERS = 4  # sesuai jumlah segmen yang kita tanam di generator (hemat/boros/investor/standar)
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "user-segmentation"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="kmeans"):
        mlflow.log_param("n_clusters", N_CLUSTERS)

        df = pd.read_csv("shared_data/synthetic_transactions.csv")

        features = build_user_features(df)
        print(f"Jumlah user: {len(features)}")

        y_true_labels = features["segment_ground_truth"]
        X = features.drop(columns=["segment_ground_truth"])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # KMeans tidak tahu nama segmen kita (dia cuma kasih label 0,1,2,3 acak) --
        # ARI dan NMI mengukur kesesuaian STRUKTUR pengelompokan, bukan mencocokkan
        # angka label secara langsung. Ini metrik yang tepat untuk unsupervised clustering.
        label_encoder = LabelEncoder()
        y_true_encoded = label_encoder.fit_transform(y_true_labels)

        ari = adjusted_rand_score(y_true_encoded, cluster_labels)
        nmi = normalized_mutual_info_score(y_true_encoded, cluster_labels)
        mlflow.log_metrics({"ari": ari, "nmi": nmi})

        print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
        print("(1.0 = cocok sempurna dengan segmen asli, 0.0 = tidak lebih baik dari acak)")

        # Cross-tab: lihat cluster mana yang paling dominan diisi segmen apa
        crosstab = pd.crosstab(y_true_labels, cluster_labels, rownames=["segmen_asli"], colnames=["cluster_kmeans"])
        print("\nCross-tabulation (segmen asli vs cluster hasil KMeans):")
        print(crosstab)

        crosstab.to_csv("cluster_crosstab.csv")
        mlflow.log_artifact("cluster_crosstab.csv")

        # Karakteristik tiap cluster -- rata-rata fitur per cluster, membantu "menamai"
        # cluster secara manual (misal cluster 2 proporsi investasi tinggi -> mirip "investor")
        features_with_cluster = X.copy()
        features_with_cluster["cluster"] = cluster_labels
        cluster_profile = features_with_cluster.groupby("cluster").mean().round(3)
        print("\nKarakteristik rata-rata tiap cluster:")
        print(cluster_profile)
        
        cluster_profile.to_csv("cluster_profile.csv")
        mlflow.log_artifact("cluster_profile.csv")


if __name__ == "__main__":
    main()