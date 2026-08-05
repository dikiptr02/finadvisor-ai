# FinAdvisor AI — Architecture Reference

**Status:** Draft Fase 0

## 1. High-Level Diagram (conceptual)

```
                         ┌─────────────────────────┐
                         │   Web Dashboard (existing)│
                         └────────────┬─────────────┘
                                      │ REST/HTTPS (JWT)
                         ┌────────────▼─────────────┐
                         │       API Gateway          │
                         │ (auth passthrough, routing,│
                         │  rate limit, request log)  │
                         └────────────┬─────────────┘
             ┌───────────────┬────────┴────────┬───────────────┐
             │               │                 │               │
      ┌──────▼─────┐  ┌──────▼──────┐  ┌───────▼──────┐  ┌─────▼──────┐
      │ RAG Service │  │Agent Service │  │  ML Service   │  │ Forecast   │
      │(retrieval,  │  │(state machine│  │(classifier,   │  │  Service   │
      │ rerank,     │  │ + HITL +     │  │ anomaly detect│  │(ARIMA/     │
      │ chunking)   │  │ verifier)    │  │ clustering)   │  │Prophet/    │
      └──────┬─────┘  └──────┬──────┘  └───────┬──────┘  │LSTM/TFT)   │
             │               │                 │          └─────┬──────┘
             │        ┌──────▼──────┐          │                │
             │        │ LLM Router  │          │                │
             │        │ Service     │◄─────────┴────────────────┘
             │        │(multi-      │
             │        │provider +   │
             │        │fallback +   │
             │        │ensemble     │
             │        │verifier)    │
             │        └──────┬──────┘
             │               │
      ┌──────▼───────────────▼──────┐
      │     Vector Store (hybrid)    │
      │  (dense index + BM25 index)  │
      └──────────────────────────────┘

      ┌───────────────────────────────────────────────────────────┐
      │                    Cross-cutting layers                    │
      │  MLflow (tracking/registry/tracing) │ Airflow (DAGs)        │
      │  Observability dashboard (cost/latency) │ CI/CD gate        │
      └───────────────────────────────────────────────────────────┘
```

## 2. Service Breakdown

### 2.1 API Gateway
- Entry point tunggal untuk dashboard existing.
- Tanggung jawab: routing ke service internal, auth passthrough (terima token dari
  dashboard, tidak generate token sendiri di v1), rate limiting, request logging (trace id).

### 2.2 RAG Service
- **Ingestion**: semantic chunking (bukan fixed-size) atas dokumen finansial user
  (statement, transaksi, artikel edukasi finansial internal).
- **Indexing**: dua index — sparse (BM25) dan dense (embedding).
- **Retrieval**: hybrid search (gabung BM25 + dense, reciprocal rank fusion atau
  weighted sum) → reranker (cross-encoder) → top-k final context.
- **Eval**: RAGAS metrics (faithfulness, answer relevancy, context precision, context
  recall) dijalankan sebagai bagian dari CI dan sebagai monthly report via Airflow DAG.

### 2.3 Agent Service
- **State machine** (finite/graph-based) mengelola alur: intent detection → retrieval →
  reasoning → (jika aksi berisiko) human-in-the-loop checkpoint → verifier → output.
- **Human-in-the-loop (HITL)**: state khusus "pending_approval" untuk aksi yang
  dikategorikan berisiko (threshold ditentukan per use case, misal rekomendasi
  realokasi > X% portofolio).
- **Verifier layer**: cek grounding — apakah klaim di output agent didukung oleh
  evidence dari RAG/ML/Forecast service sebelum diteruskan ke user. Verifier ini
  terpisah dari LLM yang generate jawaban (supaya tidak self-graded).

### 2.4 LLM Router Service
- Routing request ke provider berdasarkan kriteria (cost, latency, availability,
  task type).
- **Fallback chain**: jika provider utama gagal/timeout, otomatis coba provider
  berikutnya sesuai chain yang dikonfigurasi.
- **Ensemble verifier**: untuk aksi berisiko, jawaban di-generate dari >1 model/provider
  dan dibandingkan (voting atau cross-check) sebelum dianggap valid.

### 2.5 ML Service (Klasik)
- **Classifier**: fine-tuned (misal fine-tune embedding + classifier head, atau
  fine-tune small transformer) untuk kategorisasi transaksi — bukan TF-IDF baseline.
- **Anomaly detection**: dua pendekatan dijalankan paralel dan dibandingkan —
  Isolation Forest (tree-based) vs Autoencoder (reconstruction-error based).
  Hasil studi banding didokumentasikan (precision/recall pada anomali berlabel,
  atau proxy metric bila unlabeled).
- **Clustering**: segmentasi user (misal berdasarkan pola spending) — KMeans/HDBSCAN,
  dipakai untuk personalisasi rekomendasi.

### 2.6 Forecasting Service
- Dua kelas model dijalankan dan dibandingkan secara sistematis dengan backtesting
  yang konsisten (sama train/test split, sama horizon):
  - Classical: ARIMA, Prophet.
  - Deep learning: LSTM, TFT (Temporal Fusion Transformer).
- Output studi komparasi (metric: MAPE/RMSE per horizon) menjadi bagian dari monthly
  report DAG.

### 2.7 MLOps Layer
- **MLflow tracking**: setiap training run (classifier, anomaly, clustering,
  forecasting) log parameter, metric, artifact.
- **MLflow registry**: versi model, stage (staging/production), promotion terkontrol.
- **CI/CD gate**: pipeline CI menolak deploy model baru jika metric di bawah
  threshold dibanding model production saat ini.

### 2.8 Orkestrasi (Airflow)
- **DAG 1 — daily ingest**: tarik data transaksi/dokumen baru → chunking → update index
  RAG → jalankan anomaly detection harian.
- **DAG 2 — weekly eval/retrain**: evaluasi RAGAS, evaluasi classifier/anomaly/forecast,
  retrain jika drift terdeteksi, registrasi model baru ke MLflow.
- **DAG 3 — monthly report**: agregasi metric (RAG eval, model performance, cost/latency)
  → generate laporan.

### 2.9 Observability
- **MLflow Tracing**: trace end-to-end request — dari masuk API Gateway, retrieval,
  agent state transitions, LLM calls (termasuk provider mana yang dipakai/fallback),
  sampai output final.
- **Dashboard cost/latency**: agregasi cost per LLM call (token in/out x harga
  provider) dan latency per service, per hari/minggu.

## 3. Data Flow Ringkas (Contoh: user tanya "apakah pengeluaran saya bulan ini wajar?")

1. Dashboard → API Gateway → Agent Service.
2. Agent Service masuk state `retrieve_context` → panggil RAG Service (hybrid search +
   rerank atas data transaksi user) dan ML Service (cek anomaly/klasifikasi kategori).
3. Agent Service masuk state `reason` → panggil LLM Router → LLM Router pilih provider →
   generate draft jawaban.
4. Karena ini bukan aksi berisiko (hanya informational), verifier layer cek grounding
   draft jawaban terhadap evidence dari langkah 2. Jika lolos → lanjut. Jika gagal →
   agent retry/retrieve ulang.
5. Output dikirim balik ke dashboard, sekaligus trace tercatat ke MLflow Tracing.

Untuk kasus aksi berisiko (misal "realokasikan investasi saya"), setelah state `reason`,
agent masuk state `pending_approval` (HITL) — user harus konfirmasi eksplisit di
dashboard sebelum lanjut ke `verify` (dengan ensemble verifier, bukan single verifier)
dan `execute` (yang di v1 hanya menghasilkan rekomendasi terstruktur, bukan eksekusi
transaksi nyata).

## 4. Integrasi ke Dashboard Existing

- Kontrak API didefinisikan di `docs/03-repo-structure-and-contracts.md`.
- Prinsip: dashboard existing tetap system of record untuk data user (transaksi,
  saldo, dsb). FinAdvisor AI mengonsumsi data itu (read) dan mengembalikan insight/
  rekomendasi — tidak menjadi source of truth data finansial baru.
