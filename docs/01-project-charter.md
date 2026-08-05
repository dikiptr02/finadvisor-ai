# FinAdvisor AI — Project Charter

**Status:** Draft Fase 0
**Owner:** Putra
**Last updated:** 2026-08-05

## 1. Vision

FinAdvisor AI adalah **Personal Finance Advisor Agent** berbasis microservices yang akan
diintegrasikan ke web dashboard finansial yang sudah ada. Sistem ini menggabungkan RAG,
agentic workflow, multi-LLM routing, ML klasik, dan forecasting untuk memberikan insight
dan rekomendasi finansial yang grounded, auditable, dan aman dari hallucination — terutama
untuk aksi-aksi berisiko (misal rekomendasi transaksi besar, alokasi ulang portofolio).

## 2. Problem Statement

Dashboard finansial existing bersifat pasif (menampilkan data). Tidak ada layer yang:
- Menjawab pertanyaan finansial user dengan konteks data mereka sendiri (RAG).
- Melakukan reasoning multi-step dan mengambil aksi terverifikasi (agentic).
- Mendeteksi anomali transaksi / segmentasi user secara otomatis (ML klasik).
- Memberi proyeksi keuangan masa depan (forecasting).
- Semua di atas dengan observability, cost control, dan verifikasi ketat sebelum output
  sampai ke user.

## 3. Scope (In / Out)

### In-scope (v1 enterprise-grade)
| Layer | Cakupan |
|---|---|
| RAG | Hybrid search (BM25 + dense) + reranker + semantic chunking + eval RAGAS |
| Agentic | Full state machine, human-in-the-loop, verifier layer untuk grounding |
| Multi-LLM | Router + fallback multi-provider + ensemble verifier untuk aksi berisiko |
| ML Klasik | Fine-tuned classifier, Isolation Forest vs Autoencoder, clustering segmentasi user |
| Forecasting | Studi komparasi ARIMA/Prophet vs LSTM/TFT |
| MLOps | MLflow tracking + registry + CI/CD gate |
| Orkestrasi | Airflow — 3 DAG minimal (daily ingest, weekly eval/retrain, monthly report) |
| Observability | MLflow Tracing + dashboard cost/latency |

### Out-of-scope (v1)
- Real money movement / eksekusi transaksi otomatis (hanya rekomendasi, aksi final tetap
  manual oleh user — human-in-the-loop wajib untuk apa pun yang "berisiko").
- Multi-tenant SaaS billing (asumsi single-tenant, terintegrasi ke 1 dashboard existing).
- Mobile app native (integrasi lewat API yang dikonsumsi dashboard web existing).

## 4. Non-Functional Requirements (Enterprise Bar)

- **Grounding-first**: setiap output finansial harus melalui verifier layer sebelum
  ditampilkan ke user. Tidak ada rekomendasi tanpa sumber/evidence.
- **Auditability**: setiap keputusan agent (state transition, tool call, verifier verdict)
  harus tercatat (trace) dan bisa direplay.
- **Fail-safe LLM**: kegagalan 1 provider tidak boleh menghentikan sistem (fallback wajib).
- **Reproducibility**: setiap model (classifier, anomaly detector, forecaster) harus
  ter-track versi, dataset, metric via MLflow registry.
- **Cost-aware**: setiap LLM call dan pipeline run harus terukur cost & latency-nya.
- **Incremental delivery**: tiap fase harus deployable & testable sendiri, tidak menunggu
  seluruh sistem selesai.

## 5. Success Criteria per Fase

Didefinisikan detail di masing-masing fase saat dikerjakan, tapi kriteria umum:
- Ada test (unit minimal, eval untuk komponen ML/RAG).
- Ada dokumentasi singkat (README per service).
- Bisa dijalankan lokal via docker-compose.
- Ter-push ke git dengan commit history yang jelas per fase/sub-fase.

## 6. Working Agreement (Kolaborasi Kita)

- Saya (AI Engineer partner) bertugas: desain arsitektur, tulis code, jelaskan logic,
  review, problem-solving.
- Kamu bertugas: copy-paste code/text ke file di repo, jalankan & push ke git,
  ambil keputusan produk/prioritas.
- Dokumen (`docs/`) saya buatkan langsung sebagai file.
- Kerja dilakukan bertahap per fase, tiap fase dikonfirmasi dulu sebelum lanjut.
- Style kerja: step-by-step, confirm-each-stage (tidak lompat fase tanpa konfirmasi).

## 7. Repo Strategy

- **Sekarang**: Monorepo — semua service dalam satu repo, dipisah per folder
  (`services/<nama-service>`), memudahkan development lintas service di fase awal.
- **Nanti** (setelah stabil): migrasi ke polyrepo per service, saat masing-masing service
  sudah punya lifecycle deploy independen.
