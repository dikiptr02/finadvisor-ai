# FinAdvisor AI — Repo Structure & Contracts

**Status:** Draft Fase 0
**Repo mode:** Monorepo (rencana migrasi ke polyrepo setelah stabil)

## 1. Struktur Folder (Monorepo)

```
finadvisor-ai/
├── docs/                          # dokumen referensi (fase 0 dst)
│   ├── 01-project-charter.md
│   ├── 02-architecture.md
│   ├── 03-repo-structure-and-contracts.md
│   └── adr/                       # Architecture Decision Records
│       └── 0001-tech-stack.md
│
├── services/
│   ├── api-gateway/
│   ├── rag-service/
│   │   ├── app/
│   │   │   ├── ingestion/         # semantic chunking
│   │   │   ├── indexing/          # BM25 + dense index
│   │   │   ├── retrieval/         # hybrid search + reranker
│   │   │   └── eval/              # RAGAS eval scripts
│   │   └── tests/
│   ├── agent-service/
│   │   ├── app/
│   │   │   ├── state_machine/
│   │   │   ├── verifier/
│   │   │   └── hitl/
│   │   └── tests/
│   ├── llm-router-service/
│   │   ├── app/
│   │   │   ├── providers/         # adapter per provider
│   │   │   ├── router/            # routing + fallback logic
│   │   │   └── ensemble_verifier/
│   │   └── tests/
│   ├── ml-service/
│   │   ├── app/
│   │   │   ├── classifier/
│   │   │   ├── anomaly/           # isolation_forest.py, autoencoder.py
│   │   │   └── clustering/
│   │   └── tests/
│   └── forecast-service/
│       ├── app/
│       │   ├── classical/         # arima.py, prophet_model.py
│       │   └── deep_learning/     # lstm.py, tft.py
│       └── tests/
│
├── orchestration/
│   └── airflow/
│       └── dags/
│           ├── daily_ingest_dag.py
│           ├── weekly_eval_retrain_dag.py
│           └── monthly_report_dag.py
│
├── mlops/
│   ├── mlflow/                    # config tracking server
│   └── ci-gate/                   # script gate metric untuk CI
│
├── observability/
│   └── dashboard/                 # cost/latency dashboard
│
├── shared/                        # kode/util yang dipakai lintas service
│   ├── schemas/                   # pydantic models bersama (kontrak data)
│   └── clients/                   # http client antar service
│
├── infra/
│   └── docker-compose.yml
│
└── .github/workflows/             # CI/CD
```

## 2. Prinsip Struktur

- Tiap service di `services/` **independen secara logika** (punya `app/`, `tests/`,
  `Dockerfile`, `requirements.txt`/`pyproject.toml` sendiri) — supaya migrasi ke
  polyrepo nanti tinggal `git subtree split` atau `git filter-repo` per folder.
- `shared/` **dijaga seminimal mungkin** — hanya schema kontrak (pydantic) dan HTTP
  client tipis. Hindari coupling logic lewat shared code, karena ini akan menyulitkan
  migrasi ke polyrepo.
- Setiap service punya `README.md` sendiri berisi: cara jalan lokal, endpoint yang
  disediakan, dependency ke service lain.

## 3. Kontrak API Antar Service (Draft — akan detail per fase)

Format: OpenAPI 3.0 per service, disimpan di `services/<nama>/openapi.yaml`
(dibuat saat fase service tsb dikerjakan). Konvensi umum yang disepakati dari awal:

- Semua endpoint internal pakai prefix `/internal/v1/...`.
- Semua endpoint yang diekspos ke API Gateway (dan dikonsumsi dashboard) pakai
  prefix `/api/v1/...`.
- Response envelope konsisten:
  ```json
  {
    "success": true,
    "data": { ... },
    "trace_id": "uuid",
    "error": null
  }
  ```
- Setiap request lintas service wajib membawa header `X-Trace-Id` (untuk observability/
  MLflow Tracing correlation).

### Contoh kontrak awal: Agent Service → RAG Service

```
POST /internal/v1/retrieve
Request:
{
  "user_id": "string",
  "query": "string",
  "top_k": 5,
  "filters": { "date_from": "string", "date_to": "string" }
}

Response:
{
  "success": true,
  "data": {
    "chunks": [
      { "text": "string", "source": "string", "score": 0.0, "metadata": {} }
    ]
  },
  "trace_id": "uuid",
  "error": null
}
```

## 4. Kontrak Data dengan Dashboard Existing (perlu konfirmasi kamu saat Fase 1)

Placeholder — akan diisi begitu kamu share skema data dashboard existing (transaksi,
user profile, dsb). Yang perlu dikonfirmasi nanti:
- Format data transaksi (field apa saja: amount, category, timestamp, merchant, dll).
- Mekanisme auth yang dipakai dashboard (JWT? session? API key?) supaya API Gateway
  bisa passthrough dengan benar.
- Apakah FinAdvisor AI akses data lewat DB langsung (read replica) atau lewat API
  dashboard existing. **Rekomendasi**: lewat API/event, bukan akses DB langsung, supaya
  loose-coupling terjaga.

## 5. Penamaan & Konvensi

- Python: `snake_case` untuk file/function, `PascalCase` untuk class, pydantic model
  untuk semua request/response schema (taruh di `shared/schemas` jika dipakai >1 service).
- Commit message: `[fase-X][service-name] deskripsi singkat` — memudahkan tracking
  progres per fase di git log.
- Branch: `fase-0/docs`, `fase-1/infra-skeleton`, dst.
