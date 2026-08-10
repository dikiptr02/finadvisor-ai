# Fase 4: Implementasi Agent Service (LangGraph)

Dokumen ini menjelaskan alur kerja dan arsitektur dari **Agent Service**, yang bertindak sebagai "otak utama" atau orkestrator dari keseluruhan sistem FinAdvisor AI. Service ini dibangun menggunakan pustaka **LangGraph** untuk menciptakan *State Machine* (Mesin Status) yang stabil, tangguh, dan dapat diandalkan.

## 1. Arsitektur State Machine (Buku Catatan Sementara)

Di dalam `app/state_machine/state.py`, kita mendefinisikan `AgentState` yang bertindak sebagai buku catatan estafet. Buku ini dibawa oleh AI dari satu pos ke pos lainnya selama proses berpikir:

```python
class AgentState(TypedDict):
    user_id: str
    query: str
    intent: Literal["informational", "actionable"] | None
    retrieved_context: list[dict] | None
    draft_answer: str | None
    approval_status: Literal["approved", "rejected"] | None
    verifier_verdict: Literal["grounded", "not_grounded"] | None
    final_answer: str | None
```

## 2. Alur Kerja (Nodes & Edges)

Proses berpikir AI dibagi menjadi beberapa pos (Nodes) yang dijalankan secara berurutan atau bercabang (Edges) sesuai kondisi:

1. **`intent_detection_node`**: Pos Satpam. Menyortir apakah pertanyaan pengguna bersifat edukasi umum (`informational`) atau meminta saran tindakan finansial berisiko (`actionable`).
2. **`retrieve_context_node`**: Pos Pustakawan. Jika pertanyaan valid, Agent akan memanggil `rag-service` (Qdrant) untuk mencari artikel atau dokumen referensi yang relevan.
3. **`reason_node`**: Pos Pemikir. Agent mengirimkan dokumen yang ditemukan beserta pertanyaan pengguna ke `llm-router-service`. AI dipaksa membuat draf jawaban HANYA berdasarkan dokumen tersebut (RAG Prompt).
4. **Petugas Wesel (Router)**:
   - Jika niat = `actionable`, kereta diarahkan ke `pending_approval_node`.
   - Jika niat = `informational`, kereta diarahkan ke `verifier_node`.
5. **`pending_approval_node`**: Pos Human-in-the-Loop (HITL). Proses AI dibekukan sementara (`interrupt`). Sistem meminta izin manusia sebelum menyetujui saran finansial berisiko tinggi tersebut.
6. **`verifier_node`**: Pos Juri Pengecek Fakta. AI independen mengecek apakah draf jawaban benar-benar didukung oleh data (Grounded) atau ternyata AI berhalusinasi (Not Grounded).
7. **`respond_node`**: Pos Vonis Akhir. Mengirimkan jawaban utuh jika disetujui dan berdasar fakta. Jika ditolak manusia atau ketahuan berhalusinasi, pos ini akan merespons dengan pesan pembatalan/maaf demi keamanan pengguna.

## 3. Human-in-the-Loop (HITL)

LangGraph 0.2+ menangani proses interupsi dengan sangat elegan:
- Saat menemui fungsi `interrupt()`, LangGraph akan berhenti memproses node berikutnya dan mengembalikan status terakhir (`state`) kepada pemanggil.
- Kunci `final_answer` tidak akan terbentuk, sehingga API akan membalas pengguna dengan status `"pending_approval"`.
- Proses ini bisa dijeda dalam waktu yang tidak terbatas sampai ada perintah `Command(resume=...)` yang dikirim dari luar untuk melanjutkan proses.

## 4. API Endpoints

Aplikasi ini mengekspos dua endpoint FastAPI utama di `app/routers/agent.py`:

- **`POST /internal/v1/agent/start`**: Endpoint untuk memulai proses. Menggunakan `agent_graph.invoke()`. Jika interupsi terjadi (karena *actionable*), endpoint akan merespons status `pending_approval` beserta draf jawaban untuk direview.
- **`POST /internal/v1/agent/resume`**: Endpoint untuk membangunkan AI yang tertidur. Menerima persetujuan pengguna (`approved` atau `rejected`) dan melanjutkan grafik hingga mencapai `respond_node`.

## 5. Menjalankan Service

Karena dependensi `langgraph` dan `httpx`, pastikan melakukan proses build ulang:
```bash
docker compose up -d --build agent-service
```
Memori percakapan sementara menggunakan `MemorySaver` (in-memory). Pada tahap produksi di fase berikutnya, ini akan diganti dengan Database (misal: Postgres) agar riwayat tidak hilang saat server di-restart.
