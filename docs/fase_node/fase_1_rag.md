# Dokumentasi FinAdvisor AI - Fase 1: RAG Implementation

Fokus fase ini adalah membangun **RAG Service** agar AI memiliki kemampuan "membaca" dan "mencari referensi".

## 1. Penyiapan Data (Dummy Data)
- Membuat `dummy_articles.py` berisi artikel edukasi keuangan (menabung, dana darurat, utang produktif) sebagai bahan latihan awal.

## 2. Proses Pemasukan Data (Data Ingestion)
- **Semantic Chunker** (`chunker.py`): Memotong artikel yang panjang menjadi paragraf-paragraf kecil agar lebih mudah dipahami oleh mesin.
- **Indexer** (`indexer.py`): Bertugas menerjemahkan potongan teks tadi menjadi vektor menggunakan model *embedding* dan mengarsipkannya ke database Qdrant.
- Disediakan *script* otomatis (`ingest_dummy.py`) untuk memproses semua artikel dalam satu kali klik.

## 3. Proses Pencarian Data (Data Retrieval)
- **Retriever** (`retriever.py`): Bertugas sebagai "Pustakawan". Menggunakan teknik *Hybrid Search* (menggabungkan pencarian kata kunci biasa dan pencarian makna secara vektor) untuk mencari potongan teks yang paling cocok dengan pertanyaan.
- **FastAPI Endpoint** (`retrieve.py` & `main.py`): Menyediakan jalur API agar layanan lain bisa melakukan pencarian ke RAG Service.
