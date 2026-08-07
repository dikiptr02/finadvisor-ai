# Dokumentasi FinAdvisor AI - Fase 2: Evaluation

Fase ini memastikan bahwa mesin pencari yang dibangun di Fase 1 benar-benar akurat menggunakan pustaka **RAGAS** sebagai juri penguji otomatis.

## 1. Pembuatan Kunci Jawaban (Golden Dataset)
- Membuat `golden_dataset.py` yang berisi pasangan **Soal** dan **Kunci Jawaban Mutlak** (*Ground Truth*) sebagai pedoman penilaian.

## 2. Eksekusi Penilaian Otomatis
- **Evaluator Script** (`run_ragas_eval.py`): Naskah juri penguji.
- **Kustomisasi LLM Lokal**: Menghubungkan juri ke Ollama menggunakan model `llama3.2` (sebagai penilai) dan `nomic-embed-text` (sebagai pengukur kemiripan).
- **Penanganan Concurrency**: Menambahkan `RunConfig(max_workers=1, timeout=300)` untuk menyuruh RAGAS menilai satu per satu guna mencegah *Timeout Error*.

## 3. Hasil Evaluasi Terakhir
- **Context Precision (1.000)**: Dokumen yang ditarik oleh sistem **100% relevan**.
- **Context Recall (0.888)**: Mesin pencari berhasil menemukan **~89%** informasi yang wajib ditemukan.
- **Answer Relevancy (~0.50)**: Sistem saat ini masih memberikan jawaban berupa kutipan teks mentah (belum dimodifikasi oleh AI).
- **Faithfulness (nan)**: Keterbatasan model kecil (3B parameters) dalam menghasilkan format JSON yang ketat.
