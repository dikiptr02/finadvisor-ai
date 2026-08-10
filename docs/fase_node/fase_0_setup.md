# Dokumentasi FinAdvisor AI - Fase 0: Setup & Architecture

Pada fase ini, kita meletakkan kerangka utama (fondasi) proyek agar fleksibel, dapat dikembangkan, dan mudah dirawat.

## 1. Desain Microservices
Proyek dibagi menjadi 6 *service* independen:
- **API Gateway**: Pintu masuk utama bagi pengguna/aplikasi depan (Frontend).
- **RAG Service**: Mesin pencari dan pengelola dokumen.
- **Agent Service**: Otak utama AI yang merangkai jawaban.
- **LLM Router Service**: Pengatur lalu lintas *prompt* ke berbagai AI (OpenAI, Gemini, Ollama).
- **ML Service**: Pengelola model Machine Learning selain LLM.
- **Forecast Service**: Mesin peramal data keuangan.

## 2. Dockerisasi
- Semua *service* dibungkus dalam *container* menggunakan `Dockerfile`.
- Diatur oleh satu konduktor utama yaitu `docker-compose.yml`.

## 3. Integrasi Komponen Eksternal
- **Qdrant**: Terpasang sebagai Vector Database (tempat menyimpan artikel yang diubah jadi angka).
- **Ollama**: Menjalankan AI lokal sepenuhnya tanpa internet.
- **Optimalisasi Performa**: Konfigurasi *GPU Passthrough* di Docker agar Ollama berjalan 10-50x lebih cepat, serta pengaturan `OLLAMA_KEEP_ALIVE=-1` agar model tidak dihapus dari memori RAM.
