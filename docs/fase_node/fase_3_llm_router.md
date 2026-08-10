# Dokumentasi FinAdvisor AI - Fase 3: LLM Router Service

Fokus fase ini adalah membangun **LLM Router Service** sebagai "Pengatur Lalu Lintas" (Router) yang memastikan aplikasi tidak pernah gagal merespons meskipun server AI utama sedang mati atau *down*.

## 1. Konsep "Fallback Chain" (Antrean AI)
Sistem dirancang untuk memiliki pertahanan berlapis (High Availability). Jika AI urutan pertama mati, ia akan otomatis memanggil AI urutan kedua, dan seterusnya.
Urutan prioritas AI yang diatur di `config.py`:
1. **Google Gemini** (`gemini-3.6-flash`)
2. **Groq** (`llama-3.3-70b-versatile`)
3. **Deepseek** (`deepseek-chat`)
4. **Ollama Lokal** (`llama3.2`) — sebagai benteng pertahanan terakhir tanpa internet.

## 2. Satu Pustaka Untuk Semua (Universal Client)
- **`client.py`**: Berhasil memanfaatkan kompatibilitas *OpenAI API* yang kini diadaptasi oleh mayoritas penyedia AI. Dengan begitu, kita hanya perlu menginstal dan menggunakan satu pustaka saja (`openai` SDK) untuk berbicara dengan berbagai merek AI (Gemini, Groq, Deepseek, maupun Ollama).

## 3. Otak Router & Penanganan Eror
- **`router.py`**: Mengatur proses pemanggilan AI. Ia mengecek ketersediaan API Key sebelum menelepon.
- Dilengkapi dengan sistem *Timeout* pintar: AI berbasis internet diberi waktu 15 detik, sedangkan Ollama Lokal (yang memproses di komputer sendiri) diberi kelonggaran 180 detik.
- Jika ada *error* atau batas waktu habis, sistem mencatat alasan kegagalan dan langsung melompat ke AI berikutnya.

## 4. API Endpoint Internal
- **`generate.py`**: Menyediakan pintu masuk berformat API (`POST /internal/v1/generate`) agar layanan lain di dalam proyek (seperti Agent Service di Fase 4) bisa menggunakan kekuatan AI secara terpusat dan konsisten.

## 5. Pembaruan Infrastruktur
- Menghapus ketergantungan pada `OPENAI_API_KEY` berbayar.
- Menambahkan *Environment Variables* untuk kunci rahasia Gemini, Groq, dan Deepseek.
- Menerapkan aturan `depends_on: ollama` agar LLM Router Service selalu menunggu hingga AI cadangan lokalnya menyala dan siap menerima perintah.
