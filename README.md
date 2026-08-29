# 🐵 Detektor Meme Berbasis Gestur & Ekspresi AI + Kicau Mania Mode

Proyek *computer vision* *real-time* yang dibangun menggunakan Python, OpenCV, MediaPipe, dan Pygame untuk mendeteksi gestur tangan serta ekspresi wajah guna mengganti gambar meme di layar secara dinamis, lengkap dengan fitur interaktif **Kicau Mania** (*cat dance* & musik) berbasis sensor gerak.

---

## Fitur Utama

* **Deteksi Real-Time:** Didukung oleh MediaPipe Hands dan Face Mesh untuk pelacakan titik wajah dan tangan yang cepat serta akurat.
* **Pengenalan Gestur & Ekspresi:** 
  * ✌️ **Tanda Peace** $\rightarrow$ `peace monke.jpg`
  * ☝️ **Menunjuk** $\rightarrow$ `download (4).jpg`
  * 🖕 **Jari Tengah** $\rightarrow$ `download (5).jpg`
  * 👍 **Jempol (Thumbs Up)** $\rightarrow$ `jempol.jpg`
  * 😮 **Kaget / Mulut Terbuka** $\rightarrow$ `shock.jpg`
  * 🤔 **Pose Berpikir** (Menyentuh dagu/mulut) $\rightarrow$ `Para figurinha.jpg`
  * 😐 **Keadaan Default** $\rightarrow$ `download (6).jpg` (Aktif selama wajahmu terdeteksi kamera)
* **🎵 Fitur Kicau Mania (*Motion Sensor Mode*):**
  * Gerakkan tangan secara aktif dengan telapak terbuka untuk memicu pemutaran video `cat_dance.mp4` dan audio `kicau_mania.mp3`.
  * Video dan musik akan otomatis tertutup jika tangan didiamkan atau diturunkan.
* **Kalibrasi Otomatis:** Menyesuaikan dengan wajah netralmu secara dinamis saat pertama kali dinyalakan.
* **Tampilan Dua Jendela:** Memisahkan jendela kamera langsung dengan jendela proyeksi meme aktif.

---

## Kebutuhan Sistem & Teknologi

* **Python** (Direkomendasikan: versi `3.11` atau `3.12`)
* **OpenCV** (`cv2`)
* **MediaPipe** (`mediapipe==0.10.14`)
* **NumPy** (`numpy`)
* **Pygame** (`pygame`)

---

## Instalasi & Cara Menjalankan

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/USERNAME/nama-repo-kamu.git](https://github.com/USERNAME/nama-repo-kamu.git)
   cd nama-repo-kamu
