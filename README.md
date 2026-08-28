# 🐵 Detektor Meme Berbasis Gestur & Ekspresi AI

Proyek *computer vision* *real-time* yang dibangun menggunakan Python, OpenCV, dan MediaPipe untuk mendeteksi gestur tangan serta ekspresi wajah guna mengganti gambar meme di layar secara dinamis.

---

## Fitur

* **Deteksi Real-Time:** Didukung oleh MediaPipe Hands dan Face Mesh untuk pelacakan titik wajah dan tangan yang cepat serta akurat.
* **Pengenalan Gestur Tangan:** 
  * ✌️ **Tanda Peace** $\rightarrow$ `peace monke.jpg`
  * ☝️ **Menunjuk** $\rightarrow$ `download (4).jpg`
  * 🖕 **Jari Tengah** $\rightarrow$ `download (5).jpg`
  * 👍 **Jempol (Thumbs Up)** $\rightarrow$ `jempol.jpg`
  * 🖐️ **Telapak Terbuka (Dab)** $\rightarrow$ `dab.jpg`
* **Ekspresi Wajah & Interaksi:**
  * 😮 **Kaget / Mulut Terbuka** $\rightarrow$ `shock.jpg`
  * 🤔 **Pose Berpikir** (Menyentuh dagu/mulut) $\rightarrow$ `Para figurinha.jpg`
  * 😐 **Keadaan Default** $\rightarrow$ `download (6).jpg` (Aktif selama wajahmu terdeteksi kamera)
* **Kalibrasi Otomatis:** Menyesuaikan dengan wajah netralmu secara dinamis saat pertama kali dinyalakan.
* **Tampilan Dua Jendela:** Memisahkan jendela kamera langsung dengan jendela proyeksi meme aktif.

---

## Kebutuhan Sistem & Teknologi

* **Python** (Direkomendasikan: versi `3.11` atau `3.12`)
* **OpenCV** (`cv2`)
* **MediaPipe** (`mediapipe==0.10.14`)
* **NumPy** (`numpy`)

---

## Instalasi & Cara Menjalankan

1. **Clone repository ini:**
   ```bash
   git clone [https://github.com/USERNAME/nama-repo-kamu.git](https://github.com/USERNAME/nama-repo-kamu.git)
   cd nama-repo-kamu
