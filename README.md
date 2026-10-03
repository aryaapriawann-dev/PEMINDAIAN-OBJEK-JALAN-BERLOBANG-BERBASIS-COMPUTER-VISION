# PEMINDAIAN-OBJEK-JALAN-BERLOBANG-BERBASIS-COMPUTER-VISION

🛡️ **RoadGuard AI** — Sistem Pemindaian dan Deteksi Objek Jalan Berlubang Berbasis Computer Vision & Deep Learning (YOLOv8).

## 🚀 Fitur Utama
- **Real-Time Pothole Detection:** Deteksi jalan berlubang secara akurat dan real-time menggunakan YOLOv8.
- **High-Tech Surveillance HUD:** Antarmuka Heads-Up Display profesional bergaya sistem pemantauan jalan canggih.
- **Anti-Gepeng & Auto-Fill:** Rasio aspek kamera terjaga dan otomatis menyesuaikan jendela (tanpa ruang kosong abu-abu).
- **Mode Kamera Fleksibel:** Mendukung mode cermin / posisi tubuh (untuk pengujian) dan mode dashcam jalan raya.
- **Temporal Memory & Stabilizer:** Kotak deteksi stabil dan menetap (anti kedap-kedip) dengan konfirmasi bertahap.
- **Audio Alarm Alert:** Peringatan audio otomatis saat bahaya jalan berlubang terdeteksi.

## 🛠️ Instalasi & Persiapan
```bash
# Clone repositori
git clone https://github.com/aryaapriawann-dev/PEMINDAIAN-OBJEK-JALAN-BERLOBANG-BERBASIS-COMPUTER-VISION.git
cd PEMINDAIAN-OBJEK-JALAN-BERLOBANG-BERBASIS-COMPUTER-VISION

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Menjalankan Aplikasi
```bash
# Mode Ringan Pure OpenCV (Rekomendasi)
python app/main.py

# Mode Dashboard GUI (Tkinter)
python app/main.py --gui
```

### ⌨️ Kontrol Keyboard
| Tombol | Fungsi |
|---|---|
| `[Q]` / `[ESC]` | Keluar dari aplikasi |
| `[+]` / `[-]` | Atur tingkat kepercayaan / sensitivitas deteksi |
| `[F]` | Toggle posisi kamera (Sesuai Tubuh / Dashcam Jalan) |
| `[W]` | Toggle Fullscreen (Layar Penuh) |
| `[T]` | Atur memori durasi sensor (kotak menetap) |
| `[M]` | Mute / Nyalakan alarm audio |
| `[S]` | Simpan tangkapan layar (screenshot) |
