import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque, Counter
import pygame
import os

mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_det = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def d(a, b): return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)
def skala(lm): return d(lm[152], lm[10]) + 1e-6
def px(pt, W, H): return (int(pt.x * W), int(pt.y * H))

def status_jari(lm, kiri=False):
    ujung, tengah = [8,12,16,20], [6,10,14,18]
    out = [1 if (lm[4].x > lm[3].x if kiri else lm[4].x < lm[3].x) else 0]
    for t, m in zip(ujung, tengah): out.append(1 if lm[t].y < lm[m].y else 0)
    return out

class Kalibrasi:
    N = 45
    def __init__(self):
        self.buf = {k: [] for k in ['ci','cd','cen','lap','llb','bi_y','bd_y','gap']}
        self.selesai = False
        self.batas = dict(ci=0.180, cd=0.180, cen_lo=0.185, lap=0.055, llb=0.145, bi_y_lo=0.30, bd_y_lo=0.30, gap_lo=0.10)

    def proses(self, lm):
        if self.selesai: return
        e = skala(lm)
        self.buf['ci'].append(d(lm[52], lm[159]) / e)
        self.buf['cd'].append(d(lm[282], lm[386]) / e)
        self.buf['cen'].append(d(lm[55], lm[285]) / e)
        self.buf['lap'].append(d(lm[13], lm[14]) / e)
        self.buf['llb'].append(d(lm[17], lm[152]) / e)
        self.buf['bi_y'].append(lm[55].y - lm[9].y)
        self.buf['bd_y'].append(lm[285].y - lm[9].y)
        self.buf['gap'].append(abs(lm[55].x - lm[285].x))
        if len(self.buf['ci']) >= self.N: self._hitung()

    def _hitung(self):
        m, s = lambda k: float(np.median(self.buf[k])), lambda k: float(np.std(self.buf[k]))
        mg_c, mg_b = lambda k: max(1.5 * s(k), 0.015), lambda k, mn: max(3 * s(k), mn)
        self.batas['ci'], self.batas['cd'], self.batas['cen_lo'] = m('ci') + mg_c('ci'), m('cd') + mg_c('cd'), m('cen') - mg_c('cen')
        self.batas['lap'], self.batas['llb'] = m('lap') + mg_b('lap', 0.032), m('llb') - mg_b('llb', 0.018)
        self.batas['bi_y_lo'], self.batas['bd_y_lo'], self.batas['gap_lo'] = m('bi_y') + mg_c('bi_y'), m('bd_y') + mg_c('bd_y'), m('gap') - mg_c('gap')
        self.selesai = True
    @property
    def progres(self): return min(len(self.buf['ci']) / self.N, 1.0)

def deteksi_peace(jari): return jari[1:] == [1, 1, 0, 0]
def deteksi_nunjuk(jari): return jari[1:] == [1, 0, 0, 0]
def deteksi_jari_tengah(jari): return jari[1:] == [0, 1, 0, 0]
def deteksi_jempol(jari): return jari == [1, 0, 0, 0, 0]
def deteksi_mikir(tangan, lm_wajah): return any(d(lm[8], lm_wajah[13]) < 0.10 for _, lm in tangan)
def deteksi_shock(lm_wajah, kal): return d(lm_wajah[13], lm_wajah[14]) / skala(lm_wajah) > kal.batas['lap'] * 1.2

HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
def gambar_tangan_minimal(frame, lm, W, H, jari):
    WARNA = (140, 200, 140)
    for a, b in HAND_CONNECTIONS: cv2.line(frame, px(lm[a], W, H), px(lm[b], W, H), WARNA, 1, cv2.LINE_AA)
    for i in range(21): cv2.circle(frame, px(lm[i], W, H), 2, WARNA, -1, cv2.LINE_AA)
    for i, ujung in enumerate([4, 8, 12, 16, 20]):
        if jari[i]: cv2.circle(frame, px(lm[ujung], W, H), 3, (80, 240, 80), -1, cv2.LINE_AA)

def hud(frame, img_sekarang, info_tangan, W, H):
    nama = img_sekarang if img_sekarang else "netral"
    ov = frame.copy()
    cv2.rectangle(ov, (8, 8), (min(W - 8, 14 + len(nama) * 14 + 20), 36), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, nama, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 80) if img_sekarang else (160, 160, 160), 2, cv2.LINE_AA)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pygame.mixer.init()
    try: pygame.mixer.music.load(os.path.join(BASE_DIR, "kicau_mania.mp3"))
    except: pass
    
    cat_video = cv2.VideoCapture(os.path.join(BASE_DIR, "cat_dance.mp4"))
    
    mode_kucing = False
    is_playing = False
    history_x = deque(maxlen=5) 
    
    # Timer pengganti toggle
    kucing_timer = 0           

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened(): return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5): cap.read()
    ret, frame0 = cap.read()
    if not ret: return

    frame0 = cv2.flip(frame0, 1)
    H, W = frame0.shape[:2]
    latar = np.full((H, W, 3), 30, dtype=np.uint8)

    cv2.namedWindow("Kamera Kamu", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Meme Terdeteksi", cv2.WINDOW_AUTOSIZE)

    kal = Kalibrasi()
    buf = deque(maxlen=10)
    img_sekarang = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr = face_mesh.process(rgb)
        hr = hands_det.process(rgb)

        det, lm_wajah, tangan, info_tangan = None, None, [], []

        if not kal.selesai:
            pct = kal.progres
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, "Tatap layar dengan wajah netral", (W // 2 - 200, H // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
            if fr.multi_face_landmarks: kal.proses(fr.multi_face_landmarks[0].landmark)
            cv2.imshow("Kamera Kamu", frame)
            cv2.imshow("Meme Terdeteksi", latar)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        if fr.multi_face_landmarks:
            lm_wajah = fr.multi_face_landmarks[0].landmark

        if hr.multi_hand_landmarks:
            for i, hl in enumerate(hr.multi_hand_landmarks):
                lm = hl.landmark
                kiri = hr.multi_handedness[i].classification[0].label == "Left"
                jari = status_jari(lm, kiri)
                gambar_tangan_minimal(frame, lm, W, H, jari) 
                tangan.append((jari, lm))
                info_tangan.append(("Kiri" if kiri else "Kanan", jari))

        # --- SENSOR GERAK OTOMATIS TUTUP ---
        if len(tangan) >= 1:
            jari_m, lm_m = tangan[0] 
            history_x.append(lm_m[8].x) 
            
            if len(history_x) == 5:
                jarak_gerak = history_x[-1] - history_x[0] 
                
                # SENSITIVITAS GERAK DIPERBAIKI: Turun drastis dari 0.1 menjadi 0.02
                # Asalkan tanganmu bergeser sedikit saja, dianggap masih bergerak
                if abs(jarak_gerak) > 0.02 and sum(jari_m) >= 4: 
                    kucing_timer = 25  # Timer diperpanjang sedikit agar tidak mati saat tangan ganti arah ayunan

        # Kicau Mania menyala selama timer berisi angka. Jika tangan diam total, timer habis = mati.
        if kucing_timer > 0:
            mode_kucing = True
            kucing_timer -= 1
        else:
            mode_kucing = False

        # Kicau Mania menyala selama timer berisi angka. Jika kamu diam, timer habis = mati.
        if kucing_timer > 0:
            mode_kucing = True
            kucing_timer -= 1
        else:
            mode_kucing = False

        # --- KONTROL MUSIK & JENDELA POP-UP ---
        if mode_kucing and not is_playing:
            try: pygame.mixer.music.play(-1)
            except: pass
            cat_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            is_playing = True
            
        elif not mode_kucing and is_playing:
            pygame.mixer.music.stop()
            is_playing = False
            try: cv2.destroyWindow("kicau-mania.mp4") 
            except: pass

        # --- TAMPILAN BERDASARKAN MODE ---
        if mode_kucing:
            ret_cat, cat_frame = cat_video.read()
            if not ret_cat:
                cat_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_cat, cat_frame = cat_video.read()
            
            if ret_cat:
                cv2.imshow("kicau-mania.mp4", cv2.resize(cat_frame, (W, H)))
            else:
                err = latar.copy()
                cv2.putText(err, "Video Kucing Gagal Diputar!", (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 220), 2)
                cv2.imshow("kicau-mania.mp4", err)
                
            hud(frame, "MODE KICAU MANIA AKTIF!", info_tangan, W, H)
            cv2.imshow("Kamera Kamu", frame)
            
        else:
            if lm_wajah: det = "download (6).jpg"
            if lm_wajah and deteksi_shock(lm_wajah, kal): det = "shock.jpg"
            elif lm_wajah and tangan and deteksi_mikir(tangan, lm_wajah): det = "Para figurinha.jpg"
            elif len(tangan) >= 1:
                jari_m, lm_m = tangan[0]
                if deteksi_jempol(jari_m): det = "jempol.jpg"
                elif deteksi_peace(jari_m): det = "peace monke.jpg"
                elif deteksi_jari_tengah(jari_m): det = "download (5).jpg"
                elif deteksi_nunjuk(jari_m): det = "download (4).jpg"

            buf.append(det)
            top, voting = Counter(buf).most_common(1)[0]
            if voting >= 6: img_sekarang = top

            hud(frame, img_sekarang, info_tangan, W, H)
            cv2.imshow("Kamera Kamu", frame)

            if img_sekarang:
                meme = cv2.imread(img_sekarang)
                if meme is not None and meme.size > 0: cv2.imshow("Meme Terdeteksi", cv2.resize(meme, (W, H)))
            else: cv2.imshow("Meme Terdeteksi", latar)

        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cat_video.release()
    pygame.mixer.music.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()