import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque, Counter

mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_det = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def d(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)

def skala(lm):
    return d(lm[152], lm[10]) + 1e-6

def px(pt, W, H):
    return (int(pt.x * W), int(pt.y * H))

def status_jari(lm, kiri=False):
    ujung = [8,12,16,20]
    tengah = [6,10,14,18]
    out = [1 if (lm[4].x > lm[3].x if kiri else lm[4].x < lm[3].x) else 0]
    for t, m in zip(ujung, tengah):
        out.append(1 if lm[t].y < lm[m].y else 0)
    return out

class Kalibrasi:
    N = 45

    def __init__(self):
        self.buf = {k: [] for k in ['ci','cd','cen','lap','llb','bi_y','bd_y','gap']}
        self.selesai = False
        self.batas = dict(
            ci=0.180, cd=0.180, cen_lo=0.185,
            lap=0.055, llb=0.145,
            bi_y_lo=0.30, bd_y_lo=0.30,
            gap_lo=0.10
        )

    def proses(self, lm):
        if self.selesai:
            return
        e = skala(lm)
        self.buf['ci'].append(d(lm[52], lm[159]) / e)
        self.buf['cd'].append(d(lm[282], lm[386]) / e)
        self.buf['cen'].append(d(lm[55], lm[285]) / e)
        self.buf['lap'].append(d(lm[13], lm[14]) / e)
        self.buf['llb'].append(d(lm[17], lm[152]) / e)
        self.buf['bi_y'].append(lm[55].y - lm[9].y)
        self.buf['bd_y'].append(lm[285].y - lm[9].y)
        self.buf['gap'].append(abs(lm[55].x - lm[285].x))
        if len(self.buf['ci']) >= self.N:
            self._hitung()

    def _hitung(self):
        m  = lambda k: float(np.median(self.buf[k]))
        s  = lambda k: float(np.std(self.buf[k]))
        mg_c = lambda k: max(1.5 * s(k), 0.015)
        mg_b = lambda k, mn: max(3 * s(k), mn)
        self.batas['ci']      = m('ci')  + mg_c('ci')
        self.batas['cd']      = m('cd')  + mg_c('cd')
        self.batas['cen_lo']  = m('cen') - mg_c('cen')
        self.batas['lap']     = m('lap') + mg_b('lap', 0.032)
        self.batas['llb']     = m('llb') - mg_b('llb', 0.018)
        self.batas['bi_y_lo'] = m('bi_y') + mg_c('bi_y')
        self.batas['bd_y_lo'] = m('bd_y') + mg_c('bd_y')
        self.batas['gap_lo']  = m('gap')  - mg_c('gap')
        self.selesai = True

    @property
    def progres(self):
        return min(len(self.buf['ci']) / self.N, 1.0)

def deteksi_peace(jari):
    return jari[1:] == [1, 1, 0, 0]

def deteksi_nunjuk(jari):
    return jari[1:] == [1, 0, 0, 0]

def deteksi_jari_tengah(jari):
    return jari[1:] == [0, 1, 0, 0]

def deteksi_jempol(jari):
    # Jempol terbuka, 4 jari tertutup rapat
    return jari == [1, 0, 0, 0, 0]

def deteksi_dab(jari):
    # Pemicu Dab: Gestur 5 jari terbuka penuh (High Five)
    return jari[1:] == [1, 1, 1, 1]

def deteksi_mikir(tangan, lm_wajah):
    mulut = lm_wajah[13]
    return any(d(lm[8], mulut) < 0.10 for _, lm in tangan)

def deteksi_shock(lm_wajah, kal):
    e = skala(lm_wajah)
    # Pemicu Shock diubah menjadi lebih peka (hanya 1.2x dari batas netral)
    return d(lm_wajah[13], lm_wajah[14]) / e > kal.batas['lap'] * 1.2

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109,10]
EYE_L  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
EYE_R  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
BROW_L = [70,63,105,66,107,55,65,52,53,46]
BROW_R = [300,293,334,296,336,285,295,282,283,276]
LIPS_OUT = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
LIPS_IN  = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]
NOSE = [168,6,197,195,5,4,1,19,94,2]

def gambar_wajah_minimal(frame, lm, W, H, kal):
    e        = skala(lm)
    mulut_aktif = (d(lm[13], lm[14]) / e > kal.batas['lap'] and
                d(lm[17], lm[152]) / e < kal.batas['llb'])

    WARNA_DASAR = (140, 200, 140)
    WARNA_AKTIF = (80,  240,  80)
    WARNA_MULUT = WARNA_AKTIF if mulut_aktif else WARNA_DASAR

    def buat_garis(indeks, warna, tutup=False):
        pts = [px(lm[i], W, H) for i in indeks]
        for j in range(len(pts) - 1):
            cv2.line(frame, pts[j], pts[j+1], warna, 1, cv2.LINE_AA)
        if tutup and len(pts) > 1:
            cv2.line(frame, pts[-1], pts[0], warna, 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 1, warna, -1, cv2.LINE_AA)

    buat_garis(FACE_OVAL, WARNA_DASAR, tutup=False)
    buat_garis(EYE_L,     WARNA_DASAR, tutup=True)
    buat_garis(EYE_R,     WARNA_DASAR, tutup=True)
    buat_garis(BROW_L,    WARNA_DASAR)
    buat_garis(BROW_R,    WARNA_DASAR)
    buat_garis(NOSE,      WARNA_DASAR)
    buat_garis(LIPS_OUT,  WARNA_MULUT, tutup=True)
    buat_garis(LIPS_IN,   WARNA_MULUT, tutup=True)


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def gambar_tangan_minimal(frame, lm, W, H, jari):
    WARNA = (140, 200, 140)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px(lm[a], W, H), px(lm[b], W, H), WARNA, 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(frame, px(lm[i], W, H), 2, WARNA, -1, cv2.LINE_AA)
    for i, ujung in enumerate([4, 8, 12, 16, 20]):
        if jari[i]:
            cv2.circle(frame, px(lm[ujung], W, H), 3, (80, 240, 80), -1, cv2.LINE_AA)


def hud(frame, img_sekarang, info_tangan, W, H):
    nama = img_sekarang if img_sekarang else "netral"
    warna = (80, 220, 80) if img_sekarang else (160, 160, 160)
    ov    = frame.copy()
    cv2.rectangle(ov, (8, 8), (min(W - 8, 14 + len(nama) * 14 + 20), 36), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, nama, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, warna, 2, cv2.LINE_AA)
    for i, (sisi, jari) in enumerate(info_tangan):
        cv2.putText(frame, f"{sisi}: {jari}", (14, 58 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Kamera tidak dapat dibuka")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):
        cap.read()

    ret, frame0 = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame awal")
        cap.release()
        return

    frame0 = cv2.flip(frame0, 1)
    H, W   = frame0.shape[:2]
    latar  = np.full((H, W, 3), 30, dtype=np.uint8)

    cv2.namedWindow("Kamera Kamu",      cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Meme Terdeteksi", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Kamera Kamu",      frame0)
    cv2.imshow("Meme Terdeteksi", latar)
    cv2.waitKey(1)

    kal          = Kalibrasi()
    buf          = deque(maxlen=10)
    img_sekarang = None
    MIN_VOTE     = 6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr    = face_mesh.process(rgb)
        hr    = hands_det.process(rgb)

        det         = None
        lm_wajah    = None
        tangan      = []
        info_tangan = []

        if not kal.selesai:
            pct = kal.progres
            ov  = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cy  = H // 2
            cv2.putText(frame, "Tatap layar dengan wajah netral",
                        (W // 2 - 200, cy - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (200, 200, 200), 2, cv2.LINE_AA)
            bx1, bx2 = W // 2 - 140, W // 2 + 140
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (40, 40, 40), -1)
            cv2.rectangle(frame, (bx1, cy + 10),
                          (bx1 + int(280 * pct), cy + 28), (80, 220, 80), -1)
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (120, 120, 120), 1)
            cv2.putText(frame, f"{int(pct * 100)}%",
                        (W // 2 - 18, cy + 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (160, 160, 160), 1, cv2.LINE_AA)
            if fr.multi_face_landmarks:
                kal.proses(fr.multi_face_landmarks[0].landmark)
            cv2.imshow("Kamera Kamu",      frame)
            cv2.imshow("Meme Terdeteksi", latar)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        if fr.multi_face_landmarks:
            lm_wajah = fr.multi_face_landmarks[0].landmark
            # gambar_wajah_minimal(frame, lm_wajah, W, H, kal)

        if hr.multi_hand_landmarks:
            for i, hl in enumerate(hr.multi_hand_landmarks):
                lm   = hl.landmark
                kiri = hr.multi_handedness[i].classification[0].label == "Left"
                jari = status_jari(lm, kiri)
                gambar_tangan_minimal(frame, lm, W, H, jari) 
                tangan.append((jari, lm))
                info_tangan.append(("Kiri" if kiri else "Kanan", jari))

        # Mengatur download (6).jpg sebagai gambar default selama wajah terdeteksi
        if lm_wajah:
            det = "download (6).jpg"

        # Prioritas 1: Wajah kaget (Mangap lebar)
        if lm_wajah and deteksi_shock(lm_wajah, kal):
            det = "shock.jpg"
        
        # Prioritas 2: Interaksi wajah dan jari (Pose Mikir)
        elif lm_wajah and tangan and deteksi_mikir(tangan, lm_wajah):
            det = "Para figurinha.jpg"
        
        # Prioritas 3: Gestur murni berdasarkan kombinasi jari
        elif len(tangan) >= 1:
            jari_m, lm_m = tangan[0]
            if deteksi_jempol(jari_m):
                det = "jempol.jpg"
            elif deteksi_dab(jari_m):
                det = "dab.jpg"
            elif deteksi_peace(jari_m):
                det = "peace monke.jpg"
            elif deteksi_jari_tengah(jari_m):
                det = "download (5).jpg"
            elif deteksi_nunjuk(jari_m):
                det = "download (4).jpg"

        buf.append(det)
        hitungan    = Counter(buf)
        top, voting = hitungan.most_common(1)[0]
        if voting >= MIN_VOTE:
            img_sekarang = top

        hud(frame, img_sekarang, info_tangan, W, H)
        cv2.imshow("Kamera Kamu", frame)

        if img_sekarang:
            meme = cv2.imread(img_sekarang)
            if meme is not None and meme.size > 0:
                cv2.imshow("Meme Terdeteksi", cv2.resize(meme, (W, H)))
            else:
                err = latar.copy()
                cv2.putText(err, f"Tidak ditemukan: {img_sekarang}", (20, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 220), 2)
                cv2.imshow("Meme Terdeteksi", err)
        else:
            cv2.imshow("Meme Terdeteksi", latar)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()