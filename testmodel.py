import cv2
import torch # Umumnya YOLOv10 berbasis PyTorch
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- 1. Inisialisasi Model ---

# --- Muat model YOLOv10 ---
# (Sama seperti sebelumnya, pastikan bagian ini sesuai dengan implementasi YOLOv10 Anda)
try:
    from ultralytics import YOLO # Menggunakan YOLO sebagai placeholder
    yolo_model_path = 'yolo10m.pt' # GANTI dengan path model YOLOv10 Anda yang valid
    yolo_model = YOLO(yolo_model_path)
    print(f"Model YOLO berhasil dimuat dari: {yolo_model_path}")
except Exception as e:
    print(f"Error placeholder saat memuat model YOLOv10: {e}")
    # exit()

# --- Inisialisasi DeepSORT Realtime ---
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.7,
    nn_budget=None,
    embedder='mobilenet',
    half=True,
    bgr=True,
    embedder_gpu=True,
    polygon=False,
)

# --- 2. Buka Sumber Video ---
# video_path = 0  # Gunakan 0 untuk webcam, atau path ke file video
video_path = "video2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Tidak bisa membuka video dari {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# (Opsional) Setup VideoWriter untuk menyimpan hasil
# output_path = "output_tracked_skip5.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- 3. Proses Frame demi Frame ---
frame_id_counter = 0  # Penghitung frame
PROCESS_EVERY_N_FRAMES = 5 # Proses deteksi setiap N frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Selesai memproses video atau error membaca frame.")
        break

    frame_id_counter += 1
    detections_for_deepsort = [] # Inisialisasi list deteksi untuk frame ini

    # --- 4. Deteksi Objek dengan YOLOv10 (Hanya setiap N frame) ---
    if frame_id_counter % PROCESS_EVERY_N_FRAMES == 0 or frame_id_counter == 1:
        # **SESUAIKAN BAGIAN INI DENGAN API YOLOv10 ANDA**
        try:
            results = yolo_model.predict(frame, conf=0.4, iou=0.5, verbose=False)
            
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()

                for i in range(len(boxes)):
                    bbox = boxes[i]
                    confidence = confidences[i]
                    class_id = int(class_ids[i])
                    
                    detections_for_deepsort.append(
                        ([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                         confidence,
                         class_id
                        )
                    )
            print(f"Frame {frame_id_counter}: Melakukan deteksi YOLO. Ditemukan {len(detections_for_deepsort)} objek.")
        except Exception as e:
            print(f"Error saat deteksi YOLOv10 atau pemrosesan hasil: {e}")
            detections_for_deepsort = [] # Pastikan kosong jika ada error
    # else: # Untuk frame yang di-skip, detections_for_deepsort akan tetap kosong
        # print(f"Frame {frame_id_counter}: Melewati deteksi YOLO.")


    # --- 5. Update Tracker DeepSORT ---
    # `update_tracks` dipanggil setiap frame.
    # Jika `detections_for_deepsort` kosong (pada frame yang di-skip),
    # DeepSORT akan melakukan prediksi dan aging pada track yang ada.
    try:
        tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)
    except Exception as e:
        print(f"Error saat update DeepSORT tracks: {e}")
        tracks = [] # Pastikan tracks adalah list kosong jika ada error


    # --- 6. Gambar Hasil Tracking pada Frame ---
    for track in tracks:
        if not track.is_confirmed(): # Hanya gambar track yang sudah terkonfirmasi
            continue 

        track_id = track.track_id
        ltrb = track.to_ltrb() # Mendapatkan koordinat bounding box [kiri, atas, kanan, bawah]
        # class_id_tracked = track.get_det_class() # Mendapatkan ID kelas dari deteksi asli
        # (Anda bisa mendapatkan nama kelas jika model YOLO Anda menyediakannya)
        # class_name_tracked = yolo_model.names[int(class_id_tracked)] if hasattr(yolo_model, 'names') and int(class_id_tracked) in yolo_model.names else f'ClsID:{class_id_tracked}'


        # Konversi koordinat ke integer untuk digambar
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # --- INI BAGIAN YANG MENGGAMBAR BOUNDING BOX ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Penjelasan:
        # frame: Gambar (frame video) tempat bounding box akan digambar.
        # (x1, y1): Koordinat pojok kiri atas dari bounding box.
        # (x2, y2): Koordinat pojok kanan bawah dari bounding box.
        # (0, 255, 0): Warna bounding box dalam format BGR (Biru, Hijau, Merah).
        #              Dalam contoh ini, warnanya hijau.
        # 2: Ketebalan garis bounding box.

        # Tampilkan ID dan Kelas (opsional, sudah ada di kode sebelumnya)
        label = f"ID:{track_id}" # Anda bisa tambahkan kelas jika mau: f"ID:{track_id} C:{class_name_tracked}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- 7. Tampilkan Hasil ---
    cv2.imshow(f"YOLOv10 + DeepSORT (Proses tiap {PROCESS_EVERY_N_FRAMES} frame)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Tekan 'q' untuk keluar
        break

# --- 8. Bersihkan ---
cap.release()
# out.release() # Jika menggunakan VideoWriter
cv2.destroyAllWindows()