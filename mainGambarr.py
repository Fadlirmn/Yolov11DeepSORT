import torch
import torch.nn
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Fix untuk PyTorch >= 2.6 (safe globals) ---
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel, torch.nn.Sequential])
    print("Safe globals berhasil ditambahkan.")
except Exception as e:
    print(f"Gagal menambahkan safe globals: {e}")

def detect_with_tracking(image_path, model_path='best70s_15Kpic.pt'):
    # Load model YOLO
    try:
        model = YOLO(model_path)
        print(f"Model berhasil dimuat dari {model_path}")
    except Exception as e:
        print(f"Error: Gagal memuat model YOLO -> {e}")
        return

    # Baca gambar
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    # Inisialisasi DeepSORT
    tracker = DeepSort(max_age=15, n_init=1)

    # Jalankan deteksi YOLO
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            if confidence < 0.25:
                continue

            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], confidence, class_id))

    # Jalankan tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Tampilkan hasil dengan kotak dan ID
    for track in tracks:
        # Untuk hanya 1 frame, paksa tampilkan semua track yang baru terdeteksi
        if track.time_since_update > 1:
         continue


        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_id = track.det_class
        class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)

        label = f'{class_name} | ID: {track_id}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"Objek: {class_name}, ID: {track_id}, Confidence: {confidence:.2f}")

    # Simpan dan tampilkan hasil
    output_path = 'output_tracked_image.jpg'
    cv2.imwrite(output_path, frame)
    print(f"Hasil disimpan di: {output_path}")

    cv2.imshow("Tracking 1 Gambar", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_with_tracking('image.png')  # Ganti dengan nama gambar
