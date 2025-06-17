from ultralytics import YOLO
import cv2

def detect_image(image_path, model_path='yolo11m.pt'):
    # Load model
    try:
        model = YOLO(model_path)
        print(f"Model berhasil dimuat dari {model_path}")
    except Exception as e:
        print(f"Error: Gagal memuat model -> {e}")
        return

    # Baca gambar
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    # Jalankan deteksi
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)

            label = f"{class_name} ({confidence:.2f})"

            # Gambar kotak dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            print(f"Deteksi: {label} | Koordinat: {x1}, {y1}, {x2}, {y2}")

    # Simpan dan tampilkan hasil
    output_path = "deteksi_saja.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Hasil disimpan ke: {output_path}")

    cv2.imshow("Deteksi YOLO", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_image("image.png")  # Ganti dengan nama gambar kamu
