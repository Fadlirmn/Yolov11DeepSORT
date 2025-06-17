from ultralytics import YOLO
import cv2

def detect_vehicles_yolo(image_path, model_path='yolo11m.pt'):
    # Load YOLO model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found at path: {image_path}")
        return

    # Kendaraan yang dideteksi (ubah sesuai label model kamu)
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # Misalnya: car, motor, bus, truck
    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            # Filter hanya kendaraan
            if class_id in VEHICLE_CLASS_IDS and confidence >= 0.25:
                label = f"{class_id} {confidence:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow("YOLO Detection", image)
    cv2.imwrite("detected_yolo.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_vehicles_yolo("image.png")
