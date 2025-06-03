import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

def main():
    # Load YOLOv11 model (ensure the path to your model is correct)
    model = YOLO('yolo11m.pt')

    # Initialize DeepSort tracker with basic settings
    tracker = DeepSort(max_age=15, n_init=2)

    # Vehicle class IDs in COCO dataset used by YOLO model
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    input_video_path = 'video.mp4'
    output_video_path = 'output_vehicle_tracking.avi'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open video file {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # default fallback

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Filter to include only vehicles
                if cls not in VEHICLE_CLASS_IDS:
                    continue
                
                if conf < 0.3:
                    continue

                w = x2 - x1
                h = y2 - y1
                # detections are ([x, y, w, h], confidence, class)
                detections.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            # Draw bbox and track id
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

