import torch
import torch.nn # Required for torch.nn.Sequential (if it's part of the pickled model)
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# --- BEGIN UPDATED FIX FOR _pickle.UnpicklingError ---
# This block attempts to make specified classes safe for unpickling by torch.load,
# which is necessary for PyTorch 2.6+ if model files contain these pickled classes.

# List of classes that might need to be added to safe globals
# based on different UnpicklingErrors encountered.
classes_to_make_safe = []

# Attempt to add ultralytics.nn.tasks.DetectionModel
try:
    from ultralytics.nn.tasks import DetectionModel
    classes_to_make_safe.append(DetectionModel)
    print("Info: Included 'ultralytics.nn.tasks.DetectionModel' in the list for safe globals.")
except ImportError:
    print("Info: 'ultralytics.nn.tasks.DetectionModel' could not be imported. "
          "It will not be added to safe globals. If your model doesn't require it, this is fine.")

# Attempt to add torch.nn.Sequential (based on potential error messages)
try:
    # torch.nn.Sequential is the standard way to access torch.nn.modules.container.Sequential
    classes_to_make_safe.append(torch.nn.Sequential)
    print("Info: Included 'torch.nn.Sequential' in the list for safe globals.")
except AttributeError: # Should not generally happen with a proper PyTorch installation
    print("Warning: 'torch.nn.Sequential' could not be accessed via torch.nn. "
          "This might indicate an issue with your PyTorch installation. "
          "It will not be added to safe globals.")

# Apply the safe globals if any classes were collected and the PyTorch function exists
if classes_to_make_safe:
    if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        try:
            torch.serialization.add_safe_globals(classes_to_make_safe)
            print(f"Successfully attempted to add {len(classes_to_make_safe)} class(es) to torch's safe globals for unpickling.")
        except Exception as e:
            print(f"Error calling torch.serialization.add_safe_globals: {e}")
    else:
        # This message is relevant if using an older PyTorch version where the error shouldn't occur,
        # or if the function is somehow missing in a newer version (unlikely).
        print("Info: `torch.serialization.add_safe_globals` not found. "
              "This fix is primarily for PyTorch 2.6+ where `weights_only` defaults to True for `torch.load`.")
else:
    print("Info: No specific problematic classes were pre-identified or imported to add to torch safe_globals.")
# --- END UPDATED FIX ---

def main():
    # Load YOLO model.
    # The tracebacks indicated errors with 'best.pt'.
    # Change 'best.pt' to your actual model file name if different (e.g., 'yolo11m.pt').
    model_path = 'best70_12k.pt'
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded YOLO model from {model_path}")
    except Exception as e:
        print(f"Error loading YOLO model ({model_path}): {e}")
        print("Ensure the model file exists at the specified path and is not corrupted.")
        print("If an UnpicklingError persists, check the console messages from the fix applied above, "
              "and ensure all necessary classes are added to 'classes_to_make_safe'.")
        return

    # Initialize DeepSort tracker with basic settings
    tracker = DeepSort(max_age=15, n_init=2) # You can adjust these parameters

    # Vehicle class IDs in COCO dataset that YOLO might detect.
    # Common classes: car (2), motorcycle (3), bus (5), truck (7)
    # Adjust these IDs if your model uses a different dataset or class mapping.
    VEHICLE_CLASS_IDS = [0, 3, 1 , 2]

    input_video_path = 'video2.mp4' # Replace with your input video file
    output_video_path = 'output_vehicle_tracking.avi'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Fallback if fps is not readable
        print("Warning: FPS not readable from video, defaulting to 30.")
        fps = 30

    # Define the codec and create VideoWriter object
    # XVID is a common codec for .avi files. Alternatives: MJPG, MP4V (for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for output file {output_video_path}")
        cap.release()
        return
        
    print(f"Processing video: {input_video_path}")
    print(f"Output will be saved to: {output_video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or error in reading frame.")
            break

        frame_count += 1
        # Perform inference
        results = model(frame) #, verbose=False) # Set verbose=False to reduce console output from YOLO

        detections_for_deepsort = []
        for result in results: # Iterate through results (usually one per image)
            for box in result.boxes: # Iterate through detected boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                # Filter for vehicle classes and confidence threshold
                if class_id in VEHICLE_CLASS_IDS and confidence >= 0.25:
                    w = x2 - x1
                    h = y2 - y1
                    # DeepSort expects detections in format: ([x, y, w, h], confidence, class_name/id)
                    detections_for_deepsort.append(([x1, y1, w, h], confidence, class_id))

        # Update tracker with new detections
        # The `frame` argument is used by DeepSort for re-identification features if enabled/configured
        tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)

        # Draw bounding boxes and IDs for confirmed tracks
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1: # Optionally filter out stale tracks
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # Get coordinates in (left, top, right, bottom) format
            
            # Ensure coordinates are integers for drawing
            cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(ltrb[0]), int(ltrb[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Optionally display the frame (can slow down processing)
        # cv2.imshow('Vehicle Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        
        if frame_count % int(fps) == 0: # Log progress every second
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished processing. Video saved to {output_video_path}")

if __name__ == '__main__':
    main()