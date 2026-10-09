import cv2
import json
import os
import math
from ultralytics import YOLO

class BallTracker:
    def __init__(self, video_path, output_json_path):
        print("[BALL TRACKER] Booting dedicated physics module...")
        self.video_path = video_path
        self.output_json_path = output_json_path
        # Using standard YOLO but we will strictly isolate class 32 (sports ball)
        self.model = YOLO("yolov8n.pt") 

    def run_tracking(self):
        print(f"[BALL TRACKER] Analyzing footage: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"[FATAL ERROR] Could not open video: {self.video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ball_telemetry = {}
        
        frame_idx = 0
        last_known_ball = None
        frames_since_last_seen = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Scan frame strictly for sports balls
            results = self.model(frame, classes=[32], conf=0.05, verbose=False)
            
            current_ball = None
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # If multiple balls are detected (e.g., on the sideline), grab the one with highest confidence
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()
                
                best_conf = -1
                for box, conf in zip(boxes, confidences):
                    if conf > best_conf:
                        best_conf = conf
                        current_ball = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

            # --- MEMORY BUFFER LOGIC ---
            if current_ball is not None:
                # Ball found: Update memory
                ball_telemetry[str(frame_idx)] = {"cx": current_ball[0], "cy": current_ball[1], "visible": True}
                last_known_ball = current_ball
                frames_since_last_seen = 0
            else:
                # Ball lost: Use memory if it hasn't been gone too long (e.g., 10 frames)
                if last_known_ball is not None and frames_since_last_seen < 30:
                    ball_telemetry[str(frame_idx)] = {"cx": last_known_ball[0], "cy": last_known_ball[1], "visible": False}
                    frames_since_last_seen += 1
                else:
                    ball_telemetry[str(frame_idx)] = None

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"[BALL TRACKER] Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        
        # Save the telemetry map to a JSON file
        with open(self.output_json_path, "w") as f:
            json.dump(ball_telemetry, f, indent=4)
            
        print(f"[BALL TRACKER] Success! Ball coordinates mapped to: {self.output_json_path}")


# You can run this file directly to generate the data!
if __name__ == "__main__":
    WORKSPACE_DIR = r"D:\project 5"
    INPUT_VIDEO = os.path.join(WORKSPACE_DIR, "phase1.mp4")
    OUTPUT_JSON = os.path.join(WORKSPACE_DIR, "ball_telemetry.json")
    
    tracker = BallTracker(INPUT_VIDEO, OUTPUT_JSON)
    tracker.run_tracking()