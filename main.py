import cv2
import math
import os
import sys
import json
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. MASTER CONFIGURATION
# ==========================================
WORKSPACE_DIR = r"D:\project 5"
INPUT_VIDEO = os.path.join(WORKSPACE_DIR, "phase3.mp4")
OUTPUT_VIDEO = os.path.join(WORKSPACE_DIR, "rendered_phase3.mp4")
BALL_TELEMETRY = os.path.join(WORKSPACE_DIR, "ball_telemetry3"".json")
SAVED_TARGETS = os.path.join(WORKSPACE_DIR, "saved_targets3.json") 
TRACKER_YAML = os.path.join(WORKSPACE_DIR, "botsort_custom.yaml")

# --- SPEED OPTIMIZATIONS ---
STRIDE = 2                       
YOLO_MODEL = "yolov8s.pt"        
IMGSZ = 960                      
CONF_THRESH = 0.25               

class AIPsychiatristEngine:
    def __init__(self):
        print("[SYSTEM] Booting AI Psychiatrist Engine (High-Speed Mode)...")
        self.model = YOLO(YOLO_MODEL) 
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.memory_targets = []
        self.memory_telemetry = []
        self.id_to_slot = {}
        self.player_names = {} 
        self._create_tracker_yaml()

    def _create_tracker_yaml(self):
        yaml_content = """tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.3
track_buffer: 90
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
model: auto
"""
        with open(TRACKER_YAML, "w") as f:
            f.write(yaml_content)

    # ==========================================
    # STAGE 1: ACQUISITION (Target Locking)
    # ==========================================
    def run_acquisition(self):
        print("\n=== STAGE 1: TARGET ACQUISITION ===")
        
        if os.path.exists(SAVED_TARGETS):
            print(f"[SYSTEM] Found previous targets at {SAVED_TARGETS}!")
            use_saved = input(">>> Do you want to use the saved boxes? (y/n): ").strip().lower()
            if use_saved == 'y':
                with open(SAVED_TARGETS, "r") as f:
                    data = json.load(f)
                    self.memory_targets = data.get("targets", [])
                    self.player_names = data.get("names", {})
                print(f"[SUCCESS] Loaded {len(self.memory_targets)} boxes instantly. Skipping manual draw!")
                return 
            else:
                print("[SYSTEM] Ignoring saved boxes. Starting fresh...")

        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            print(f"[FATAL ERROR] Could not open video: {INPUT_VIDEO}")
            sys.exit()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        window_name = "Stage 1 - Target Lock"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print(" [D] / [A] -> Skip 30 frames (+/- 1s)")
        print(" [W] / [S] -> Fine step 5 frames")
        print(" [.] / [,] -> Micro step 1 frame")
        print(" [SPACE]   -> Freeze this frame & Draw Box")
        print(" [F]       -> FINISH & SAVE TO ENGINE")
        
        while True:
            frame_index = max(0, min(frame_index, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            
            if not success:
                frame_index = max(0, frame_index - 10)
                continue

            display_frame = frame.copy()
            hud_text = f"Frame: {frame_index}/{total_frames} | Boxes: {len(self.memory_targets)} | [SPACE] Box | [F] DONE"
            cv2.putText(display_frame, hud_text, (20, 40), self.font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(0) & 0xFF

            if key in [ord('d'), ord('D')]: frame_index += 30
            elif key in [ord('a'), ord('A')]: frame_index = max(0, frame_index - 30)
            elif key in [ord('w'), ord('W')]: frame_index += 5
            elif key in [ord('s'), ord('S')]: frame_index = max(0, frame_index - 5)
            elif key in [ord('.'), ord('>')]: frame_index += 1
            elif key in [ord(','), ord('<')]: frame_index = max(0, frame_index - 1)
            elif key in [ord('q'), ord('Q'), 27]:
                print("\n[ABORT] Canceled by user.")
                sys.exit()
            elif key in [ord('f'), ord('F')]:
                if len(self.memory_targets) == 0:
                    print("\n[WARNING] You haven't boxed any players yet!")
                    continue
                
                with open(SAVED_TARGETS, "w") as f:
                    json.dump({"targets": self.memory_targets, "names": self.player_names}, f, indent=4)
                
                print(f"\n[SUCCESS] {len(self.memory_targets)} boxes saved permanently to {SAVED_TARGETS}!")
                print("Pushing to Telemetry...")
                break
            elif key == 32:  
                box = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
                if box != (0, 0, 0, 0) and box[2] > 5 and box[3] > 5:
                    print(f"\n>>> [PAUSED] Look at this terminal window.")
                    print("Who did you just draw a box on?")
                    for slot_id, name in self.player_names.items():
                        print(f" {slot_id}. {name}")
                    new_slot_id = str(len(self.player_names) + 1)
                    print(f" {new_slot_id}. [Add New Player]")
                    
                    choice = input(">>> Enter number: ").strip()
                    if choice == new_slot_id:
                        name = input(">>> Enter the new player's name: ").strip()
                        self.player_names[new_slot_id] = name
                        assigned_slot = new_slot_id
                        print(f">>> [{name}] created and locked!")
                    elif choice in self.player_names:
                        assigned_slot = choice
                        print(f">>> Box successfully linked to [{self.player_names[assigned_slot]}]!")
                    else:
                        print(">>> Invalid choice! Box discarded.")
                        continue

                    self.memory_targets.append({"slot_id": assigned_slot, "frame": frame_index, "box": list(box)})

        cv2.destroyAllWindows()
        cap.release()

    def _calculate_iou(self, yolo_box, drawn_box):
        xA, yA = max(yolo_box[0], drawn_box[0]), max(yolo_box[1], drawn_box[1])
        xB, yB = min(yolo_box[2], drawn_box[0] + drawn_box[2]), min(yolo_box[3], drawn_box[1] + drawn_box[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (yolo_box[2] - yolo_box[0]) * (yolo_box[3] - yolo_box[1])
        boxBArea = drawn_box[2] * drawn_box[3]
        denom = float(boxAArea + boxBArea - interArea)
        return 0.0 if denom <= 0 else interArea / denom

    # ==========================================
    # STAGE 2: BUILT-IN BALL TRACKER
    # ==========================================
    def _generate_ball_telemetry(self):
        print("\n[SYSTEM] Auto-initiating built-in high-speed ball tracker...")
        cap = cv2.VideoCapture(INPUT_VIDEO)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_data = {}
        i = 0
        
        while True:
            ok, frame = cap.read()
            if not ok: break
            
            # Print logic moved UP before the stride continue so it actually prints
            if i % 100 == 0:
                print(f"  [BALL TRACKER] Scanning frame {i}/{total_frames}")
                
            if i % STRIDE != 0:
                raw_data[str(i)] = {"visible": False}
                i += 1
                continue
                
            r = self.model.predict(frame, classes=[32], conf=0.05, imgsz=IMGSZ, verbose=False)[0]
            best = None
            if r.boxes is not None and len(r.boxes):
                xy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                for b, c in sorted(zip(xy, cf), key=lambda t: -t[1]):
                    cx = float((b[0] + b[2]) / 2.0)
                    cy = float((b[1] + b[3]) / 2.0)
                    best = (cx, cy)
                    break
            
            if best:
                raw_data[str(i)] = {"visible": True, "cx": float(best[0]), "cy": float(best[1])}
            else:
                raw_data[str(i)] = {"visible": False}
                
            i += 1
                
        cap.release()
        
        with open(BALL_TELEMETRY, "w") as f:
            json.dump(raw_data, f, indent=4)
        print(f"[SUCCESS] Ball telemetry saved to {BALL_TELEMETRY}")
        return raw_data

    # ==========================================
    # STAGE 3: TELEMETRY GENERATION
    # ==========================================
    def run_telemetry(self):
        print("\n=== STAGE 3: TELEMETRY GENERATION ===")
        
        if not os.path.exists(BALL_TELEMETRY):
            ball_data = self._generate_ball_telemetry()
        else:
            with open(BALL_TELEMETRY, "r") as f:
                ball_data = json.load(f)
        
        cap = cv2.VideoCapture(INPUT_VIDEO)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        active_track_ids = set()
        player_history = {}
        player_focus = {}
        player_scan_time = {}
        persistent_boxes = {}
        track_timeouts = {} 
        possession_states = {}
        
        target_frames = {t["frame"] for t in self.memory_targets}
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % STRIDE != 0 and frame_idx not in target_frames:
                frame_idx += 1
                continue

            timestamp = float(frame_idx / fps)
            
            results = self.model.track(
                frame, 
                persist=True, 
                tracker=TRACKER_YAML, 
                classes=[0], 
                conf=CONF_THRESH,  
                iou=0.4,
                imgsz=IMGSZ,
                verbose=False
            )
            current_ball = ball_data.get(str(frame_idx))

            for tgt in self.memory_targets:
                if tgt["frame"] == frame_idx and results[0].boxes.id is not None:
                    y_boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    y_ids = results[0].boxes.id.int().cpu().tolist()
                    
                    best_iou, best_id = 0, -1
                    for yb, yid in zip(y_boxes, y_ids):
                        iou = self._calculate_iou(yb, tgt["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_id = yid
                    if best_iou > 0.2:
                        active_track_ids.add(best_id)
                        self.id_to_slot[str(best_id)] = tgt["slot_id"]
                        track_timeouts[best_id] = 0 

            frame_data = {"frame_idx": int(frame_idx), "timestamp": float(timestamp), "players": {}}
            current_frame_detections = {}

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    if track_id in active_track_ids:
                        current_frame_detections[track_id] = box

            for track_id in list(active_track_ids):
                if track_id in current_frame_detections:
                    persistent_boxes[track_id] = current_frame_detections[track_id]
                    track_timeouts[track_id] = 0 
                else:
                    track_timeouts[track_id] = track_timeouts.get(track_id, 0) + 1
                    if track_timeouts[track_id] > 2 and track_id in persistent_boxes:
                        del persistent_boxes[track_id]
                    if track_timeouts[track_id] > 90: 
                        active_track_ids.remove(track_id)
                        continue 

            for track_id in active_track_ids:
                if track_id not in persistent_boxes: 
                    continue 
                    
                box = persistent_boxes[track_id]
                str_id = str(track_id)
                x1, y1, x2, y2 = box
                cx, cy = float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
                
                margin_x = frame_w * 0.05
                margin_y = frame_h * 0.05
                is_out_of_bounds = (cx < margin_x or cx > (frame_w - margin_x) or cy < margin_y or cy > (frame_h - margin_y))

                has_ball = False
                if current_ball is not None and current_ball.get("visible", False):
                    bcx, bcy = float(current_ball["cx"]), float(current_ball["cy"])
                    if (x1 - 10) <= bcx <= (x2 + 10) and (y1) <= bcy <= (y2 + 30):
                        has_ball = True

                if track_id not in player_history:
                    player_history[track_id] = []
                    player_focus[track_id] = 0.95
                    player_scan_time[track_id] = 0.0

                speed, heading, std_dev = 0.0, 0.0, 0.0
                if len(player_history[track_id]) >= 1:
                    last_cx, last_cy, last_heading = player_history[track_id][-1]

                    # 1. Instant speed for dribble/possession timers
                    speed = math.hypot(cx - last_cx, cy - last_cy)

                    # 2. THE MACRO-HEADING FIX (Solves grapple/tackle jitter)
                    # Look back up to 5 frames to find their TRUE trajectory
                    lookback_frames = min(5, len(player_history[track_id]))
                    past_cx, past_cy, _ = player_history[track_id][-lookback_frames]

                    macro_dx = cx - past_cx
                    macro_dy = cy - past_cy

                    # If they haven't moved a net distance of 5 pixels over the last 5 frames, they are tangled/stationary
                    if math.hypot(macro_dx, macro_dy) > 5.0:
                        heading = math.degrees(math.atan2(macro_dy, macro_dx))
                    else:
                        heading = last_heading

                player_history[track_id].append((cx, cy, heading))
                if len(player_history[track_id]) > 15:
                    player_history[track_id].pop(0)

                headings = [h for _, _, h in player_history[track_id]]
                if len(headings) >= 5:
                    # FIX 1: Circular Standard Deviation (Solves the -180/180 leftward boundary wrap)
                    sin_sum = sum(math.sin(math.radians(h)) for h in headings)
                    cos_sum = sum(math.cos(math.radians(h)) for h in headings)
                    mx = cos_sum / len(headings)
                    my = sin_sum / len(headings)
                    R = max(1e-6, min(1.0, math.hypot(mx, my)))
                    std_dev = math.degrees(math.sqrt(-2.0 * math.log(R)))

                # COGNITIVE FLOW & SCAN MATH
                if std_dev > 10.0:
                    player_focus[track_id] = min(0.99, player_focus[track_id] + 0.04)
                    player_scan_time[track_id] += (float(STRIDE) / float(fps))
                elif std_dev < 7.0:
                    player_focus[track_id] = max(0.35, player_focus[track_id] - 0.03)
                    player_scan_time[track_id] = 0.0

                if track_id not in possession_states:
                    possession_states[track_id] = {
                        "start_ts": None, 
                        "proc_time": 0.0, 
                        "total_time": 0.0, 
                        "freeze_until": 0.0,
                        "missed_frames": 0
                    }

                p_state = possession_states[track_id]

                if has_ball:
                    p_state["missed_frames"] = 0
                    if p_state["start_ts"] is None:
                        p_state["start_ts"] = timestamp
                        p_state["proc_time"] = 0.0

                    p_state["total_time"] = timestamp - p_state["start_ts"]

                    if speed > 1.0 and p_state["proc_time"] == 0.0:
                        p_state["proc_time"] = p_state["total_time"]
                else:
                    if p_state["start_ts"] is not None:
                        p_state["missed_frames"] += 1
                        p_state["total_time"] = timestamp - p_state["start_ts"]

                        # FIX 2: 10-Frame Buffer (Prevents the 0.00s timer wipe)
                        if p_state["missed_frames"] > 10:
                            if p_state["proc_time"] == 0.0:
                                p_state["proc_time"] = p_state["total_time"]

                            p_state["freeze_until"] = timestamp + 3.0
                            p_state["start_ts"] = None
                            p_state["missed_frames"] = 0

                show_timers = has_ball or (timestamp < p_state["freeze_until"])

                frame_data["players"][str_id] = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "focus_score": round(float(player_focus[track_id]), 3),
                    "scan_time": float(player_scan_time[track_id]),
                    "total_time": float(p_state["total_time"]),
                    "proc_time": float(p_state["proc_time"]),
                    "show_timers": bool(show_timers),
                    "has_ball_live": bool(has_ball),
                    "heading_angle": float(heading),
                    "vision_trigger_active": bool(std_dev > 10.0),
                    "is_out_of_bounds": bool(is_out_of_bounds)
                }

            self.memory_telemetry.append(frame_data)
            for _ in range(1, STRIDE):
                if frame_idx % STRIDE == 0: 
                    self.memory_telemetry.append(frame_data)

            frame_idx += 1
            if frame_idx % 100 == 0: 
                print(f"Processing Tactical Physics... {frame_idx}/{total_frames}")

        cap.release()
        print("[SUCCESS] Telemetry generated. Handoff to Renderer...")

    # ==========================================
    # STAGE 4: CINEMATIC RENDERING
    # ==========================================
    def run_rendering(self):
        print("\n=== STAGE 4: CINEMATIC RENDERING ===")
        cap = cv2.VideoCapture(INPUT_VIDEO)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_data = self.memory_telemetry[frame_idx] if frame_idx < len(self.memory_telemetry) else {}
            players = frame_data.get("players", {})

            for str_id, p_info in players.items():
                if p_info.get("is_out_of_bounds", False):
                    continue

                x1, y1, x2, y2 = [int(v) for v in p_info["bbox"]]
                slot = self.id_to_slot.get(str_id, "1")
                name = self.player_names.get(slot, "UNKNOWN")
                score = p_info.get("focus_score", 0.9)
                scan_time = p_info.get("scan_time", 0.0)

                box_color = (0, 255, 255) if score >= 0.5 else (0, 0, 255)

                cl = 15
                thick = 4
                for pt1, pt2 in [((x1, y1), (x1 + cl, y1)), ((x1, y1), (x1, y1 + cl)), 
                                 ((x2, y1), (x2 - cl, y1)), ((x2, y1), (x2, y1 + cl)),
                                 ((x1, y2), (x1 + cl, y2)), ((x1, y2), (x1, y2 - cl)),
                                 ((x2, y2), (x2 - cl, y2)), ((x2, y2), (x2, y2 - cl))]:
                    cv2.line(frame, pt1, pt2, box_color, thick)

                cv2.putText(frame, name, (x1, y1 - 10), self.font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                if p_info.get("show_timers", False):
                    t_total = p_info.get("total_time", 0.0)
                    t_proc = p_info.get("proc_time", 0.0)
                    
                    # RESTORED LOGIC: Instant Hesitation wakeup if focus drops below 65%
                    if score < 0.65:
                        timer_str = f"Hesitation: {t_total:.2f}s"
                        text_color = (0, 0, 255)
                    elif t_proc > 0.0:
                        if t_total >= 1.5:
                            timer_str = f"Proc: {t_proc:.2f}s | La Pausa: {t_total:.2f}s"
                            text_color = (0, 255, 255)
                        else:
                            timer_str = f"Proc: {t_proc:.2f}s"
                            text_color = (0, 255, 255)
                    else:
                        timer_str = f"Proc: {t_total:.2f}s"
                        text_color = (0, 255, 255)
                        
                    cv2.putText(frame, timer_str, (x1 - 10, y2 + 25), self.font, 0.5, text_color, 1, cv2.LINE_AA)
                elif scan_time > 0.1:
                    cv2.putText(frame, f"Scan: {scan_time:.2f}s", (x1 - 10, y2 + 25), self.font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                if p_info.get("vision_trigger_active", False):
                    head_center = ((x1 + x2) // 2, y1 + 15)
                    angle = p_info.get("heading_angle", 0.0)
                    overlay_cone = frame.copy()
                    cv2.ellipse(overlay_cone, head_center, (90, 90), 0, angle - 25, angle + 25, (255, 200, 50), -1)
                    cv2.addWeighted(overlay_cone, 0.4, frame, 0.6, 0, frame)

            overlay = frame.copy()
            hud_w = 340
            hud_h = max(120, 80 + (len(self.player_names) * 65))
            cv2.rectangle(overlay, (20, 60), (20 + hud_w, 60 + hud_h), (20, 20, 25), -1)
            frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
            cv2.putText(frame, "COGNITIVE FLOW // HUD", (35, 95), self.font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            y_pos = 145
            for slot_num, player_name in self.player_names.items():
                current_score = 0.50 
                
                for s_id, slot in self.id_to_slot.items():
                    if slot == slot_num and s_id in players:
                        p_info = players[s_id]
                        if not p_info.get("is_out_of_bounds", False):
                            current_score = p_info["focus_score"]
                        break

                pct = int(current_score * 100)
                text_color = (0, 255, 255) if current_score == 0.50 else (200, 200, 200)
                cv2.putText(frame, f"{player_name} [{pct}%]", (35, y_pos - 8), self.font, 0.5, text_color, 1, cv2.LINE_AA)
                cv2.rectangle(frame, (35, y_pos), (35 + 260, y_pos + 16), (40, 40, 40), -1)
                
                bar_fill = int(260 * current_score)
                fill_color = (0, 255, 0) if current_score > 0.7 else ((0, 200, 255) if current_score >= 0.50 else (0, 0, 255))
                cv2.rectangle(frame, (35, y_pos), (35 + bar_fill, y_pos + 16), fill_color, -1)
                cv2.rectangle(frame, (35, y_pos), (35 + 260, y_pos + 16), (200, 200, 200), 1)
                y_pos += 65

            out.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0: 
                print(f"Rendering Video... {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        print(f"\n[DONE] Pipeline Complete! Video saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    app = AIPsychiatristEngine()
    app.run_acquisition()  
    app.run_telemetry()    
    app.run_rendering()