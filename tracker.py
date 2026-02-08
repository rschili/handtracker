import os
# Silence TensorFlow/MediaPipe noise
os.environ['GLOG_minloglevel'] = '2'

import cv2
import mediapipe as mp
import math
import uvicorn
import threading
import time
from typing import Tuple, Optional
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel

# --- Configuration & Constants ---
MODEL_PATH = 'hand_landmarker.task'
CAMERA_INDEX = 0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Wrist to Pinky
]

# --- Data Models ---
class TrackerConfig(BaseModel):
    min_ratio: float
    max_ratio: float

class TrackerStatus(BaseModel):
    percentage: int
    detected: bool
    hand: str
    orientation: str

# --- The Tracker Engine ---
class HandTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # State
        self.percentage = 0.0
        self.is_detected = False
        self.hand_label = "None"
        self.orientation = "None"
        self.latest_frame_bytes = None
        
        # Config (Tuned Defaults)
        self.min_ratio = 0.28
        self.max_ratio = 0.95
        
        # Optimization: Only encode JPEGs if clients are watching
        self.active_clients = 0

        # --- SMOOTHING ---
        self.history = deque(maxlen=15)

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"--- Tracker Started ---")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        print("--- Tracker Stopped ---")

    def update_config(self, min_r, max_r):
        with self.lock:
            self.min_ratio = min_r
            self.max_ratio = max_r
            self.history.clear()
            print(f"Config Updated: {self.min_ratio} - {self.max_ratio}")

    def _get_dist(self, p1, p2) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _calculate_logic(self, landmarks) -> Tuple[float, str]:
        # 4=ThumbTip, 5=IndexMCP, 6=IndexPIP, 17=PinkyMCP
        t_tip, i_mcp = landmarks[4], landmarks[5]
        i_pip, p_mcp = landmarks[6], landmarks[17]

        # 1. Orientation Analysis
        dy = p_mcp.y - i_mcp.y 
        dx = p_mcp.x - i_mcp.x
        
        # If horizontal spread > vertical spread, ignore it
        if abs(dx) > abs(dy):
            return 0.0, "Horizontal"
            
        is_up = dy > 0
        orientation = "Up" if is_up else "Down"

        # 2. Palm Width (Scale)
        width = self._get_dist(i_mcp, p_mcp)
        if width == 0: return 0.0, orientation

        # 3. Elevation (Thumb)
        ref_y = i_mcp.y
        
        if orientation == "Up":
            elevation = ref_y - t_tip.y
        else:
            elevation = t_tip.y - ref_y
            
        if elevation < 0: return 0.0, orientation

        return elevation / width, orientation

    def _loop(self):
        if not os.path.exists(MODEL_PATH):
            print(f"CRITICAL ERROR: {MODEL_PATH} missing.")
            return

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cap = cv2.VideoCapture(CAMERA_INDEX)
        
        with HandLandmarker.create_from_options(options) as landmarker:
            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    time.sleep(0.1)
                    continue

                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

                raw_ratio = 0.0
                orient = "None"
                detected = False
                label = "None"
                draw_proto = None

                if result.hand_landmarks:
                    detected = True
                    closest = 0
                    min_d = 100.0
                    for i, lm in enumerate(result.hand_landmarks):
                        d = math.hypot(lm[0].x - 0.5, lm[0].y - 0.5)
                        if d < min_d: min_d, closest = d, i
                    
                    target = result.hand_landmarks[closest]
                    draw_proto = target
                    if len(result.handedness) > closest:
                        label = result.handedness[closest][0].category_name
                    
                    raw_ratio, orient = self._calculate_logic(target)

                # --- SMOOTHING LOGIC ---
                final_percentage = 0.0
                
                if detected and orient in ["Up", "Down"]:
                    if raw_ratio < self.min_ratio:
                        target_pct = 0.0
                    else:
                        norm = (raw_ratio - self.min_ratio) / (self.max_ratio - self.min_ratio)
                        val = max(0.0, min(1.0, norm))
                        if val > 0.99: val = 1.0 
                        target_pct = val * 100.0
                    
                    self.history.append(target_pct)
                    final_percentage = sum(self.history) / len(self.history)
                else:
                    self.history.clear()
                    final_percentage = 0.0

                # Update State
                with self.lock:
                    self.percentage = final_percentage
                    self.is_detected = detected
                    self.hand_label = label
                    self.orientation = orient
                    
                    if self.active_clients > 0:
                        self._draw_debug(frame, draw_proto, final_percentage, orient)
                        _, buf = cv2.imencode('.jpg', frame)
                        self.latest_frame_bytes = buf.tobytes()

                time.sleep(0.01)
        cap.release()

    def _draw_debug(self, frame, landmarks, val, orient):
        if not landmarks: 
             cv2.putText(frame, "No Hand", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
             return

        h, w, _ = frame.shape
        px = [(int(l.x*w), int(l.y*h)) for l in landmarks]

        for i, j in HAND_CONNECTIONS:
            if i<len(px) and j<len(px): cv2.line(frame, px[i], px[j], (0,255,0), 1)

        i_mcp, i_pip = px[5], px[6]
        t_tip = px[4]
        ref_y = int((i_mcp[1] + i_pip[1]) / 2)

        cv2.line(frame, (px[5][0]-40, ref_y), (px[5][0]+40, ref_y), (255,0,0), 2)
        cv2.line(frame, px[5], px[17], (0,255,255), 2)

        if (orient == "Up" and t_tip[1] < ref_y) or (orient == "Down" and t_tip[1] > ref_y):
            cv2.line(frame, t_tip, (t_tip[0], ref_y), (0,0,255), 3)

        cv2.putText(frame, f"Val: {int(round(val))}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Orient: {orient}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

# --- FastAPI Setup ---
tracker = HandTracker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    tracker.start()
    yield
    tracker.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status", response_model=TrackerStatus)
async def get_status():
    with tracker.lock:
        return TrackerStatus(
            percentage=int(round(tracker.percentage)),
            detected=tracker.is_detected,
            hand=tracker.hand_label,
            orientation=tracker.orientation
        )

@app.post("/config")
async def set_config(config: TrackerConfig):
    tracker.update_config(config.min_ratio, config.max_ratio)
    return {"message": "Updated", "config": config}

@app.get("/video_feed")
def video_feed(request: Request):
    return StreamingResponse(generate_frames(request), media_type="multipart/x-mixed-replace; boundary=frame")

async def generate_frames(request: Request):
    with tracker.lock:
        tracker.active_clients += 1
    try:
        while True:
            if await request.is_disconnected(): break
            frame_data = None
            with tracker.lock:
                frame_data = tracker.latest_frame_bytes
            if frame_data:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.03)
    finally:
        with tracker.lock:
            tracker.active_clients -= 1

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)