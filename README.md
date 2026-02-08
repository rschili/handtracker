# Hand Tracking Thumbs Up/Down Detector

This was an exercise trying how far I could get only using agent coding. It took many iterations using Sonnet 4.5 and gemini 3 pro until it worked reliably. The code is completely AI crafted, I did not review it.

A real-time hand tracking application that detects and measures "thumbs up" and "thumbs down" gestures from a webcam feed. The tracker provides a percentage value (0-100%) indicating how far the thumb is extended, along with the orientation (up/down).

## Demo

![Image](https://github.com/user-attachments/assets/f5262480-7765-458d-ab2f-99fa2a0c0d17)

## Features

- **Real-time hand tracking** using MediaPipe's hand landmarker model
- **Thumbs up/down detection** with percentage measurement (0-100%)
- **Orientation detection** (Up/Down)
- **Dominant hand selection** - automatically tracks the hand closest to the center
- **FastAPI web server** with REST API endpoints
- **Live video feed** with visual debugging overlay
- **Configurable sensitivity** thresholds via API
- **Smooth value transitions** for stable readings

## Requirements

- Python 3.7+
- Webcam/camera access
- Dependencies:
  - OpenCV (`cv2`)
  - MediaPipe
  - FastAPI
  - Uvicorn

## Installation

1. **Clone the repository**
   ```bash
   cd tracker
   ```

2. **Install dependencies**
   ```bash
   pip install opencv-python mediapipe fastapi uvicorn
   ```

3. **Download the MediaPipe hand landmarker model**
   ```bash
   wget -O hand_landmarker.task -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

## Usage

### Running the Tracker

Start the application with:

```bash
python tracker.py
```

The server will start on `http://0.0.0.0:8000` and begin processing the webcam feed.

### API Endpoints

#### Get Current Status
```bash
curl http://localhost:8000/status
```

Returns the current tracking status:
```json
{
  "percentage": 75,
  "detected": true,
  "hand": "Right",
  "orientation": "Up"
}
```

#### Update Configuration
```bash
POST http://localhost:8000/config
Content-Type: application/json

{
  "min_ratio": 0.28,
  "max_ratio": 0.95
}
```

Adjusts the sensitivity thresholds for gesture detection.

#### Video Feed
```
GET http://localhost:8000/video_feed
```

Returns a live MJPEG stream with visual debugging overlays showing:
- Hand skeleton (green lines)
- Reference line (blue) - zero point for measurement
- Palm width line (yellow) - normalization reference
- Thumb elevation line (red) - current thumb position
- Percentage value and orientation text

### How It Works

The tracker uses MediaPipe's hand landmarker to detect 21 hand landmarks in real-time. The thumbs up/down detection algorithm:

1. **Detects orientation** by comparing the vertical positions of index MCP and pinky MCP landmarks
2. **Establishes a zero reference** at the midpoint between index MCP and PIP joints
3. **Measures thumb elevation** from the reference point to the thumb tip
4. **Normalizes** the elevation by dividing by the palm width (distance between index MCP and pinky MCP)
5. **Maps the ratio** to a 0-100% scale based on configurable min/max thresholds
6. **Smooths the output** using exponential moving average for stable readings

The default thresholds (0.28 - 0.95 ratio) are tuned for typical hand sizes and gesture ranges.

## Configuration

Key parameters in the code:

- `MODEL_PATH`: Path to the hand landmarker model file
- `CAMERA_INDEX`: Camera device index (default: 0)
- `min_ratio`: Minimum thumb elevation ratio for 0% (default: 0.28)
- `max_ratio`: Maximum thumb elevation ratio for 100% (default: 0.95)

## Performance Optimization

- Video encoding only occurs when clients are actively watching the feed
- Adjustable camera resolution (commented out by default)
- Efficient landmark detection with MediaPipe's VIDEO mode
- Smooth value filtering to reduce jitter

## License

MIT

## Credits

Built with [MediaPipe](https://mediapipe.dev/) hand tracking technology.
