# AI Interviewer: Anti-Cheat & Proctoring System

An intelligent AI Interviewer platform equipped with a real-time, low-latency client-side and server-side proctoring engine. The system monitors candidate behaviors using facial landmark detection (TensorFlow.js & MediaPipe FaceMesh) in the browser, combined with server-side object detection (FastAPI & YOLOv8) to flag cheating behaviors such as cell phone usage, looking away, or multiple people in the frame.

---

## Key Features

### 🖥️ Client-Side Proctoring (Browser)
* **Face Missing / Left Frame Detection**: Monitors if the candidate leaves the frame or if their face is not visible.
* **Multiple People Detection**: Alerts when multiple faces or another person in the background is detected.
* **Looked Away from Screen**: Calculates head yaw and pitch angles to detect when the candidate looks away.
* **Eye Shifting / REM Detection**: Tracks pupils to alert on rapid eye movements or shifting gaze.
* **Face Partially Hidden**: Detects if the face is obscured (using advanced mouth symmetry, width collapse, and 3D Z-depth compression checking) or if the face is partially off-screen (using boundary drift history and YOLOv8 person-detection fallback).

### 🧠 Server-Side Proctoring (API)
* **Cell Phone Detection**: Periodically analyzes webcam screenshots using YOLOv8 (`yolov8n.pt`) to detect smartphones, tablets, or books.
* **In-Memory Session Store**: Tracks transcripts, violation timestamps, and base64 violation screenshots.
* **Score & Evaluation Report**: Generates a final score (starting at 90, deducting 5 points per violation) and provides structured behavioral feedback.

---

## Project Structure

```text
Anti-Cheat-System/
├── backend/                  # FastAPI Backend API
│   ├── main.py               # Fast API endpoints & server setup
│   ├── analyzer.py           # Mock AI Interviewer and scoring logic
│   ├── memory_store.py       # Session memory manager
│   ├── requirements.txt      # Python dependencies
│   └── yolov8n.pt            # Pre-trained YOLOv8 Nano model weights
└── ai-interview-frontend/    # React/Vite/TS Frontend
    ├── src/
    │   ├── pages/Interview/  # Core Interview pages and subcomponents
    │   │   ├── InterviewScreen.tsx # Facial tracking & detection loop
    │   │   └── ui/           # Custom components (Avatar, Report, Header)
    │   ├── App.tsx           # Application root
    │   └── index.css         # Styling system
```

---

## Getting Started

### 1. Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Set up a Python virtual environment and activate it:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI development server:
   ```bash
   python main.py
   ```
   *The backend will be running at:* `http://localhost:3000`

### 2. Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd ../ai-interview-frontend
   ```
2. Install the frontend dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
   *The application will be accessible at:* `http://localhost:5173` (or the port shown in your terminal).

---

## How It Works

1. **Face Mesh Estimation**: The browser uses the `@tensorflow-models/face-landmarks-detection` package loaded on startup.
2. **Real-time Assessment**: Bounding box size, coordinates, relative landmark ratios, and pupil locations are calculated dynamically inside the browser.
3. **Throttled Violation Calls**: When a violation condition is met, the screenshot is captured, compressed, and posted to the backend `/api/log-violation` endpoint. These events are throttled to a maximum of once every 10 seconds per type to save bandwidth.
4. **Final Reporting**: Clicking "End Interview" fetches the generated report from `/api/get-report` and presents it visually to the user, displaying each flagged event with its screenshot and timestamp.

---

## Advanced Detection Algorithms

To increase accuracy and reduce false positives/negatives, the proctoring engine implements several advanced geometric filters:

### 🎭 1. Occlusion Detection (Hand-over-Mouth/Face)
Even if FaceMesh maintains high overall confidence (`>0.90`) when a candidate covers their face, the system checks:
* **3D Z-Depth Compression**: Measures the Z-depth gap between the nose tip and the lips. When a hand covers the mouth, the landmarks project forward onto the hand, compressing the nose-to-lip depth gap to near zero.
* **Mouth Width Collapse**: Triggers if the horizontal lip width shrinks below 16% of the overall face width.
* **Horizontal Asymmetry**: Measures the skew ratio between the nose and the left/right mouth corners.

### 🖼️ 2. Boundary Drift & YOLO Fallback
When a candidate moves partially off-screen (e.g. only forehead or half face is visible), the FaceMesh detector completely fails (`faces.length === 0`). The system resolves this using:
* **Boundary Memory**: If the face disappears within 3 seconds of being near any of the four frame boundaries (within 80px), it is flagged as **Face Partially Hidden** instead of **Face Missing**.
* **YOLOv8 Person Detection Fallback**: The client-side YOLOv8 model runs a single-pass inference on the webcam stream. If FaceMesh fails but YOLOv8 still detects a **"Person" (Class ID 0)** in the frame (indicating only part of the head or body is visible), it is flagged as **Face Partially Hidden**. If no person is detected at all, it is classified as **Face Missing from Frame**.
