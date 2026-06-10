# AI Interviewer - React Frontend

This directory contains the React + TypeScript + Vite frontend of the **AI Interviewer Proctoring System**.

The frontend runs real-time client-side proctoring in the browser using TensorFlow.js and MediaPipe FaceMesh to track candidate behaviors (yaw, pitch, eye-shifting, etc.) with extremely low latency.

---

## Technical Features

* **Real-time Camera Stream Integration**: Directly requests webcam permissions and maps the input stream to a canvas for proctoring.
* **In-Browser Face Mesh Estimation**: Runs `@tensorflow/tfjs` with WebGL acceleration to estimate facial keypoints and tracking boundaries.
* **Gaze & Pose Analytics**: Evaluates eye shift patterns (REM) and head position thresholds.
* **Server-Side Verification Sync**: Periodically captures frame screenshots and POSTs them to the backend object detection service for device/materials check.

---

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Run Development Server**:
   ```bash
   npm run dev
   ```

3. **Production Build**:
   ```bash
   npm run build
   ```

## Configuration & Environment

* **API Base URL**: Configured in [InterviewScreen.tsx](file:///c:/Users/sachd/OneDrive/Desktop/internship/Anti-Cheat-System-/ai-interview-frontend/src/pages/Interview/InterviewScreen.tsx) to query the local FastAPI instance running at `http://localhost:3000`.
* **Proctoring Models Configuration**: Uses `tfjs` backend set to `webgl` for fast matrix computations, falling back to CPU if WebGL is unavailable.

For details on the proctoring algorithms and backend instructions, please refer to the [Root README](file:///c:/Users/sachd/OneDrive/Desktop/internship/Anti-Cheat-System-/README.md).
