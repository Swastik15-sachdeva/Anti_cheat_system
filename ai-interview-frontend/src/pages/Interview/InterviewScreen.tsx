import { useState, useRef, useEffect } from "react";
import toast from "react-hot-toast";
import { X } from "lucide-react";
import { AiInterviewLayout, InterviewHeader, InterviewStage, TranscriptPanel, InterviewReport } from "./ui";
import type { InterviewReportData } from "./ui/InterviewReport";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as ort from "onnxruntime-web";

// Configure WASM paths for onnxruntime-web to load from the backend to bypass Vite dynamic import issues
ort.env.wasm.wasmPaths = "http://localhost:3000/static/wasm/";

interface Detection {
  box: [number, number, number, number]; // [y_min, x_min, y_max, x_max]
  score: number;
  classId: number;
}

// Draw video frame to a 640x640 canvas and return a Float32Array ONNX Tensor in CHW format
async function preprocess(video: HTMLVideoElement): Promise<ort.Tensor> {
  const size = 640;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get 2D context");

  ctx.drawImage(video, 0, 0, size, size);
  const imgData = ctx.getImageData(0, 0, size, size);
  const data = imgData.data;

  const numPixels = size * size;
  const inputBuffer = new Float32Array(3 * numPixels);

  // Normalize pixel values to [0, 1] and transpose from HWC to CHW
  for (let i = 0; i < numPixels; i++) {
    inputBuffer[i] = data[i * 4] / 255.0;                   // R
    inputBuffer[numPixels + i] = data[i * 4 + 1] / 255.0;   // G
    inputBuffer[2 * numPixels + i] = data[i * 4 + 2] / 255.0; // B
  }

  return new ort.Tensor("float32", inputBuffer, [1, 3, 640, 640]);
}

// Decodes raw YOLOv8 ONNX outputs [1, 4 + numClasses, 8400] using TensorFlow.js
function decodeYOLOv8(
  outputTensor: ort.Tensor,
  numClasses: number,
  targetClassId: number | null = null
): { tfBoxes: tf.Tensor2D; scores: tf.Tensor1D; classIds: tf.Tensor1D } {
  return tf.tidy(() => {
    const rawData = outputTensor.data as Float32Array;
    const rawOutput = tf.tensor3d(rawData, [1, 4 + numClasses, 8400]);

    // Transpose from [1, 4+numClasses, 8400] to [8400, 4+numClasses]
    const reshaped = rawOutput.squeeze([0]).transpose([1, 0]) as tf.Tensor2D;

    // Split box coordinates and class scores
    const boxes = reshaped.slice([0, 0], [8400, 4]); // [x_center, y_center, w, h]
    // Apply sigmoid activation to output logits to compute true confidence scores
    const classesScores = tf.sigmoid(
      reshaped.slice([0, 4], [8400, numClasses])
    ) as tf.Tensor2D;

    let maxScores: tf.Tensor1D;
    let classIds: tf.Tensor1D;

    if (numClasses === 1) {
      maxScores = classesScores.squeeze([1]) as tf.Tensor1D;
      classIds = tf.zeros([8400]) as tf.Tensor1D;
    } else {
      maxScores = classesScores.max(1) as tf.Tensor1D;
      classIds = classesScores.argMax(1) as tf.Tensor1D;
    }

    // Convert coordinates from [x_center, y_center, w, h] to [y_min, x_min, y_max, x_max]
    const [x_center, y_center, w, h] = tf.split(boxes, 4, 1);
    const halfW = tf.div(w, 2);
    const halfH = tf.div(h, 2);
    const x_min = tf.sub(x_center, halfW);
    const y_min = tf.sub(y_center, halfH);
    const x_max = tf.add(x_center, halfW);
    const y_max = tf.add(y_center, halfH);
    const tfBoxes = tf.concat([y_min, x_min, y_max, x_max], 1) as tf.Tensor2D;

    let scores = maxScores;
    if (targetClassId !== null) {
      const classMask = tf.equal(classIds, tf.scalar(targetClassId, "int32"));
      scores = tf.where(classMask, maxScores, tf.zerosLike(maxScores)) as tf.Tensor1D;
    }

    return {
      tfBoxes: tfBoxes.clone(),
      scores: scores.clone(),
      classIds: classIds.clone()
    };
  });
}

// Runs inference, decodes, and applies NMS using TensorFlow.js GPU
async function runYOLOInference(
  session: any,
  inputTensor: ort.Tensor,
  numClasses: number,
  scoreThreshold: number = 0.25,
  iouThreshold: number = 0.45,
  targetClassId: number | null = null
): Promise<Detection[]> {
  const outputs = await session.run({ images: inputTensor });
  const outputName = session.outputNames[0];
  const outputTensor = outputs[outputName];

  const { tfBoxes, scores, classIds } = decodeYOLOv8(outputTensor, numClasses, targetClassId);

  const nmsIndices = await tf.image.nonMaxSuppressionAsync(
    tfBoxes,
    scores,
    20, // max detections
    iouThreshold,
    scoreThreshold
  );

  const selectedBoxes = tfBoxes.gather(nmsIndices);
  const selectedScores = scores.gather(nmsIndices);
  const selectedClasses = classIds.gather(nmsIndices);

  const boxesArray = (await selectedBoxes.array()) as number[][];
  const scoresArray = (await selectedScores.array()) as number[];
  const classesArray = (await selectedClasses.array()) as number[];

  // Clean up tensors to avoid WebGL memory leaks
  tfBoxes.dispose();
  scores.dispose();
  classIds.dispose();
  nmsIndices.dispose();
  selectedBoxes.dispose();
  selectedScores.dispose();
  selectedClasses.dispose();

  const detections: Detection[] = [];
  for (let i = 0; i < boxesArray.length; i++) {
    detections.push({
      box: boxesArray[i] as [number, number, number, number],
      score: scoresArray[i],
      classId: classesArray[i]
    });
  }

  return detections;
}

// Checks if bounding box A overlaps with bounding box B
function checkOverlap(boxA: [number, number, number, number], boxB: [number, number, number, number]): boolean {
  const [yMinA, xMinA, yMaxA, xMaxA] = boxA;
  const [yMinB, xMinB, yMaxB, xMaxB] = boxB;

  return !(xMinA > xMaxB || xMaxA < xMinB || yMinA > yMaxB || yMaxA < yMinB);
}

// Checks if bounding box A overlaps significantly with bounding box B
function checkSignificantOverlap(
  boxA: [number, number, number, number],
  boxB: [number, number, number, number],
  minRatio = 0.15
): boolean {
  const [yMinA, xMinA, yMaxA, xMaxA] = boxA;
  const [yMinB, xMinB, yMaxB, xMaxB] = boxB;

  const yMinIntersect = Math.max(yMinA, yMinB);
  const xMinIntersect = Math.max(xMinA, xMinB);
  const yMaxIntersect = Math.min(yMaxA, yMaxB);
  const xMaxIntersect = Math.min(xMaxA, xMaxB);

  if (yMinIntersect >= yMaxIntersect || xMinIntersect >= xMaxIntersect) {
    return false;
  }

  const intersectArea = (yMaxIntersect - yMinIntersect) * (xMaxIntersect - xMinIntersect);
  const areaA = (yMaxA - yMinA) * (xMaxA - xMinA) || 1;
  const areaB = (yMaxB - yMinB) * (xMaxB - xMinB) || 1;

  // Overlap is significant if intersection area is at least minRatio of either box
  return (intersectArea / areaA > minRatio) || (intersectArea / areaB > minRatio);
}


const MOCK_MESSAGES = [
  { role: "system", text: "Interview started", timestamp: new Date() },
  { role: "ai", text: "Hello! Welcome to your interview for the Frontend Engineer position. Are you ready to begin?", timestamp: new Date() }
];

export default function InterviewScreen() {
  const [isMuted] = useState(false);
  const [isAiSpeaking] = useState(false);
  const [isUserSpeaking] = useState(false);
  const [isVideoOn] = useState(true);
  const [timeLeft] = useState(300); // 5 minutes
  const [isInterviewComplete, setIsInterviewComplete] = useState(false);
  const [reportData, setReportData] = useState<InterviewReportData | null>(null);
  const [isProcessingEnd, setIsProcessingEnd] = useState(false);
  
  // Real-time proctoring states
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [detector, setDetector] = useState<any>(null);
  const [yoloSession, setYoloSession] = useState<any>(null);
  const [handSession, setHandSession] = useState<any>(null);
  const [tabSwitchCount, setTabSwitchCount] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const [faceWarning, setFaceWarning] = useState<string | null>(null);
  const [gazeWarning, setGazeWarning] = useState<string | null>(null);
  const [secondPersonWarning, setSecondPersonWarning] = useState<string | null>(null);
  const [blurWarning, setBlurWarning] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  // Ref tracking for throttling, history, and timestamps
  const lastLoggedRef = useRef<{ [key: string]: number }>({});
  const faceMissingStartRef = useRef<number | null>(null);
  const gazeHistoryRef = useRef<number[]>([]);
  const lastActiveFaceRef = useRef<{ bbox: any; timestamp: number } | null>(null);

  // Baselines for head pose self-calibration
  const baselineYawRef = useRef<number | null>(null);
  const baselinePitchRef = useRef<number | null>(null);
  const baselineYaw3DRef = useRef<number | null>(null);
  const baselinePitch3DRef = useRef<number | null>(null);

  // Proctoring State Machine and Buffers (IDLE -> SUSPECTED -> CONFIRMED -> LOGGED)
  const stateRef = useRef<{ [key: string]: "IDLE" | "SUSPECTED" | "CONFIRMED" | "LOGGED" }>({});
  const suspectStartRef = useRef<{ [key: string]: number }>({});
  const bufferRef = useRef<{ [key: string]: boolean[] }>({});

  const updateProctorState = (type: string, isCurrentlyViolating: boolean, durationMs: number) => {
    if (!bufferRef.current[type]) {
      bufferRef.current[type] = [];
    }
    if (!stateRef.current[type]) {
      stateRef.current[type] = "IDLE";
    }

    const history = bufferRef.current[type];
    history.push(isCurrentlyViolating);
    if (history.length > 5) {
      history.shift();
    }

    // Majority voting: At least 4 of the last 5 frames must indicate a violation
    const trueCount = history.filter(v => v).length;
    const filteredViolating = history.length >= 4 ? trueCount >= 4 : isCurrentlyViolating;

    const currentState = stateRef.current[type];
    const now = Date.now();

    if (filteredViolating) {
      if (currentState === "IDLE") {
        stateRef.current[type] = "SUSPECTED";
        suspectStartRef.current[type] = now;
      } else if (currentState === "SUSPECTED") {
        const suspectTime = suspectStartRef.current[type] || now;
        if (now - suspectTime >= durationMs) {
          stateRef.current[type] = "CONFIRMED";
        }
      } else if (currentState === "CONFIRMED" || currentState === "LOGGED") {
        // Cooldown throttling check (10 seconds)
        const lastLog = lastLoggedRef.current[type] || 0;
        const isCoolingDown = now - lastLog < 10000;

        if (!isCoolingDown) {
          // Re-log the violation now that cooldown has passed
          stateRef.current[type] = "LOGGED";
          triggerViolation(type);
        }
        // While cooling down, stay in LOGGED so the UI banner remains visible
        // and we re-try every frame until the cooldown expires.
      }
    } else {
      stateRef.current[type] = "IDLE";
      suspectStartRef.current[type] = 0;
    }
  };

  const formatTime = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // 1. Initialize TFJS and load Models
  useEffect(() => {
    let active = true;
    async function loadModels() {
      try {
        console.log("Initializing TensorFlow...");
        await tf.ready();
        await tf.setBackend("webgl");
        
        console.log("Loading Face Mesh detector...");
        const faceMeshModel = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const faceDetector = await faceLandmarksDetection.createDetector(faceMeshModel, {
          runtime: "tfjs",
          refineLandmarks: true
        });

        console.log("Loading YOLO ONNX models in browser...");
        const ySession = await ort.InferenceSession.create("http://localhost:3000/static/models/yolov8n.onnx", {
          executionProviders: ["wasm"]
        });
        const hSession = await ort.InferenceSession.create("http://localhost:3000/static/models/hand_yolov8n.onnx", {
          executionProviders: ["wasm"]
        });

        if (active) {
          setDetector(faceDetector);
          setYoloSession(ySession);
          setHandSession(hSession);
          setModelsLoaded(true);
          console.log("Proctoring models loaded successfully.");
        }
      } catch (err) {
        console.error("Error loading proctoring models:", err);
        toast.error("Proctoring system failed to load. Running in offline mode.");
      }
    }
    loadModels();
    return () => {
      active = false;
    };
  }, []);

  // 2. Camera Setup
  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
    setupCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // 3. Helper to capture frame screenshot
  const captureScreenshot = (): string => {
    if (!videoRef.current) return "";
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.6); // compressed JPEG base64
  };

  // 4. Violation triggering and throttling (max once every 10s per violation type)
  const triggerViolation = (type: string) => {
    const now = Date.now();
    const lastTime = lastLoggedRef.current[type] || 0;
    if (now - lastTime < 10000) {
      return; // Throttled
    }
    lastLoggedRef.current[type] = now;
    const screenshot = captureScreenshot();

    // Log to backend FastAPI
    fetch("http://localhost:3000/api/log-violation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: "mock-session-id",
        type: type,
        timestamp: new Date().toISOString(),
        screenshot_base64: screenshot
      })
    })
    .then(res => {
      if (!res.ok) console.error("Failed to log violation:", type);
    })
    .catch(console.error);

    // Show proctoring toast warning
    toast(
      (t) => (
        <div className="flex items-center justify-between gap-3 min-w-[200px]">
          <span className="text-sm font-medium">Violation: {type}</span>
          <button
            onClick={() => toast.dismiss(t.id)}
            className="p-1 cursor-pointer rounded-full hover:bg-black/10 transition-colors text-gray-500 hover:text-gray-700"
            title="Close"
          >
            <X size={16} strokeWidth={3} />
          </button>
        </div>
      ),
      {
        duration: 4000,
        position: "top-center",
        className: "toast-info-border",
      }
    );
  };

  // 5. Main Proctoring Detection Loop
  useEffect(() => {
    if (!modelsLoaded || !detector || isInterviewComplete || !yoloSession || !handSession) {
      return;
    }

    const intervalId = setInterval(async () => {
      // Avoid running tracking and logging false missing violations when the tab is not focused
      if (document.hidden) {
        return;
      }

      const video = videoRef.current;
      if (!video || video.readyState !== 4 || video.paused || video.ended) {
        return;
      }

      try {
        // Run face landmark detector
        const faces = await detector.estimateFaces(video, { flipHorizontal: false });

        let phoneDetected = false;
        let handOverlapsFace = false;
        let phoneOverlapsFace = false;

        let handDetections: Detection[] = [];
        let phoneDetections: Detection[] = [];

        try {
          const inputTensor = await preprocess(video);

          try {
            // 1. Detect Person and Phone (Class 0 and 67 in COCO dataset) in one pass
            // The YOLO scores are now sigmoid-scaled, representing proper probabilities.
            const yoloDetections = await runYOLOInference(yoloSession, inputTensor, 80, 0.25, 0.45, null);
            for (const det of yoloDetections) {
              if (det.classId === 67 && det.score > 0.60) {
                phoneDetections.push(det);
              }
            }

            // 2. Detect Hands (Class ID 0 in custom hand model)
            handDetections = await runYOLOInference(handSession, inputTensor, 1, 0.25, 0.45, 0);
          } finally {
            // Always dispose the ONNX tensor to prevent GPU/WASM memory leaks
            inputTensor.dispose();
          }

          // 3. Check Overlap if face is found
          if (faces && faces.length > 0) {
            const mainFace = faces[0];
            const keypoints = mainFace.keypoints;
            const videoWidth = video.videoWidth || 640;
            const videoHeight = video.videoHeight || 480;

            let face_x_min = Infinity;
            let face_y_min = Infinity;
            let face_x_max = -Infinity;
            let face_y_max = -Infinity;

            for (const kp of keypoints) {
              const scaledX = kp.x * (640 / videoWidth);
              const scaledY = kp.y * (640 / videoHeight);
              if (scaledX < face_x_min) face_x_min = scaledX;
              if (scaledY < face_y_min) face_y_min = scaledY;
              if (scaledX > face_x_max) face_x_max = scaledX;
              if (scaledY > face_y_max) face_y_max = scaledY;
            }
            // Normalize to [0, 1] space to match YOLO box coordinates which are
            // also in [0, 1] (YOLO outputs are in 0–640 pixel space, divided by 640).
            const faceBox: [number, number, number, number] = [
              face_y_min / 640, face_x_min / 640,
              face_y_max / 640, face_x_max / 640
            ];

            // Check if hand overlaps with the face
            for (const det of handDetections) {
              if (checkOverlap(det.box, faceBox)) {
                handOverlapsFace = true;
                break;
              }
            }

            // Check if phone overlaps significantly with the face (confidence > 0.60, area overlap >= 15%)
            for (const det of phoneDetections) {
              if (checkSignificantOverlap(det.box, faceBox, 0.15)) {
                phoneOverlapsFace = true;
                phoneDetected = true;
                break;
              }
            }
          }
        } catch (mlErr) {
          console.error("Error in browser edge ML inference:", mlErr);
        }
        // Note: inputTensor is disposed inside the inner try/finally above.

        // Determine face state checks
        if (!faces || faces.length === 0) {
          if (faceMissingStartRef.current === null) {
            faceMissingStartRef.current = Date.now();
          }

          // Check boundary memory if face disappeared
          const now = Date.now();
          let wasPartiallyOffScreen = false;
          const videoWidth = video.videoWidth || 640;
          const videoHeight = video.videoHeight || 480;
          
          if (lastActiveFaceRef.current && (now - lastActiveFaceRef.current.timestamp < 3000)) {
            const lastBbox = lastActiveFaceRef.current.bbox;
            const edgeMargin = 80;
            wasPartiallyOffScreen = lastBbox.xMin < edgeMargin || 
                                   lastBbox.yMin < edgeMargin || 
                                   (lastBbox.xMax > videoWidth - edgeMargin) || 
                                   (lastBbox.yMax > videoHeight - edgeMargin);
          }

          // When no face is detected, only classify as "Partially Hidden" if the face
          // was seen recently near the video edge (suggesting it slid off-screen).
          // Hand/phone detections alone are NOT reliable here — the YOLO model can
          // mis-detect fabric, lamps, or other objects when there is no face present.
          const isPartiallyHiddenRaw = wasPartiallyOffScreen;
          const isFaceMissingRaw = !isPartiallyHiddenRaw;

          updateProctorState("Face Partially Hidden", isPartiallyHiddenRaw, 2000);
          updateProctorState("Face Missing from Frame", isFaceMissingRaw, 2000);
          updateProctorState("Multiple Faces Detected", false, 1000);
          updateProctorState("Cell Phone Detected", false, 1500);
          updateProctorState("Looked Away from Screen", false, 3000);
          updateProctorState("Eye Shifting / Rapid Eye Movement", false, 2000);
        } else {
          // Update last active face
          lastActiveFaceRef.current = {
            bbox: faces[0].box,
            timestamp: Date.now()
          };
          faceMissingStartRef.current = null;

          // 1. Multiple faces check (1.0 second persistence)
          const isMultipleFacesRaw = faces.length > 1;
          updateProctorState("Multiple Faces Detected", isMultipleFacesRaw, 1000);

          const mainFace = faces[0];
          const bbox = mainFace.box;
          const videoWidth = video.videoWidth || 640;
          const videoHeight = video.videoHeight || 480;

          // Check partially hidden indicators — use a tight margin so a face that is
          // merely large or close to the camera edge doesn't false-positive.
          const margin = 20;
          const isOffScreen = bbox.xMin < margin || 
                             bbox.yMin < margin || 
                             (bbox.xMax > videoWidth - margin) || 
                             (bbox.yMax > videoHeight - margin);
          // NOTE: MediaPipe confidence scores fluctuate with lighting and viewing angle
          // and are NOT a reliable indicator that the face is occluded, so we no longer
          // include them in the partially-hidden score.

          // Advanced face geometry occlusion check (natural expressions, speaking, smiling should be ignored)
          let isFaceOccluded = false;
          const keypoints = mainFace.keypoints;
          if (keypoints && keypoints.length >= 153) {
            const kp = (idx: number) => keypoints[idx] || { x: 0, y: 0, z: 0 };
            
            const nose = kp(4);
            const leftCheek = kp(234);
            const rightCheek = kp(454);
            
            const mouthLeft = kp(61);
            const mouthRight = kp(291);

            // Compute ratios using proper baseline width in pixel space
            const mouthWidth = Math.sqrt(
              Math.pow(mouthRight.x - mouthLeft.x, 2) + 
              Math.pow(mouthRight.y - mouthLeft.y, 2)
            );
            const pxFaceWidth = Math.sqrt(
              Math.pow(rightCheek.x - leftCheek.x, 2) + 
              Math.pow(rightCheek.y - leftCheek.y, 2)
            ) || 1;
            const mouthWidthRatio = mouthWidth / pxFaceWidth;

            const d_nose_mouth_left = Math.sqrt(Math.pow(mouthLeft.x - nose.x, 2) + Math.pow(mouthLeft.y - nose.y, 2));
            const d_nose_mouth_right = Math.sqrt(Math.pow(mouthRight.x - nose.x, 2) + Math.pow(mouthRight.y - nose.y, 2));
            // Increased symmetry ratio to 2.0 to ignore speaking asymmetry and natural head tilts
            const mouthSymmetryRatio = Math.max(d_nose_mouth_left, d_nose_mouth_right) / (Math.min(d_nose_mouth_left, d_nose_mouth_right) || 1);

            // Occlusion triggers: mouth width collapse or extreme asymmetry only.
            // Z-depth ratio is excluded — MediaPipe's relative Z near nose/mouth is too
            // noisy and nearly zero for a forward-facing face, causing false positives.
            if (mouthWidthRatio < 0.14 || mouthSymmetryRatio > 2.5) {
              isFaceOccluded = true;
            }
          }

          // Weighted scoring system.
          // Threshold is STRICTLY > 2.0 so two weak signals (off-screen + occluded)
          // alone cannot trigger it — a phone or hand overlap is required.
          let partiallyHiddenScore = 0;
          if (phoneOverlapsFace) partiallyHiddenScore += 2.5; // phone over face is definitive
          if (handOverlapsFace)  partiallyHiddenScore += 2.1; // hand over face alone is sufficient
          if (isOffScreen)       partiallyHiddenScore += 0.5;
          if (isFaceOccluded)    partiallyHiddenScore += 0.5;

          const isPartiallyHiddenRaw = partiallyHiddenScore > 2.0;
          updateProctorState("Face Partially Hidden", isPartiallyHiddenRaw, 2000);
          updateProctorState("Face Missing from Frame", false, 2000);

          // 2. Cell Phone checks (confidence > 0.60, area overlap >= 15%, 1.5 second persistence)
          updateProctorState("Cell Phone Detected", phoneDetected, 1500);

          // 3. Gaze and Head Pose checks (3.0 seconds persistence)
          let isLookingAway = false;
          if (keypoints && keypoints.length >= 153) {
            const kp = (idx: number) => keypoints[idx] || { x: 0, y: 0, z: 0 };
            
            // Yaw (looking left/right)
            const leftCheek = kp(234);
            const rightCheek = kp(454);
            const nose = kp(4);
            const d_left = Math.abs(nose.x - leftCheek.x);
            const d_right = Math.abs(rightCheek.x - nose.x);
            const yawRatio = d_left / (d_left + d_right || 1);

            // Z depth calculations (independent of pixel coordinate scaling)
            const yaw3D = leftCheek.z - rightCheek.z;

            // Pitch (looking up/down)
            const forehead = kp(10);
            const chin = kp(152);
            const d_up = Math.abs(nose.y - forehead.y);
            const d_down = Math.abs(chin.y - nose.y);
            const pitchRatio = d_up / (d_up + d_down || 1);

            // Z depth calculations (independent of pixel coordinate scaling)
            const pitch3D = forehead.z - chin.z;

            // Self-Calibration
            if (baselineYawRef.current === null) baselineYawRef.current = yawRatio;
            if (baselinePitchRef.current === null) baselinePitchRef.current = pitchRatio;
            if (baselineYaw3DRef.current === null) baselineYaw3DRef.current = yaw3D;
            if (baselinePitch3DRef.current === null) baselinePitch3DRef.current = pitch3D;

            const yawDev = Math.abs(yawRatio - baselineYawRef.current);
            const pitchDev = Math.abs(pitchRatio - baselinePitchRef.current);
            const yaw3DDev = Math.abs(yaw3D - baselineYaw3DRef.current);
            const pitch3DDev = Math.abs(pitch3D - baselinePitch3DRef.current);

            isLookingAway = yawDev > 0.06 || pitchDev > 0.07 || yaw3DDev > 0.16 || pitch3DDev > 0.16;
          }
          updateProctorState("Looked Away from Screen", isLookingAway, 3000);

          // 4. Eye Shifting / Pupil tracking (2.0 seconds persistence)
          let isEyeShiftingRaw = false;
          if (keypoints && keypoints.length >= 153) {
            const kp = (idx: number) => keypoints[idx] || { x: 0, y: 0, z: 0 };
            let iris_left = keypoints.find((k: any) => k.name === "leftIris" || k.name === "leftEyeIris");
            let iris_right = keypoints.find((k: any) => k.name === "rightIris" || k.name === "rightEyeIris");
            if (!iris_left) iris_left = keypoints[468];
            if (!iris_right) iris_right = keypoints[473];

            const isIrisFallbackActive = !iris_left || !iris_right;

            if (isIrisFallbackActive) {
              const top_left = kp(159);
              const bottom_left = kp(145);
              const top_right = kp(386);
              const bottom_right = kp(374);
              iris_left = { x: (top_left.x + bottom_left.x) / 2, y: (top_left.y + bottom_left.y) / 2, z: 0 };
              iris_right = { x: (top_right.x + bottom_right.x) / 2, y: (top_right.y + bottom_right.y) / 2, z: 0 };
            }

            const c_outer_left = kp(33);
            const c_inner_left = kp(133);
            const left_eye_width = Math.abs(c_inner_left.x - c_outer_left.x) || 1;
            const gazeRatioLeft = Math.abs(iris_left.x - Math.min(c_outer_left.x, c_inner_left.x)) / left_eye_width;

            const c_inner_right = kp(362);
            const c_outer_right = kp(263);
            const right_eye_width = Math.abs(c_outer_right.x - c_inner_right.x) || 1;
            const gazeRatioRight = Math.abs(iris_right.x - Math.min(c_inner_right.x, c_outer_right.x)) / right_eye_width;

            const avgGaze = (gazeRatioLeft + gazeRatioRight) / 2;
            gazeHistoryRef.current.push(avgGaze);
            if (gazeHistoryRef.current.length > 6) {
              gazeHistoryRef.current.shift();
            }

            if (!isIrisFallbackActive) {
              // ±10% band around center (0.5). Tighter bands cause false positives
              // when the candidate reads a question or scans their screen.
              const isEyeShifting = avgGaze < 0.40 || avgGaze > 0.60;
              let shiftsCount = 0;
              const history = gazeHistoryRef.current;
              for (let i = 1; i < history.length; i++) {
                // Require a larger per-frame jump to count as rapid eye movement
                if (Math.abs(history[i] - history[i-1]) >= 0.08) {
                  shiftsCount++;
                }
              }
              const isREM = shiftsCount >= 3;
              isEyeShiftingRaw = isEyeShifting || isREM;
            }
          }
          updateProctorState("Eye Shifting / Rapid Eye Movement", isEyeShiftingRaw, 2000);
        }

        // Map state machine warning levels to warning triggers
        const getWarningText = (type: string, activeText: string): string | null => {
          const s = stateRef.current[type];
          // Only surface a warning once the violation has been CONFIRMED (held for the
          // full persistence window), not on the first SUSPECTED frame.  This prevents
          // brief or spurious detections from flashing a banner on screen.
          return (s === "CONFIRMED" || s === "LOGGED") ? activeText : null;
        };

        const currentFaceWarning = getWarningText("Face Missing from Frame", "Face Missing from Frame") || 
                                   getWarningText("Face Partially Hidden", "Face Partially Hidden");
        const currentGazeWarning = getWarningText("Looked Away from Screen", "Looked Away from Screen") || 
                                   getWarningText("Eye Shifting / Rapid Eye Movement", "Eye Shifting / Rapid Eye Movement");
        const currentSecondPersonWarning = getWarningText("Multiple Faces Detected", "Multiple Faces Detected");
        const currentBlurWarning = getWarningText("Cell Phone Detected", "Cell Phone Detected");

        setFaceWarning(currentFaceWarning);
        setGazeWarning(currentGazeWarning);
        setSecondPersonWarning(currentSecondPersonWarning);
        setBlurWarning(currentBlurWarning);

      } catch (err) {
        console.error("Error in detection loop:", err);
      }
    }, 800);

    return () => clearInterval(intervalId);
  }, [modelsLoaded, detector, yoloSession, handSession, isInterviewComplete]);

  // 6. Browser event listeners for tab switching and fullscreen
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        setTabSwitchCount(prev => prev + 1);
        triggerViolation("Tab Switched / Left Window");
      }
    };

    const handleBlur = () => {
      // Log the violation but do NOT increment tabSwitchCount here — visibilitychange
      // also fires on alt-tab and would double-count it. Only visibilitychange counts.
      if (!document.hidden) {
        triggerViolation("Tab Switched / Left Window");
      }
    };

    const handleFullscreenChange = () => {
      const isFull = !!(
        document.fullscreenElement ||
        (document as any).webkitFullscreenElement ||
        (document as any).mozFullScreenElement ||
        (document as any).msFullscreenElement
      );
      setIsFullscreen(isFull);
      if (!isFull && !isInterviewComplete) {
        triggerViolation("Exited Fullscreen");
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("blur", handleBlur);
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    document.addEventListener("mozfullscreenchange", handleFullscreenChange);
    document.addEventListener("MSFullscreenChange", handleFullscreenChange);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("blur", handleBlur);
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.removeEventListener("mozfullscreenchange", handleFullscreenChange);
      document.removeEventListener("MSFullscreenChange", handleFullscreenChange);
    };
  }, [isInterviewComplete]);

  const enterFullscreen = () => {
    const element = document.documentElement as any;
    if (element.requestFullscreen) {
      element.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch((err: any) => {
        console.error("Error attempting to enable fullscreen:", err);
      });
    } else if (element.webkitRequestFullscreen) { /* Safari */
      element.webkitRequestFullscreen();
      setIsFullscreen(true);
    } else if (element.mozRequestFullScreen) { /* Firefox */
      element.mozRequestFullScreen();
      setIsFullscreen(true);
    } else if (element.msRequestFullscreen) { /* IE11 */
      element.msRequestFullscreen();
      setIsFullscreen(true);
    }
  };

  if (isInterviewComplete && reportData) {
    return (
      <AiInterviewLayout>
        <InterviewReport 
          reportData={reportData} 
          onBack={() => setIsInterviewComplete(false)} 
        />
      </AiInterviewLayout>
    );
  }

  return (
    <AiInterviewLayout>
      <InterviewHeader
        interviewData={{ job: { title: "Frontend Engineer" } }}
        isMuted={isMuted}
        isUserSpeaking={isUserSpeaking}
        isAiSpeaking={isAiSpeaking}
        timeLeft={timeLeft}
        formatTime={formatTime}
        sessionId="mock-session-id"
        interviewComplete={false}
      />
      


      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden min-h-0 bg-[#fafafa]">
        <InterviewStage
          isUserSpeaking={isUserSpeaking}
          isMuted={isMuted}
          isVideoOn={isVideoOn}
          videoRef={videoRef}
          faceWarning={faceWarning}
          blurWarning={blurWarning}
          gazeWarning={gazeWarning}
          secondPersonWarning={secondPersonWarning}
          showTour={false}
          isLoading={!modelsLoaded}
          isInitializing={!modelsLoaded}
          isAiSpeaking={isAiSpeaking}
          isProcessing={isProcessingEnd}
          speechFailed={false}
          audioAnalyser={null}
          handleEndInterview={async () => {
            setIsProcessingEnd(true);
            try {
              const res = await fetch("http://localhost:3000/api/get-report?session_id=mock-session-id");
              if (!res.ok) throw new Error("Failed to fetch report");
              const data = await res.json();
              setReportData(data);
              setIsInterviewComplete(true);
            } catch (error) {
              console.error("Backend fetch failed, using fallback mock data:", error);
              toast.error("Backend offline. Showing sample report.");
              
              // Fallback Mock Data
              setReportData({
                overallScore: 85,
                feedback: "This is a sample generated report because the backend is currently offline. The candidate demonstrated strong skills but had a few minor hesitations.",
                totalViolations: 1,
                violations: [
                  {
                    type: "Looked Away",
                    timestamp: new Date().toISOString(),
                    screenshot_base64: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                  }
                ]
              });
              setIsInterviewComplete(true);
            } finally {
              setIsProcessingEnd(false);
            }
          }}
          simliMode={false}
          simliVideoRef={{ current: null }}
          simliAudioRef={{ current: null }}
          isSimliConnected={false}
          interviewComplete={false}
        />
        <TranscriptPanel
          isMuted={isMuted}
          isAiSpeaking={isAiSpeaking}
          isProcessing={false}
          speechFailed={false}
          handleRetryMic={() => {}}
          messages={MOCK_MESSAGES}
          chatEndRef={chatEndRef}
          isModelLoaded={true}
          tabSwitchCount={tabSwitchCount}
          isFullscreen={isFullscreen}
          sessionId="mock-session-id"
          interviewComplete={isInterviewComplete}
          isEnding={isProcessingEnd}
          enterFullscreen={enterFullscreen}
        />
      </div>
    </AiInterviewLayout>
  );
}
