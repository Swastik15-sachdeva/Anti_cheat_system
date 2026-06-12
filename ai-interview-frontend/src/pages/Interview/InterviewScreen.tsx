import { useState, useRef, useEffect } from "react";
import toast from "react-hot-toast";
import { X } from "lucide-react";
import { AiInterviewLayout, InterviewHeader, InterviewStage, TranscriptPanel, InterviewReport } from "./ui";
import type { InterviewReportData } from "./ui/InterviewReport";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";
import * as ort from "onnxruntime-web";

// Configure WASM paths for onnxruntime-web to load from CDN (avoids Vite transformation issues in development)
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

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
    const classesScores = reshaped.slice([0, 4], [8400, numClasses]);

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
        const ySession = await ort.InferenceSession.create("/models/yolov8n.onnx", {
          executionProviders: ["wasm"]
        });
        const hSession = await ort.InferenceSession.create("/models/hand_yolov8n.onnx", {
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
      const video = videoRef.current;
      if (!video || video.readyState !== 4 || video.paused || video.ended) {
        return;
      }

      try {
        let currentFaceWarning: string | null = null;
        let currentGazeWarning: string | null = null;
        let currentSecondPersonWarning: string | null = null;
        let currentBlurWarning: string | null = null;

        // Run face landmark detector
        const faces = await detector.estimateFaces(video, { flipHorizontal: false });

        let phoneDetected = false;
        let handOverlapsFace = false;
        let phoneOverlapsFace = false;
        let personDetected = false;

        try {
          const inputTensor = await preprocess(video);
          
          // 1. Detect Person and Phone (Class 0 and 67 in COCO dataset) in one pass
          const yoloDetections = await runYOLOInference(yoloSession, inputTensor, 80, 0.25, 0.45, null);
          const phoneDetections = [];
          for (const det of yoloDetections) {
            if (det.classId === 0) {
              personDetected = true;
            } else if (det.classId === 67) {
              phoneDetections.push(det);
              phoneDetected = true;
            }
          }

          // 2. Detect Hands (Class ID 0 in custom hand model)
          const handDetections = await runYOLOInference(handSession, inputTensor, 1, 0.25, 0.45, 0);

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
            const faceBox: [number, number, number, number] = [face_y_min, face_x_min, face_y_max, face_x_max];

            // Check if hand overlaps with the face
            for (const det of handDetections) {
              if (checkOverlap(det.box, faceBox)) {
                handOverlapsFace = true;
                break;
              }
            }

            // Check if phone overlaps with the face
            for (const det of phoneDetections) {
              if (checkOverlap(det.box, faceBox)) {
                phoneOverlapsFace = true;
                break;
              }
            }
          }
        } catch (mlErr) {
          console.error("Error in browser edge ML inference:", mlErr);
        }

        if (!faces || faces.length === 0) {
          if (faceMissingStartRef.current === null) {
            faceMissingStartRef.current = Date.now();
          }
          const missingSec = (Date.now() - faceMissingStartRef.current) / 1000;
          if (missingSec >= 1.5) {
            // Check if the face was recently near the edge before disappearing
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

            // Trigger "Face Partially Hidden" if:
            // - The face was recently near the edge of the frame
            // - OR a person is still detected in the frame by YOLO (e.g. only forehead is visible, or face covered)
            if (wasPartiallyOffScreen || personDetected) {
              currentFaceWarning = "Face Partially Hidden";
              triggerViolation("Face Partially Hidden");
            } else {
              currentFaceWarning = "Face Missing from Frame";
              triggerViolation("Face Missing from Frame");
            }
          }
        } else {
          // Update last active face
          lastActiveFaceRef.current = {
            bbox: faces[0].box,
            timestamp: Date.now()
          };
          // Face found, reset missing start
          faceMissingStartRef.current = null;

          // Multiple faces check
          if (faces.length >= 2) {
            currentSecondPersonWarning = "Multiple Faces Detected";
            triggerViolation("Multiple Faces Detected");
          }

          const mainFace = faces[0];
          const bbox = mainFace.box;
          const videoWidth = video.videoWidth || 640;
          const videoHeight = video.videoHeight || 480;

          // Check if partially hidden, off-screen, or covered by hand/phone
          const margin = 40;
          const isOffScreen = bbox.xMin < margin || 
                             bbox.yMin < margin || 
                             (bbox.xMax > videoWidth - margin) || 
                             (bbox.yMax > videoHeight - margin);
          const isLowConfidence = (mainFace.score !== undefined && mainFace.score < 0.90);

          // Advanced face geometry occlusion detection (e.g. hand covering mouth/nose)
          let isFaceOccluded = false;
          const keypoints = mainFace.keypoints;
          if (keypoints && keypoints.length >= 153) {
            const kp = (idx: number) => keypoints[idx] || { x: 0, y: 0, z: 0 };
            
            const nose = kp(4);
            const leftCheek = kp(234);
            const rightCheek = kp(454);
            
            // Calculate 3D Euclidean distance for face width
            const faceWidth = Math.sqrt(
              Math.pow(rightCheek.x - leftCheek.x, 2) + 
              Math.pow(rightCheek.y - leftCheek.y, 2) + 
              Math.pow((rightCheek.z ?? 0) - (leftCheek.z ?? 0), 2)
            ) || 1;

            // Mouth landmarks
            const mouthLeft = kp(61);
            const mouthRight = kp(291);
            const upperLip = kp(0);
            const lowerLip = kp(17);

            // Mouth width and ratio
            const mouthWidth = Math.sqrt(
              Math.pow(mouthRight.x - mouthLeft.x, 2) + 
              Math.pow(mouthRight.y - mouthLeft.y, 2)
            );
            const mouthWidthRatio = mouthWidth / faceWidth;

            // Horizontal mouth symmetry (distance from nose to left/right corners)
            const d_nose_mouth_left = Math.sqrt(Math.pow(mouthLeft.x - nose.x, 2) + Math.pow(mouthLeft.y - nose.y, 2));
            const d_nose_mouth_right = Math.sqrt(Math.pow(mouthRight.x - nose.x, 2) + Math.pow(mouthRight.y - nose.y, 2));
            const mouthSymmetryRatio = Math.max(d_nose_mouth_left, d_nose_mouth_right) / (Math.min(d_nose_mouth_left, d_nose_mouth_right) || 1);

            // Z-depth ratio (mouth depth behind nose tip, normalized by face width)
            // Normally, mouth is significantly behind the nose tip (positive diff in MediaPipe coordinate space).
            // If covered by hand/object, the mouth landmarks map forward onto the hand/object, reducing this depth gap.
            const avgMouthZ = ((upperLip.z ?? 0) + (lowerLip.z ?? 0)) / 2;
            const mouthZDepthRatio = (avgMouthZ - (nose.z ?? 0)) / faceWidth;

            // Occlusion triggers:
            // 1. Mouth width ratio collapses (mouth landmarks compressed horizontally)
            // 2. Mouth symmetry ratio is highly skewed (half of mouth covered/distorted)
            // 3. Mouth Z-depth ratio is compressed (mouth pushed unnaturally forward, i.e., < 0.04 of face width)
            if (mouthWidthRatio < 0.16 || mouthSymmetryRatio > 1.8 || mouthZDepthRatio < 0.04) {
              isFaceOccluded = true;
            }
          }

          if (isOffScreen || isLowConfidence || handOverlapsFace || phoneOverlapsFace || isFaceOccluded) {
            currentFaceWarning = "Face Partially Hidden";
            triggerViolation("Face Partially Hidden");
          }

          // Gaze and Head Pose checks
          if (keypoints && keypoints.length >= 153) {
            const kp = (idx: number) => keypoints[idx] || { x: 0, y: 0, z: 0 };
            
            // Yaw (looking left/right)
            const leftCheek = kp(234);
            const rightCheek = kp(454);
            const nose = kp(4);
            const d_left = Math.abs(nose.x - leftCheek.x);
            const d_right = Math.abs(rightCheek.x - nose.x);
            const yawRatio = d_left / (d_left + d_right || 1);

            // Pitch (looking up/down)
            const forehead = kp(10);
            const chin = kp(152);
            const d_up = Math.abs(nose.y - forehead.y);
            const d_down = Math.abs(chin.y - nose.y);
            const pitchRatio = d_up / (d_up + d_down || 1);

            if (yawRatio < 0.40 || yawRatio > 0.60 || pitchRatio < 0.41 || pitchRatio > 0.68) {
              currentGazeWarning = "Looked Away from Screen";
              triggerViolation("Looked Away from Screen");
            }

            // Eye Shifting / Pupil tracking
            let iris_left = keypoints[468];
            let iris_right = keypoints[473];
            if (!iris_left || !iris_right) {
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

            if (avgGaze < 0.38 || avgGaze > 0.62) {
              currentGazeWarning = "Eye Shifting / Rapid Eye";
              triggerViolation("Eye Shifting / Rapid Eye Movement");
            } else {
              // Check rapid transitions (REM)
              let shiftsCount = 0;
              const history = gazeHistoryRef.current;
              for (let i = 1; i < history.length; i++) {
                if (Math.abs(history[i] - history[i-1]) > 0.12) {
                  shiftsCount++;
                }
              }
              if (shiftsCount >= 3) {
                currentGazeWarning = "Eye Shifting / Rapid Eye";
                triggerViolation("Eye Shifting / Rapid Eye Movement");
              }
            }
          }
        }

        // Handle cell phone violation warnings locally
        if (phoneDetected) {
          currentBlurWarning = "Cell Phone Detected";
          triggerViolation("Cell Phone Detected");
        }

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
      // Only count as window blur if tab is not already hidden (to avoid double count)
      if (!document.hidden) {
        setTabSwitchCount(prev => prev + 1);
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
