import { useState, useRef, useEffect, useCallback } from "react";
import toast from "react-hot-toast";
import { X } from "lucide-react";
import { AiInterviewLayout, InterviewHeader, InterviewStage, TranscriptPanel, InterviewReport } from "./ui";
import type { InterviewReportData } from "./ui/InterviewReport";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import "@tensorflow/tfjs-backend-webgl";

const MOCK_MESSAGES = [
  { role: "system", text: "Interview started", timestamp: new Date() },
  { role: "ai", text: "Hello! Welcome to your interview for the Frontend Engineer position. Are you ready to begin?", timestamp: new Date() }
];

// Configuration constants for detection thresholds
const DETECTION_CONFIG = {
  FACE_MISSING_THRESHOLD_MS: 3000,
  YAW_THRESHOLD_MIN: 0.35,
  YAW_THRESHOLD_MAX: 0.65,
  PITCH_THRESHOLD_MIN: 0.38,
  PITCH_THRESHOLD_MAX: 0.70,
  GAZE_RATIO_EXTREME: 0.3,
  GAZE_HISTORY_SIZE: 8,
  REM_SHIFT_THRESHOLD: 0.10,
  REM_MIN_SHIFTS: 2,
  VIOLATION_THROTTLE_MS: 10000,
  DETECTION_INTERVAL_MS: 800,
};

export default function InterviewScreen() {
  const [isMuted, setIsMuted] = useState(false);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [timeLeft, setTimeLeft] = useState(300);
  const [isInterviewComplete, setIsInterviewComplete] = useState(false);
  const [reportData, setReportData] = useState<InterviewReportData | null>(null);
  const [isProcessingEnd, setIsProcessingEnd] = useState(false);
  const [sessionId, setSessionId] = useState("mock-session-id");
  
  // Real-time proctoring states
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [detector, setDetector] = useState<any>(null);
  const [tabSwitchCount, setTabSwitchCount] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  
  const [faceWarning, setFaceWarning] = useState<string | null>(null);
  const [gazeWarning, setGazeWarning] = useState<string | null>(null);
  const [secondPersonWarning, setSecondPersonWarning] = useState<string | null>(null);
  const [blurWarning, setBlurWarning] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  const lastLoggedRef = useRef<{ [key: string]: number }>({});
  const faceMissingStartRef = useRef<number | null>(null);
  const gazeHistoryRef = useRef<number[]>([]);
  const detectionInProgressRef = useRef(false);
  const remTrackingRef = useRef<{ shifts: number; lastValue: number | null }>({ shifts: 0, lastValue: null });

  const formatTime = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const debug = useCallback((label: string, data: any) => {
    if (debugMode) {
      console.log(`[PROCTORING DEBUG] ${label}:`, data);
    }
  }, [debugMode]);

  // Initialize session
  useEffect(() => {
    const newSessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    
    // Start session on backend
    fetch("http://localhost:3000/api/start-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: newSessionId })
    })
    .then(res => res.json())
    .then(data => {
      debug("Session Started", { sessionId: data.session_id });
    })
    .catch(err => console.error("Failed to start session:", err));
  }, [debug]);

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

        if (active) {
          setDetector(faceDetector);
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
        toast.error("Camera access denied. Please enable camera permissions.");
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
    return canvas.toDataURL("image/jpeg", 0.6);
  };

  // 4. Violation triggering with throttling
  const triggerViolation = useCallback((type: string, screenshot?: string) => {
    const now = Date.now();
    const lastTime = lastLoggedRef.current[type] || 0;
    if (now - lastTime < DETECTION_CONFIG.VIOLATION_THROTTLE_MS) {
      debug("Violation Throttled", { type, timeSinceLast: now - lastTime });
      return;
    }
    lastLoggedRef.current[type] = now;
    const imageData = screenshot || captureScreenshot();

    debug("Violation Triggered", { type, timestamp: new Date().toISOString() });

    // Log to backend FastAPI
    fetch("http://localhost:3000/api/log-violation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        type: type,
        timestamp: new Date().toISOString(),
        screenshot_base64: imageData
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
  }, [debug, sessionId]);

  // 5. Main Proctoring Detection Loop
  useEffect(() => {
    if (!modelsLoaded || !detector || isInterviewComplete) {
      return;
    }

    const intervalId = setInterval(async () => {
      if (detectionInProgressRef.current) {
        return;
      }
      detectionInProgressRef.current = true;

      const video = videoRef.current;
      if (!video || video.readyState !== 4 || video.paused || video.ended) {
        detectionInProgressRef.current = false;
        return;
      }

      try {
        let currentFaceWarning: string | null = null;
        let currentGazeWarning: string | null = null;
        let currentSecondPersonWarning: string | null = null;
        let currentBlurWarning: string | null = null;

        const screenshot = captureScreenshot();
        const faces = await detector.estimateFaces(video, { flipHorizontal: false });

        debug("Face Detection", { facesDetected: faces?.length || 0 });

        if (!faces || faces.length === 0) {
          if (faceMissingStartRef.current === null) {
            faceMissingStartRef.current = Date.now();
          }
          const missingSec = (Date.now() - faceMissingStartRef.current) / 1000;
          
          debug("Face Missing Timer", { missingSec, threshold: DETECTION_CONFIG.FACE_MISSING_THRESHOLD_MS / 1000 });
          
          if (missingSec >= DETECTION_CONFIG.FACE_MISSING_THRESHOLD_MS / 1000) {
            currentFaceWarning = "Face Missing from Frame";
            triggerViolation("Face Missing from Frame", screenshot);
          }
        } else {
          faceMissingStartRef.current = null;

          if (faces.length >= 2) {
            currentSecondPersonWarning = "Multiple Faces Detected";
            triggerViolation("Multiple Faces Detected", screenshot);
          }

          const mainFace = faces[0];
          const bbox = mainFace.box;
          const videoWidth = video.videoWidth || 640;
          const videoHeight = video.videoHeight || 480;

          const margin = 40;
          const isOffScreen = bbox.xMin < margin || 
                             bbox.yMin < margin || 
                             (bbox.xMax > videoWidth - margin) || 
                             (bbox.yMax > videoHeight - margin);
          const isLowConfidence = (mainFace.score !== undefined && mainFace.score < 0.90);

          if (isOffScreen || isLowConfidence) {
            currentFaceWarning = "Face Partially Hidden";
            triggerViolation("Face Partially Hidden", screenshot);
          }

          const keypoints = mainFace.keypoints;
          if (keypoints && keypoints.length >= 153) {
            const kp = (idx: number) => keypoints[idx] || { x: 0, y: 0, z: 0 };
            
            const leftCheek = kp(234);
            const rightCheek = kp(454);
            const nose = kp(4);
            const d_left = Math.abs(nose.x - leftCheek.x);
            const d_right = Math.abs(rightCheek.x - nose.x);
            const yawRatio = d_left / (d_left + d_right || 1);

            const forehead = kp(10);
            const chin = kp(152);
            const d_up = Math.abs(nose.y - forehead.y);
            const d_down = Math.abs(chin.y - nose.y);
            const pitchRatio = d_up / (d_up + d_down || 1);

            debug("Head Pose", { yawRatio, pitchRatio });

            if (yawRatio < DETECTION_CONFIG.YAW_THRESHOLD_MIN || 
                yawRatio > DETECTION_CONFIG.YAW_THRESHOLD_MAX || 
                pitchRatio < DETECTION_CONFIG.PITCH_THRESHOLD_MIN || 
                pitchRatio > DETECTION_CONFIG.PITCH_THRESHOLD_MAX) {
              currentGazeWarning = "Looked Away from Screen";
              triggerViolation("Looked Away from Screen", screenshot);
            }

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
            
            debug("Gaze Metrics", { gazeRatioLeft, gazeRatioRight, avgGaze });

            if (avgGaze < DETECTION_CONFIG.GAZE_RATIO_EXTREME || avgGaze > (1 - DETECTION_CONFIG.GAZE_RATIO_EXTREME)) {
              currentGazeWarning = "Eye Shifting / Rapid Eye";
              triggerViolation("Eye Shifting / Rapid Eye Movement", screenshot);
              remTrackingRef.current = { shifts: 0, lastValue: null };
            } else {
              gazeHistoryRef.current.push(avgGaze);
              if (gazeHistoryRef.current.length > DETECTION_CONFIG.GAZE_HISTORY_SIZE) {
                gazeHistoryRef.current.shift();
              }

              if (remTrackingRef.current.lastValue !== null) {
                const shift = Math.abs(avgGaze - remTrackingRef.current.lastValue);
                if (shift > DETECTION_CONFIG.REM_SHIFT_THRESHOLD) {
                  remTrackingRef.current.shifts++;
                  debug("REM Detected", { shift, shiftsCount: remTrackingRef.current.shifts });
                } else {
                  remTrackingRef.current.shifts = Math.max(0, remTrackingRef.current.shifts - 1);
                }

                if (remTrackingRef.current.shifts >= DETECTION_CONFIG.REM_MIN_SHIFTS) {
                  currentGazeWarning = "Eye Shifting / Rapid Eye";
                  triggerViolation("Eye Shifting / Rapid Eye Movement", screenshot);
                  remTrackingRef.current = { shifts: 0, lastValue: null };
                }
              }
              remTrackingRef.current.lastValue = avgGaze;
            }
          }
        }

        // YOLOv8 object detection with retry logic
        try {
          if (screenshot) {
            const res = await fetch("http://localhost:3000/api/detect-objects", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ screenshot_base64: screenshot })
            });
            if (res.ok) {
              const data = await res.json();
              if (data.status === "success" && data.detected) {
                debug("Object Detection", { detected: data.detected });
                const hasPhone = data.detected.some((className: string) => 
                  className.toLowerCase().includes("cell phone") || 
                  className.toLowerCase().includes("phone") ||
                  className.toLowerCase().includes("book")
                );
                if (hasPhone) {
                  currentBlurWarning = "Cell Phone Detected";
                  triggerViolation("Cell Phone Detected", screenshot);
                }
              }
            }
          }
        } catch (phoneErr) {
          debug("Object Detection Error", phoneErr);
        }

        setFaceWarning(currentFaceWarning);
        setGazeWarning(currentGazeWarning);
        setSecondPersonWarning(currentSecondPersonWarning);
        setBlurWarning(currentBlurWarning);

      } catch (err) {
        console.error("Error in detection loop:", err);
      } finally {
        detectionInProgressRef.current = false;
      }
    }, DETECTION_CONFIG.DETECTION_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, [modelsLoaded, detector, isInterviewComplete, triggerViolation, debug]);

  // 6. Browser event listeners
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        setTabSwitchCount(prev => prev + 1);
        triggerViolation("Tab Switched / Left Window");
      }
    };

    const handleBlur = () => {
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
  }, [isInterviewComplete, triggerViolation]);

  const enterFullscreen = () => {
    const element = document.documentElement as any;
    if (element.requestFullscreen) {
      element.requestFullscreen().then(() => {
        setIsFullscreen(true);
      }).catch((err: any) => {
        console.error("Error attempting to enable fullscreen:", err);
      });
    } else if (element.webkitRequestFullscreen) {
      element.webkitRequestFullscreen();
      setIsFullscreen(true);
    } else if (element.mozRequestFullScreen) {
      element.mozRequestFullScreen();
      setIsFullscreen(true);
    } else if (element.msRequestFullscreen) {
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
        sessionId={sessionId}
        interviewComplete={false}
      />
      
      {/* Debug Toggle */}
      <div className="absolute top-2 right-2 z-50">
        <button
          onClick={() => setDebugMode(!debugMode)}
          className="text-xs px-2 py-1 bg-gray-200 rounded hover:bg-gray-300"
          title="Toggle debug logging in console"
        >
          Debug: {debugMode ? "ON" : "OFF"}
        </button>
      </div>

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
              const res = await fetch(`http://localhost:3000/api/get-report?session_id=${sessionId}`);
              if (!res.ok) throw new Error("Failed to fetch report");
              const data = await res.json();
              setReportData(data);
              setIsInterviewComplete(true);
            } catch (error) {
              console.error("Backend fetch failed, using fallback mock data:", error);
              toast.error("Backend offline. Showing sample report.");
              
              setReportData({
                overallScore: 85,
                feedback: "This is a sample generated report because the backend is currently offline. The candidate demonstrated strong skills but had a few minor hesitations.",
                totalViolations: tabSwitchCount,
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
          sessionId={sessionId}
          interviewComplete={isInterviewComplete}
          isEnding={isProcessingEnd}
          enterFullscreen={enterFullscreen}
        />
      </div>
    </AiInterviewLayout>
  );
}