import { useState, useRef, useEffect } from "react";
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
    if (!modelsLoaded || !detector || isInterviewComplete) {
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

        if (!faces || faces.length === 0) {
          if (faceMissingStartRef.current === null) {
            faceMissingStartRef.current = Date.now();
          }
          const missingSec = (Date.now() - faceMissingStartRef.current) / 1000;
          if (missingSec >= 1.5) {
            currentFaceWarning = "Face Missing from Frame";
            triggerViolation("Face Missing from Frame");
          }
        } else {
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

          // Check if partially hidden or off-screen
          const margin = 40;
          const isOffScreen = bbox.xMin < margin || 
                             bbox.yMin < margin || 
                             (bbox.xMax > videoWidth - margin) || 
                             (bbox.yMax > videoHeight - margin);
          const isLowConfidence = (mainFace.score !== undefined && mainFace.score < 0.90);

          if (isOffScreen || isLowConfidence) {
            currentFaceWarning = "Face Partially Hidden";
            triggerViolation("Face Partially Hidden");
          }

          // Gaze and Head Pose checks
          const keypoints = mainFace.keypoints;
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

        // Run YOLOv8 object detector via backend for cell phone presence
        try {
          const screenshot = captureScreenshot();
          if (screenshot) {
            const res = await fetch("http://localhost:3000/api/detect-objects", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ screenshot_base64: screenshot })
            });
            if (res.ok) {
              const data = await res.json();
              if (data.status === "success" && data.detected) {
                const hasPhone = data.detected.some((className: string) => 
                  className === "cell phone" || className === "phone"
                );
                if (hasPhone) {
                  currentBlurWarning = "Cell Phone Detected";
                  triggerViolation("Cell Phone Detected");
                }
              }
            }
          }
        } catch (phoneErr) {
          console.error("Error detecting cell phone via backend:", phoneErr);
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
  }, [modelsLoaded, detector, isInterviewComplete]);

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
