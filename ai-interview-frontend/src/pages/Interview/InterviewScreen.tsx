import { useState, useRef, useEffect } from "react";
import toast from "react-hot-toast";
import { X } from "lucide-react";
import { AiInterviewLayout, InterviewHeader, InterviewStage, TranscriptPanel, InterviewReport } from "./ui";
import type { InterviewReportData } from "./ui/InterviewReport";

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
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  
  const formatTime = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  useEffect(() => {
    // Demonstrate violation using Ram's custom UI toast
    const timer = setTimeout(() => {
      toast(
        (t) => (
          <div className="flex items-center justify-between gap-3 min-w-[200px]">
            <span className="text-sm font-medium">Violation Detected: Looked Away</span>
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
          duration: Infinity,
          position: "top-center",
          className: "toast-info-border",
        }
      );
      
      // Log violation to backend
      fetch("http://localhost:3000/api/log-violation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "mock-session-id",
          type: "Looked Away",
          timestamp: new Date().toISOString(),
          screenshot_base64: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==" // Mock base64
        })
      }).catch(console.error);

    }, 5000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    // Turn on the user's camera
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
      // Cleanup camera stream
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

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
          faceWarning={null}
          blurWarning={null}
          gazeWarning={null}
          secondPersonWarning={null}
          showTour={false}
          isLoading={false}
          isInitializing={false}
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
          tabSwitchCount={0}
          isFullscreen={false}
          sessionId="mock-session-id"
          interviewComplete={false}
          isEnding={false}
          enterFullscreen={() => {}}
        />
      </div>
    </AiInterviewLayout>
  );
}
