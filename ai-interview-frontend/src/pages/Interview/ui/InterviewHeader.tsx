import { Bot, Mic, MicOff, Clock } from "lucide-react";

interface Props {
  interviewData: any;
  isMuted: boolean;
  isUserSpeaking: boolean;
  isAiSpeaking: boolean;
  timeLeft: number;
  formatTime: (s: number) => string;
  sessionId: string | null;
  interviewComplete: boolean;
}

export default function InterviewHeader({
  interviewData,
  isMuted,
  isUserSpeaking,
  isAiSpeaking,
  timeLeft,
  formatTime,
  sessionId,
  interviewComplete,
}: Props) {
  return (
    <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-white">
      <div id="tour-header-info" className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center">
          <Bot size={20} className="text-gray-700" />
        </div>

        <div>
          <h1 className="text-lg font-semibold text-gray-900 tracking-tight">
            Technical Interview Session
          </h1>

          <div className="text-xs text-gray-500 font-medium mt-0.5 flex items-center gap-2">
            <span>AI Interviewer</span>
            <span className="w-1 h-1 rounded-full bg-gray-300" />
            <span>
              Role: {interviewData?.job?.title || interviewData?.job_post?.title || "Loading..."}
            </span>
          </div>
        </div>
      </div>

      <div id="tour-header-controls" className="flex items-center gap-3">
        {sessionId && !interviewComplete && (
          <div
            className={`flex items-center gap-2 px-3 py-1.5 border rounded-md ${
              isMuted || isAiSpeaking
                ? "bg-gray-50 border-gray-200"
                : isUserSpeaking
                ? "bg-green-50 border-green-300"
                : "bg-blue-50 border-blue-200"
            }`}
          >
            {isMuted || isAiSpeaking ? (
              <MicOff size={14} className="text-gray-400" />
            ) : (
              <Mic size={14} className="text-blue-500" />
            )}

            <span className="text-xs font-bold uppercase">
              {isMuted || isAiSpeaking ? "Muted" : "Mic On"}
            </span>
          </div>
        )}

        <div
          className={`flex items-center gap-2 px-3 py-1.5 border rounded-md ${
            timeLeft < 60
              ? "bg-red-50 border-red-200"
              : "bg-gray-50 border-gray-200"
          }`}
        >
          <Clock size={14} />
          <span className="text-sm font-mono font-semibold">
            {formatTime(timeLeft)}
          </span>
        </div>
      </div>
    </div>
  );
}