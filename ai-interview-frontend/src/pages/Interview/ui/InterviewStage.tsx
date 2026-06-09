import React from "react";
import AiAvatar from "./AiAvatar";
import SimliAvatarView from "./SimliAvatarView";
import {
  User,
  MicOff,
  PhoneOff,
  Loader2,
  Activity,
  Users,
  EyeOff,
  Eye,
  Camera
} from "lucide-react";

interface Props {
  isUserSpeaking: boolean;
  isMuted: boolean;
  isVideoOn: boolean;
videoRef: React.RefObject<HTMLVideoElement | null>;
  faceWarning: string | null;
  blurWarning: string | null;
  gazeWarning: string | null;
  secondPersonWarning: string | null;
  showTour: boolean;
  isLoading: boolean;
  isInitializing: boolean;
  isAiSpeaking: boolean;
  isProcessing: boolean;
  speechFailed: boolean;
  audioAnalyser: any;
  handleEndInterview: (
    reason?: "CANDIDATE_ENDED" | "TIMER_ENDED",
    overrideMessages?: any[]
  ) => Promise<void> | void;
  isEnding?: boolean;
  // --- Simli Props ---
  simliMode: boolean;
  simliVideoRef: React.RefObject<HTMLVideoElement | null>;
  simliAudioRef: React.RefObject<HTMLAudioElement | null>;
  isSimliConnected: boolean;
  interviewComplete: boolean;
  handleManualSubmit?: () => void;
  isTranscribing?: boolean;
}

export default function InterviewStage({
  isUserSpeaking,
  isMuted,
  isVideoOn,
  videoRef,
  faceWarning,
  blurWarning,
  gazeWarning,
  secondPersonWarning,
  showTour,
  isLoading,
  isInitializing,
  isAiSpeaking,
  isProcessing,
  speechFailed,
  audioAnalyser,
  handleEndInterview,
  isEnding,
  simliMode,
  simliVideoRef,
  simliAudioRef,
  isSimliConnected,
  interviewComplete,
  handleManualSubmit,
  isTranscribing

}: Props) {
  return (
    <div className="flex-[1.5] flex flex-col p-4 sm:p-6 bg-gray-50/50 min-h-0">

      <div className="flex-1 bg-[#0A0A0A] rounded-xl overflow-hidden relative shadow-inner border border-gray-200/50 flex flex-col min-h-[300px]">

        {/* User PiP */}
        <div
          id="tour-camera"
          className={`absolute top-4 right-4 w-48 sm:w-56 md:w-64 aspect-video rounded-lg border-2 shadow-xl z-20 bg-gray-900 transition-all duration-300 ${
            isUserSpeaking && !isMuted
              ? "border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.4)] scale-105"
              : "border-white/10"
          }`}
        >

          {isUserSpeaking && !isMuted && (
            <>
              <div className="absolute inset-0 rounded-lg border-2 border-blue-400/50 animate-ping" style={{ animationDuration: "2s" }} />
              <div className="absolute inset-[-8px] rounded-xl border border-blue-400/20 animate-pulse" />
            </>
          )}

          <div className="absolute inset-0 overflow-hidden rounded-lg">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover scale-x-[-1] ${isVideoOn ? 'block' : 'hidden'}`}
            />
            {!isVideoOn && (
              <div className="w-full h-full flex items-center justify-center">
                <div className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center border border-gray-700">
                  <User size={20} className="text-gray-400" />
                </div>
              </div>
            )}
          </div>

          <div
            className={`absolute bottom-2 left-2 px-2 py-1 rounded text-[10px] font-medium text-white shadow-sm border ${
              isMuted
                ? "bg-red-500/80 border-red-400"
                : "bg-black/60 border-white/10"
            } backdrop-blur-sm z-20 flex items-center gap-1.5`}
          >
            {isMuted ? (
              <MicOff size={10} />
            ) : (
              <div className="flex items-center gap-0.5">
                <div className="w-1 h-1 bg-green-400 rounded-full animate-pulse" />
                <div className="w-1 h-3 bg-green-400 rounded-full animate-pulse delay-75" />
                <div className="w-1 h-1.5 bg-green-400 rounded-full animate-pulse delay-150" />
              </div>
            )}
            You {isMuted ? "(Muted)" : "(Live)"}
          </div>

          {(faceWarning || blurWarning || gazeWarning || secondPersonWarning) && (
            <div className="absolute top-0 left-0 right-0 bg-red-600/90 backdrop-blur-sm text-white text-[9px] font-bold px-2 py-1 text-center z-30 flex items-center justify-center gap-1 animate-pulse">
              {secondPersonWarning && (
                <>
                  <Users size={10} /> {secondPersonWarning}
                </>
              )}
              {!secondPersonWarning && faceWarning && (
                <>
                  <EyeOff size={10} /> {faceWarning}
                </>
              )}
              {!secondPersonWarning && !faceWarning && gazeWarning && (
                <>
                  <Eye size={10} /> {gazeWarning}
                </>
              )}
              {!secondPersonWarning && !faceWarning && !gazeWarning && blurWarning && (
                <>
                  <Camera size={10} /> {blurWarning}
                </>
              )}
            </div>
          )}
        </div>

        {/* AI View */}
        <div id="tour-ai-avatar" className="flex-1 flex flex-col items-center justify-center relative">
          
          {/* Always render SimliAvatarView in DOM to ensure refs are valid for background prewarming.
              We use opacity-0/absolute to hide it visually without using display:none which might pause the video */}
          <div className={`relative flex items-center justify-center mb-8 ${(showTour || isLoading) ? 'opacity-0 absolute pointer-events-none' : 'opacity-100'}`}>
            {isAiSpeaking && !showTour && !isLoading && (
              <>
                <div className="absolute w-64 h-64 rounded-full border border-sky-500/20 animate-ping" />
                <div className="absolute w-48 h-48 rounded-full border border-sky-500/30 animate-ping" />
              </>
            )}

            <div className="w-32 h-32 sm:w-48 sm:h-48 rounded-full border border-sky-900 shadow-[0_0_50px_rgba(56,189,248,0.15)] flex items-center justify-center relative z-10 overflow-hidden bg-slate-900">
              {simliMode ? (
                <SimliAvatarView
                  videoRef={simliVideoRef}
                  audioRef={simliAudioRef}
                  isSimliConnected={isSimliConnected}
                />
              ) : (
                <AiAvatar isSpeaking={isAiSpeaking} analyserNode={audioAnalyser} />
              )}
            </div>
          </div>

          {!showTour && !isLoading && (
            <div className="text-center z-10">
              <h2 className="text-white font-medium text-lg">
                AI Interviewer
              </h2>

              <div className="flex items-center gap-2 justify-center mt-2 text-gray-400 text-sm">
                <Activity size={14} />
                <span>
                  {isInitializing
                    ? "Waking up AI..."
                    : isAiSpeaking
                    ? "Speaking..."
                    : isProcessing
                    ? "Thinking..."
                    : isUserSpeaking
                    ? "Hearing you..."
                    : speechFailed
                    ? "Mic unavailable — type below"
                    : isMuted
                    ? "Waiting..."
                    : "Listening..."}
                </span>
              </div>
            </div>
          )}

          {showTour && (
            <div className="text-gray-400 text-sm font-medium z-20">
              Interview will begin after tour
            </div>
          )}
          
          {!showTour && isLoading && (
            <div className="flex flex-col items-center gap-4 z-20">
              <Loader2 size={40} className="text-gray-400 animate-spin" />
              <span className="text-gray-400 text-sm font-medium">
                Connecting to AI Interviewer...
              </span>
            </div>
          )}
        </div>

        {/* Controls */}
        {!interviewComplete && !isEnding && (
        <div
          id="tour-controls"
          className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 bg-gray-900/90 backdrop-blur-sm px-6 py-3 rounded-full border border-white/10 shadow-2xl z-20"
        >

          {!simliMode && handleManualSubmit && !isAiSpeaking && !isMuted && !isProcessing && !isLoading && !showTour && (
            <button
              onClick={handleManualSubmit}
              disabled={isTranscribing}
              className={`px-4 py-2 sm:px-6 sm:py-3 h-10 sm:h-12 rounded-full font-semibold flex items-center gap-2 shadow-[0_0_15px_rgba(59,130,246,0.4)] border transition-all duration-300 whitespace-nowrap ${
                isTranscribing 
                  ? 'bg-gray-600 border-gray-500 text-gray-300 cursor-not-allowed' 
                  : 'bg-linear-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 text-white border-blue-400/30 hover:scale-105 active:scale-95'
              }`}
            >
              {isTranscribing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span className="hidden sm:inline">Submitting...</span>
                  <span className="sm:hidden">Wait...</span>
                </>
              ) : (
                <>
                  <span className="hidden sm:inline">Submit Answer</span>
                  <span className="sm:hidden">Submit</span>
                </>
              )}
            </button>
          )}

          <button
            onClick={() => handleEndInterview("CANDIDATE_ENDED")}
            className="px-4 py-2 sm:px-6 sm:py-3 h-10 sm:h-12 rounded-full bg-linear-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white font-semibold flex items-center gap-2 shadow-[0_0_15px_rgba(220,38,38,0.4)] border border-red-400/30 transition-all duration-300 hover:scale-105 active:scale-95 whitespace-nowrap"
          >
            <PhoneOff size={18} />
            <span className="hidden sm:inline">End</span>
          </button>
        </div>
        )}

      </div>
    </div>
  );
}