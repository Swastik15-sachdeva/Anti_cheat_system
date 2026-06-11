import React from "react";
import {
  MessageSquare,
  Bot,
  User,
  Loader2,
  Shield,
  AlertTriangle,
  RefreshCw,
  Maximize
} from "lucide-react";

interface Props {
  isMuted: boolean;
  isAiSpeaking: boolean;
  isUserSpeaking?: boolean;
  isProcessing: boolean;
  speechFailed: boolean;
  handleRetryMic: () => void;
  messages: any[];
  chatEndRef: React.RefObject<HTMLDivElement | null>;
  isModelLoaded: boolean;
  tabSwitchCount: number;
  isFullscreen: boolean;
  sessionId: string | null;
  interviewComplete: boolean;
  isEnding: boolean;
  enterFullscreen: () => void;
}

export default function TranscriptPanel({
  isMuted,
  isAiSpeaking,
  isProcessing,
  speechFailed,
  handleRetryMic,
  messages,
  chatEndRef,
  isModelLoaded,
  tabSwitchCount,
  isFullscreen,
  sessionId,
  interviewComplete,
  isEnding,
  enterFullscreen
}: Props) {
  return (
    <>
      {/* Right - Transcript */}
                    <div id="tour-transcript" className="flex-1 lg:max-w-[400px] xl:max-w-[450px] border-t lg:border-t-0 lg:border-l border-gray-200 flex flex-col bg-white min-h-0 shrink-0">
                        <div className="px-5 py-4 border-b border-gray-100 flex justify-between items-center">
                            <div className="flex items-center gap-2">
                                <MessageSquare size={16} className="text-gray-500" />
                                <h2 className="font-semibold text-gray-900 text-sm">Live Transcript</h2>
                            </div>
                            {!isMuted && !isAiSpeaking && !isProcessing && !speechFailed && (
                                <div className="flex items-center gap-1.5">
                                    <span className="relative flex h-2 w-2">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                                        <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
                                    </span>
                                    <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider">Mic Active</span>
                                </div>
                            )}
                            {speechFailed && (
                                <button
                                    onClick={handleRetryMic}
                                    className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-amber-50 border border-amber-200 hover:bg-amber-100 transition-colors"
                                >
                                    <RefreshCw size={12} className="text-amber-600" />
                                    <span className="text-[10px] font-bold text-amber-600 uppercase tracking-wider">Retry Mic</span>
                                </button>
                            )}
                        </div>

                        <div className="flex-1 overflow-y-auto p-5 space-y-5 scrollbar-thin">
                            {messages.map((msg, idx) => {
                                if (msg.role === 'system') {
                                    return (
                                        <div key={idx} className="flex justify-center">
                                            <div className="max-w-[80%] bg-gray-50 text-gray-500 rounded-xl px-4 py-3 text-[13px] leading-relaxed border border-gray-200">
                                                <div className="flex items-center justify-between mb-1">
                                                    <div className="text-[10px] font-bold uppercase tracking-wider text-gray-400">
                                                        System
                                                    </div>
                                                    <div className={`text-[9px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-sm bg-black/5 text-gray-500/70`}>
                                                        Live
                                                    </div>
                                                </div>
                                                <p className="text-sm leading-relaxed">{msg.text}</p>
                                                <div className="mt-2 text-[9px] font-medium text-gray-400 text-left">
                                                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                }
                                if (msg.role === 'ai') {
                                    return (
                                        <div key={idx} className="flex gap-3 items-start">
                                            <div className="w-8 h-8 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center shrink-0 mt-0.5">
                                                <Bot size={16} className="text-gray-600" />
                                            </div>
                                            <div className="flex-1">
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-[11px] font-bold text-gray-500 block uppercase tracking-wider">AI Interviewer</span>
                                                <span className="text-[9px] font-bold text-gray-400 uppercase tracking-wider bg-gray-100 px-2 py-0.5 rounded-sm">
                                                    {msg.isFallback ? 'Troubleshooting' : 'Question'}
                                                </span>
                                            </div>
                                            <div className={`text-gray-800 rounded-lg rounded-tl-none px-4 py-3 text-[13px] leading-relaxed border border-gray-200 ${msg.isFallback ? 'bg-amber-50/50' : 'bg-gray-50' }`}>
                                                    {msg.text}
                                                </div>
                                                <div className="mt-1 text-[9px] font-medium text-gray-400 text-left">
                                                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                }
                                return (
                                    <div key={idx} className="flex gap-3 items-start flex-row-reverse">
                                        <div className="w-8 h-8 rounded-full bg-gray-900 flex items-center justify-center shrink-0 mt-0.5">
                                            <User size={16} className="text-white" />
                                        </div>
                                        <div className="flex-1 flex flex-col items-end">
                                            <span className="text-[11px] font-bold text-gray-500 mb-1 block uppercase tracking-wider">You</span>
                                            <div className={`rounded-lg rounded-tr-none px-4 py-3 text-[13px] leading-relaxed shadow-sm min-w-[140px] transition-all duration-300 ease-in-out ${
                                                msg.text 
                                                    ? 'bg-gray-900 text-white border border-gray-800' 
                                                    : 'bg-linear-to-r from-gray-800 to-gray-900 text-gray-300 border border-gray-700'
                                            }`}>
                                                {msg.text ? (
                                                    msg.text
                                                ) : msg._isTranscribing ? (
                                                    <div className="flex items-center gap-2 py-0.5">
                                                        <Loader2 size={12} className="animate-spin text-gray-400" />
                                                        <span className="text-xs font-medium text-gray-400 italic">Transcribing...</span>
                                                    </div>
                                                ) : (
                                                    <div className="flex items-center gap-2.5 py-0.5">
                                                        <div className="flex items-center gap-[3px]">
                                                            <span className="w-[3px] h-3 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_infinite]" />
                                                            <span className="w-[3px] h-4 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_0.15s_infinite]" />
                                                            <span className="w-[3px] h-2.5 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_0.3s_infinite]" />
                                                            <span className="w-[3px] h-3.5 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_0.45s_infinite]" />
                                                        </div>
                                                        <span className="text-xs font-medium text-blue-300">Listening...</span>
                                                    </div>
                                                )}
                                            </div>
                                            <div className="mt-1 text-[9px] font-medium text-gray-400 text-right">
                                                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}



                             {/* Processing */}
                            {isProcessing && !isEnding && (
                                <div className="flex gap-3 items-start">
                                    <div className="w-8 h-8 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center shrink-0 mt-0.5 shadow-sm">
                                        <Bot size={16} className="text-gray-600" />
                                    </div>
                                    <div className="flex-1">
                                        <span className="text-[11px] font-bold text-gray-500 mb-1 block uppercase tracking-wider">AI Interviewer</span>
                                        <div className="bg-gray-50 text-gray-500 rounded-lg rounded-tl-none px-4 py-3 text-[13px] leading-relaxed border border-gray-200 flex items-center gap-2">
                                            <Loader2 size={14} className="animate-spin" /> Thinking...
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Active User Turn (Listening) — hide if a live bubble already shows "Transcribing..." */}
                            {!isAiSpeaking && !isProcessing && !isEnding && !interviewComplete && messages.some(m => m.role === 'ai') && !messages.some(m => m._isLiveBubble) && (
                                <div className="flex gap-3 items-start flex-row-reverse animate-in fade-in slide-in-from-bottom-2 duration-300">
                                    <div className="w-8 h-8 rounded-full bg-gray-900 flex items-center justify-center shrink-0 mt-0.5 shadow-md">
                                        <User size={16} className="text-white" />
                                    </div>
                                    <div className="flex-1 flex flex-col items-end">
                                        <span className="text-[11px] font-bold text-gray-500 mb-1 block uppercase tracking-wider">You</span>
                                        <div className="rounded-lg rounded-tr-none px-4 py-3 text-[13px] leading-relaxed shadow-sm min-w-[140px] bg-[linear-gradient(to_right,var(--color-gray-800),var(--color-gray-900))] text-gray-300 border border-gray-700">
                                            <div className="flex items-center gap-2.5 py-0.5">
                                                <div className="flex items-center gap-[3px]">
                                                    <span className="w-[3px] h-3 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_infinite]" />
                                                    <span className="w-[3px] h-4 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_0.15s_infinite]" />
                                                    <span className="w-[3px] h-2.5 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_0.3s_infinite]" />
                                                    <span className="w-[3px] h-3.5 bg-blue-400 rounded-full animate-[pulse_0.8s_ease-in-out_0.45s_infinite]" />
                                                </div>
                                                <span className="text-xs font-medium text-blue-300">Listening...</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div ref={chatEndRef} />
                        </div>

                        {/* Security Status Bar */}
                        <div className="px-4 sm:px-5 py-3 border-t border-gray-200 bg-gray-50 flex flex-wrap items-center justify-between gap-y-3 mt-auto">
                            <div className="flex items-center gap-3 sm:gap-4 shrink-0">
                                <div className="flex items-center gap-2">
                                    <Shield size={14} className="text-green-600 shrink-0" />
                                    <span className="text-[10px] sm:text-[11px] font-bold text-green-700 uppercase tracking-wider whitespace-nowrap">Proctored Session</span>
                                </div>

                                {isModelLoaded && (
                                    <div className="flex items-center gap-1.5 border-l border-gray-300 pl-3 sm:pl-4 h-4 shrink-0">
                                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse shrink-0" />
                                        <span className="text-[9px] sm:text-[10px] font-bold text-green-600 uppercase tracking-wider whitespace-nowrap">Face Detection Active</span>
                                    </div>
                                )}
                            </div>

                            <div className="flex items-center gap-3 sm:gap-4 shrink-0">
                                {tabSwitchCount > 0 && (
                                    <div className="flex items-center gap-1.5">
                                        <AlertTriangle size={14} className="text-red-500 shrink-0" />
                                        <span className="text-[10px] sm:text-[11px] font-bold text-red-600 uppercase tracking-wider whitespace-nowrap">{tabSwitchCount} violation{tabSwitchCount > 1 ? 's' : ''}</span>
                                    </div>
                                )}
                                {!isFullscreen && sessionId && !interviewComplete && (
                                    <button
                                        onClick={enterFullscreen}
                                        className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 rounded-md bg-amber-50 border border-amber-200 hover:bg-amber-100 transition-colors cursor-pointer shrink-0"
                                    >
                                        <Maximize size={12} className="text-amber-600 shrink-0" />
                                        <span className="text-[9px] sm:text-[10px] font-bold text-amber-700 uppercase tracking-wider whitespace-nowrap">Fullscreen</span>
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
    </>
  );
}