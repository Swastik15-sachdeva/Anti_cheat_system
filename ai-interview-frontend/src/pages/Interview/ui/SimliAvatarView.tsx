import React from 'react';
import { Loader2 } from 'lucide-react';

interface Props {
    videoRef: React.RefObject<HTMLVideoElement | null>;
    audioRef: React.RefObject<HTMLAudioElement | null>;
    isSimliConnected: boolean;
}

export default function SimliAvatarView({ videoRef, audioRef, isSimliConnected }: Props) {
    return (
        <div className="w-full h-full bg-slate-900 flex items-center justify-center relative overflow-hidden rounded-full border border-sky-500/30">
            {/* The Simli WebRTC Video */}
            <video
                ref={videoRef as React.RefObject<HTMLVideoElement>}
                autoPlay
                playsInline
                className={`w-full h-full object-cover transition-opacity duration-1000 ${isSimliConnected ? 'opacity-100' : 'opacity-0'}`}
            />
            {/* The Simli WebRTC Audio (hidden) */}
            <audio ref={audioRef as React.RefObject<HTMLAudioElement>} autoPlay playsInline className="hidden" />
            
            {!isSimliConnected && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0B1120] z-10 rounded-full overflow-hidden">
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(56,189,248,0.15),transparent_70%)] animate-pulse" style={{ animationDuration: '2s' }} />
                    <div className="absolute inset-2 border border-sky-500/20 rounded-full animate-[spin_4s_linear_infinite]" />
                    <div className="absolute inset-4 border border-sky-400/30 rounded-full animate-[spin_3s_linear_infinite_reverse]" />
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 z-20">
                        <Loader2 size={18} className="text-sky-400 animate-spin" />
                        <span className="text-[9px] text-sky-300 font-medium tracking-widest uppercase">Initializing</span>
                    </div>
                </div>
            )}
        </div>
    );
}
