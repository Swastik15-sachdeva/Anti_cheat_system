import React, { useEffect, useRef } from 'react';

interface AiAvatarProps {
    isSpeaking: boolean;
    analyserNode: AnalyserNode | null;
}

const AiAvatar: React.FC<AiAvatarProps> = ({ isSpeaking, analyserNode }) => {
    // Refs for the various glowing rings of the orb
    const coreRef = useRef<HTMLDivElement>(null);
    const ring1Ref = useRef<HTMLDivElement>(null);
    const ring2Ref = useRef<HTMLDivElement>(null);
    const ring3Ref = useRef<HTMLDivElement>(null);
    const animationRef = useRef<number | null>(null);

    useEffect(() => {
        if (!isSpeaking || !analyserNode) {
            // Smoothly collapse back to idle state
            if (coreRef.current) coreRef.current.style.transform = 'scale(1)';
            if (ring1Ref.current) ring1Ref.current.style.transform = 'scale(1)';
            if (ring2Ref.current) ring2Ref.current.style.transform = 'scale(1)';
            if (ring3Ref.current) ring3Ref.current.style.transform = 'scale(1)';

            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            return;
        }

        const dataArray = new Uint8Array(analyserNode.frequencyBinCount);

        const animateOrb = () => {
            animationRef.current = requestAnimationFrame(animateOrb);
            analyserNode.getByteTimeDomainData(dataArray);

            // Calculate RMS (Root Mean Square) for volume
            let sumSquares = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const normalized = (dataArray[i] / 128.0) - 1.0;
                sumSquares += normalized * normalized;
            }
            const rms = Math.sqrt(sumSquares / dataArray.length);

            // Map the RMS to different scaling factors for a 3D expanding effect
            // Base radius is 1. RMS goes ~0.0 to 0.5 for loud speech
            const scaleCore = 1 + (rms * 1.5);
            const scaleRing1 = 1 + (rms * 2.5);
            const scaleRing2 = 1 + (rms * 4.0);
            const scaleRing3 = 1 + (rms * 6.0);

            // Apply transforms (using clamp to prevent crazy huge spikes)
            if (coreRef.current) coreRef.current.style.transform = `scale(${Math.min(scaleCore, 1.5)})`;
            if (ring1Ref.current) ring1Ref.current.style.transform = `scale(${Math.min(scaleRing1, 2.0)})`;
            if (ring2Ref.current) ring2Ref.current.style.transform = `scale(${Math.min(scaleRing2, 2.8)})`;
            if (ring3Ref.current) ring3Ref.current.style.transform = `scale(${Math.min(scaleRing3, 3.5)})`;
        };

        animateOrb();

        return () => {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
        };
    }, [isSpeaking, analyserNode]);

    return (
        <div className="w-full h-full bg-slate-900 flex items-center justify-center relative overflow-hidden rounded-full">

            {/* Ambient Background Glow */}
            <div className={`absolute inset-0 bg-sky-500/10 mix-blend-screen filter blur-3xl transition-opacity duration-1000 ${isSpeaking ? 'opacity-100' : 'opacity-30'}`}></div>

            {/* The Orb Container */}
            <div className="relative w-16 h-16 flex items-center justify-center z-10">

                {/* Outer Audio Ring 3 */}
                <div
                    ref={ring3Ref}
                    className="absolute w-full h-full rounded-full border border-sky-400/20 mix-blend-screen transition-transform"
                    style={{ transitionDuration: '50ms', filter: 'blur(4px)' }}
                ></div>

                {/* Outer Audio Ring 2 */}
                <div
                    ref={ring2Ref}
                    className="absolute w-full h-full rounded-full border-2 border-sky-400/40 mix-blend-screen transition-transform"
                    style={{ transitionDuration: '50ms', filter: 'blur(2px)' }}
                ></div>

                {/* Inner Audio Ring 1 */}
                <div
                    ref={ring1Ref}
                    className="absolute w-full h-full rounded-full border-[3px] border-sky-300/60 mix-blend-screen transition-transform"
                    style={{ transitionDuration: '50ms', boxShadow: 'inset 0 0 20px rgba(125, 211, 252, 0.4)' }}
                ></div>

                {/* Core AI Sphere */}
                <div
                    ref={coreRef}
                    className={`absolute w-full h-full rounded-full bg-linear-to-br from-sky-300 via-sky-500 to-indigo-600 transition-transform ${!isSpeaking && 'animate-pulse'}`}
                    style={{
                        transitionDuration: '50ms',
                        boxShadow: '0 0 30px rgba(56, 189, 248, 0.6), inset 0 0 15px rgba(255, 255, 255, 0.8)'
                    }}
                >
                    {/* Highlight reflection to make it look like 3D glass */}
                    <div className="absolute top-1 left-2 w-6 h-4 bg-white/40 rounded-full -rotate-45 filter blur-[1px]"></div>
                </div>

            </div>

        </div>
    );
};

export default AiAvatar;
