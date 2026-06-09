import React from "react";

interface Props {
  children: React.ReactNode;
}

const AiInterviewLayout: React.FC<Props> = ({ children }) => {
  return (
    <div className="h-screen bg-[#fafafa] flex flex-col p-2 sm:p-4 overflow-hidden select-none">
      <div className="w-full max-w-[1400px] mx-auto flex-1 bg-white border border-gray-200 rounded-xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden flex flex-col min-h-0">
        {children}
      </div>
    </div>
  );
};

export default AiInterviewLayout;