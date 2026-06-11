import { AlertTriangle, ArrowLeft } from "lucide-react";

export interface ViolationData {
  type: string;
  timestamp: string;
  screenshot_base64: string;
}

export interface InterviewReportData {
  overallScore: number;
  feedback: string;
  violations: ViolationData[];
  totalViolations: number;
}

interface Props {
  reportData: InterviewReportData;
  onBack: () => void;
}

const ALL_VIOLATIONS = [
  "Face Missing from Frame",
  "Multiple Faces Detected",
  "Cell Phone Detected",
  "Looked Away from Screen",
  "Eye Shifting / Rapid Eye Movement",
  "Face Partially Hidden",
  "Tab Switched / Left Window",
  "Exited Fullscreen"
];

export default function InterviewReport({ reportData, onBack }: Props) {
  return (
    <div className="flex-1 bg-[#fafafa] flex flex-col items-center p-4 sm:p-8 h-full overflow-y-auto">
      <div className="max-w-3xl w-full space-y-6 pb-20">
        
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Interview Completed</h1>
            <p className="text-gray-500 mt-1">Here is your detailed session report.</p>
          </div>
          <button 
            onClick={onBack}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-lg shadow-sm hover:bg-gray-50 font-medium text-gray-700 transition-colors"
          >
            <ArrowLeft size={16} /> Back to Dashboard
          </button>
        </div>

        {/* Violations Summary List */}
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="flex items-center gap-2 mb-6">
            <AlertTriangle className="text-gray-700" size={20} />
            <h2 className="text-lg font-bold text-gray-900">Proctoring Violations Log</h2>
          </div>
          
          <div className="grid grid-cols-1 gap-4">
            {ALL_VIOLATIONS.map((type, index) => {
              // Normalize the type match in case backend/frontend naming differs slightly
              // We'll use exact match for now
              const instances = reportData.violations.filter(v => 
                v.type === type || 
                (v.type === "Looked Away" && type === "Looked Away from Screen")
              );
              const count = instances.length;
              
              return (
                <div key={index} className={`flex flex-col p-4 border rounded-lg ${count > 0 ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-100'}`}>
                  <div className="flex items-center justify-between">
                    <span className={`font-medium ${count > 0 ? 'text-red-900' : 'text-gray-700'}`}>{type}</span>
                    <span className={`px-3 py-1 rounded-full font-bold text-sm ${count > 0 ? 'bg-red-200 text-red-900' : 'bg-gray-200 text-gray-700'}`}>
                      {count} {count === 1 ? 'time' : 'times'}
                    </span>
                  </div>
                  
                  {count > 0 && (
                    <div className="mt-4 flex gap-3 overflow-x-auto pb-2">
                      {instances.map((v, i) => (
                        <div key={i} className="flex flex-col gap-1 shrink-0">
                          {v.screenshot_base64 && (
                            <img 
                              src={v.screenshot_base64} 
                              alt={`Violation: ${v.type}`} 
                              className="w-32 h-24 object-cover rounded shadow border border-red-300"
                            />
                          )}
                          <span className="text-xs text-red-600 font-medium">{new Date(v.timestamp).toLocaleTimeString()}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
}
