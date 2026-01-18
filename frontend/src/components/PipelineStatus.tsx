import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, Scissors, BrainCircuit, Database, CheckCircle2, 
  Loader2, ScanLine, Sparkles, Eye, EyeOff, Trash2
} from 'lucide-react';

export type PipelineStep = 'idle' | 'extract' | 'clean' | 'chunk' | 'embed' | 'index' | 'complete' | 'error';

interface PipelineStatusProps {
  currentStep: PipelineStep;
  fileName: string;
  onDelete?: () => void; // New prop for deletion
}

export const PipelineStatus: React.FC<PipelineStatusProps> = ({ currentStep, fileName, onDelete }) => {
  // Default to hidden (collapsed) unless it is currently processing
  const [isHidden, setIsHidden] = useState(currentStep === 'complete' || currentStep === 'idle');

  // Auto-expand if a process starts
  useEffect(() => {
    if (currentStep !== 'complete' && currentStep !== 'idle' && currentStep !== 'error') {
      setIsHidden(false);
    }
  }, [currentStep]);

  const steps = [
    { id: 'extract', label: 'Parsing PDF', icon: ScanLine },
    { id: 'clean',   label: 'Cleaning', icon: Sparkles },
    { id: 'chunk',   label: 'Chunking', icon: Scissors },
    { id: 'embed',   label: 'Embedding', icon: BrainCircuit },
    { id: 'index',   label: 'Indexing', icon: Database },
  ];

  const getStepStatus = (stepId: string) => {
    if (currentStep === 'error') return 'pending';
    if (currentStep === 'complete') return 'completed'; 
    
    const stepIndices: Record<string, number> = { 
      idle: -1, extract: 0, clean: 1, chunk: 2, embed: 3, index: 4, complete: 5 
    };
    
    const currentIndex = stepIndices[currentStep] || 0;
    const thisIndex = stepIndices[stepId];

    if (thisIndex < currentIndex) return 'completed';
    if (thisIndex === currentIndex) return 'active';
    return 'pending';
  };

  const isAllComplete = currentStep === 'complete';
  const isError = currentStep === 'error';

  return (
    <div className="w-full mb-1">
      <AnimatePresence mode="wait">
        
        {/* === STATE 1: HIDDEN (Compact Row with Actions) === */}
        {isHidden ? (
          <motion.div
            key="hidden"
            initial={{ opacity: 0, height: 'auto' }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center justify-between p-2 rounded-md bg-transparent hover:bg-[#111] border border-transparent hover:border-white/5 transition-all group"
          >
            <div className="flex items-center gap-2.5 min-w-0 flex-1 cursor-pointer" onClick={() => setIsHidden(false)}>
              {/* Status Indicator */}
              {isAllComplete ? (
                 <FileText size={14} className="text-zinc-600 group-hover:text-zinc-500 shrink-0" />
              ) : (
                <div className={`w-3 h-3 rounded-full flex items-center justify-center ${isError ? 'bg-red-500/20' : 'bg-blue-500/20'}`}>
                    <div className={`w-1.5 h-1.5 rounded-full ${isError ? 'bg-red-500' : 'bg-blue-500 animate-pulse'}`} />
                </div>
              )}
              
              <span className={`text-[13px] truncate transition-colors ${isAllComplete ? 'text-zinc-400 group-hover:text-zinc-300' : 'text-zinc-200'}`}>
                {fileName}
              </span>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-1 ">
              <button 
                onClick={() => setIsHidden(false)}
                className="p-1.5 text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10 rounded transition-colors"
                title="View Pipeline"
              >
                <Eye size={13} />
              </button>
              
              {onDelete && (
                <button 
                  onClick={(e) => { e.stopPropagation(); onDelete(); }}
                  className="p-1.5 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded transition-colors"
                  title="Delete File"
                >
                  <Trash2 size={13} />
                </button>
              )}
            </div>
          </motion.div>
        ) : (

        /* === STATE 2: EXPANDED (Full Pipeline) === */
          <motion.div 
            key="expanded"
            initial={{ opacity: 0, scale: 1 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.01 }}
            transition={{ duration: 0.1 }}
            className="bg-[#0A0A0A] border border-white/10 rounded-lg p-3 overflow-hidden relative shadow-lg shadow-black/50 my-2 mx-1"
          >
            {/* Header */}
            <div className="flex items-center gap-3 select-none mb-3">
              <div className={`p-1.5 rounded-md ${isAllComplete ? 'bg-emerald-500/10 text-emerald-400' : 'bg-zinc-800 text-zinc-400'}`}>
                <FileText size={14} />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-zinc-200 truncate">
                    {fileName}
                  </span>
                </div>
                <p className="text-[10px] text-zinc-500 font-medium uppercase tracking-wider">
                  {isAllComplete ? <span className="text-emerald-500">Ready</span> : 
                   isError ? <span className="text-red-500">Failed</span> : 
                   'Processing...'}
                </p>
              </div>

              {/* Hide Button */}
              <button 
                onClick={() => setIsHidden(true)}
                className="p-1.5 text-zinc-500 hover:text-zinc-300 hover:bg-white/5 rounded-md transition-colors"
              >
                <EyeOff size={14} />
              </button>
            </div>

            {/* Steps List */}
            <div className="space-y-3 relative ml-1">
              <div className="absolute left-[11px] top-2 bottom-2 w-0.5 bg-white/5 z-0" />
              
              {steps.map((step) => {
                const status = getStepStatus(step.id);
                const Icon = step.icon;
                const isCompleted = status === 'completed' || isAllComplete;
                const isActive = status === 'active' && !isAllComplete;
                
                return (
                  <div key={step.id} className="relative z-10 flex items-center gap-3">
                    <div className={`
                      w-6 h-6 rounded-full flex items-center justify-center border transition-all duration-500
                      ${isActive ? 'bg-blue-500/10 border-blue-500/50 text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.2)]' : ''}
                      ${isCompleted ? 'bg-emerald-500/10 border-emerald-500/50 text-emerald-400' : ''}
                      ${!isActive && !isCompleted ? 'bg-[#111] border-white/5 text-zinc-700' : ''}
                    `}>
                      {isActive ? <Loader2 size={10} className="animate-spin" /> : 
                       isCompleted ? <CheckCircle2 size={10} /> : <Icon size={10} />}
                    </div>

                    <span className={`text-[11px] font-medium transition-colors duration-300 ${
                      isActive ? 'text-blue-400' : 
                      isCompleted ? 'text-emerald-500/80' : 'text-zinc-600'
                    }`}>
                      {step.label}
                    </span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};