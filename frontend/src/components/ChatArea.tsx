import React, { useRef, useEffect } from 'react';
import { Send, StopCircle, ArrowUp, Paperclip, Sparkles, Zap, Activity } from 'lucide-react';

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  onSend: () => void;
  loading: boolean;
  onStop: () => void;
  deepMode: boolean;
  setDeepMode: (mode: boolean) => void; // <--- NEW PROP
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void; // <--- NEW PROP
}

export const ChatInput: React.FC<ChatInputProps> = ({ 
  input, setInput, onSend, loading, onStop, deepMode, setDeepMode, onUpload 
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null); // <--- Ref for file upload

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !loading) onSend();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto relative z-40 px-4 pb-6">
      {/* Hidden File Input for the Attachment Button */}
      <input 
        type="file" 
        hidden 
        ref={fileInputRef} 
        accept=".pdf" 
        onChange={onUpload} 
      />

      <div className="relative group">
        {/* Ambient Glow Effect */}
        <div 
          className={`absolute -inset-0.5 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur-lg 
          ${deepMode ? 'bg-gradient-to-r from-purple-600 to-blue-600' : 'bg-gradient-to-r from-blue-600 to-cyan-600'}`} 
        />
        
        <div className="relative flex flex-col bg-[#0A0A0A] rounded-3xl border border-white/10 shadow-3xl overflow-hidden ring-1 ring-white/5 focus-within:ring-white/10 focus-within:border-white/20 transition-all">
          <textarea 
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={deepMode ? "Ask a complex reasoning question..." : "Ask H_GPT..."}
            className="w-full bg-transparent border-none text-zinc-200 px-4 py-4 focus:outline-none placeholder:text-zinc-600 resize-none min-h-[56px] max-h-[200px] text-[15px] leading-relaxed scrollbar-hide"
            rows={1}
          />
          
          <div className="flex items-center justify-between px-3 pb-3">
            <div className="flex items-center gap-3">
              {/* ATTACH BUTTON (Now functional) */}
              <button 
                onClick={() => fileInputRef.current?.click()} 
                className="p-2 hover:bg-white/5 rounded-lg text-zinc-500 hover:text-zinc-300 transition-colors group-hover:text-zinc-400"
                title="Attach Context"
              >
                <Paperclip size={18} />
              </button>

              {/* MODE SWITCH (New Segmented Control) */}
              <div className="hidden sm:flex items-center p-0.5 bg-white/5 border border-white/5 rounded-full">
                <button
                  onClick={() => setDeepMode(false)}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-medium transition-all ${
                    !deepMode
                      ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/50'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  <Activity size={10} />
                  <span>Fast</span>
                </button>
                <button
                  onClick={() => setDeepMode(true)}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-full text-[10px] font-medium transition-all ${
                    deepMode
                      ? 'bg-purple-500/20 text-purple-400 ring-1 ring-purple-500/50'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  <Zap size={10} />
                  <span>Deep</span>
                </button>
              </div>
            </div>
            
            <button 
              onClick={loading ? onStop : onSend}
              disabled={!input.trim() && !loading}
              className={`p-2 rounded-xl transition-all flex items-center justify-center gap-2 ${
                loading 
                  ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 ring-1 ring-red-500/50' 
                  : input.trim() 
                      ? (deepMode ? 'bg-purple-600 text-white shadow-lg shadow-purple-900/20 hover:bg-purple-500' : 'bg-white text-black hover:bg-zinc-200 shadow-lg shadow-white/10') 
                      : 'bg-zinc-800/50 text-zinc-600 cursor-not-allowed'
              }`}
            >
              {loading ? <StopCircle size={18} /> : <ArrowUp size={18} strokeWidth={2.5} />}
            </button>
          </div>
        </div>
        
        {/* Footer Text */}
        <div className="absolute -bottom-6 left-0 right-0 text-center flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity duration-500 delay-100">
          <Sparkles size={10} className="text-zinc-500" />
          <span className="text-[10px] text-zinc-500 tracking-wider font-medium">
            Double check for mistakes
          </span>
        </div>
      </div>
    </div>
  );
};