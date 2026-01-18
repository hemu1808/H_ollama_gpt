import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { 
  Plus, Upload, MessageSquare, Trash2, Settings
} from 'lucide-react';
import { PipelineStatus } from './PipelineStatus';

interface SidebarProps {
  sessions: any[];
  currentSessionId: string;
  documents: string[];
  uploadStatus: { state: string; message: string };
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onClear: () => void;       
  onClearUpload: () => void; 
  onNewChat: () => void;
  onSwitchSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onDeleteFile?: (fileName: string) => void; // Added this prop
  isBackendOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  sessions, currentSessionId, documents, uploadStatus, 
  onUpload, onClear, onClearUpload, onNewChat, onSwitchSession, onDeleteSession, onDeleteFile, isBackendOnline
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const getActiveFileName = () => {
    if (fileInputRef.current?.files?.[0]) {
      return fileInputRef.current.files[0].name;
    }
    return "Document.pdf";
  };

  return (
    <aside className="w-[320px] h-screen bg-[#020202] border-r border-white/5 flex flex-col z-20 shrink-0 select-none">
      <input type="file" hidden ref={fileInputRef} accept=".pdf" onChange={onUpload} />
      
      {/* 1. Header */}
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between px-2">
           <div className="flex items-center gap-2">
              <div className="w-5 h-5 rounded bg-white flex items-center justify-center shadow-lg shadow-white/10">
                 <div className="w-2.5 h-2.5 bg-black rounded-sm" />
              </div>
              <span className="font-semibold text-sm tracking-tight text-white">H_GPT</span>
           </div>
           <div className="flex items-center gap-2">
             <span className="text-[10px] font-mono text-zinc-600 bg-white/5 px-1.5 py-0.5 rounded">PRO</span>
           </div>
        </div>

        <button 
          onClick={onNewChat}
          className="group w-full flex items-center gap-2 px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/5 rounded-md transition-all text-sm font-medium text-zinc-200 active:scale-[0.98]"
        >
          <Plus size={14} className="text-zinc-400 group-hover:text-white" />
          <span>New Thread</span>
          <div className="ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
             <kbd className="px-1.5 py-0.5 rounded bg-black border border-white/10 text-[10px] text-zinc-500 font-sans">C</kbd>
          </div>
        </button>
      </div>

      {/* 2. Navigation List */}
      <div className="flex-1 overflow-y-auto px-2 space-y-6 no-scrollbar">
        
        {/* History Section */}
        <div>
           <div className="px-3 mb-2 flex items-center justify-between group/header">
             <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Recent</span>
           </div>
           <div className="space-y-0.5">
             {sessions.map((session) => (
               <button 
                 key={session.id}
                 onClick={() => onSwitchSession(session.id)}
                 className={`group w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-[13px] transition-all relative ${
                   session.id === currentSessionId 
                     ? 'bg-[#151515] text-white font-medium' 
                     : 'text-zinc-400 hover:bg-[#111] hover:text-zinc-200'
                 }`}
               >
                 <MessageSquare size={14} className={`shrink-0 ${session.id === currentSessionId ? 'text-white' : 'text-zinc-600 group-hover:text-zinc-500'}`} />
                 <span className="truncate flex-1 text-left">{session.title}</span>
                 
                 <div 
                   onClick={(e) => { e.stopPropagation(); onDeleteSession(session.id); }}
                   className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition-opacity"
                 >
                   <Trash2 size={12} className="text-zinc-500 hover:text-red-400" />
                 </div>
                 
                 {session.id === currentSessionId && (
                   <motion.div layoutId="active-nav" className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-white rounded-r-full" />
                 )}
               </button>
             ))}
           </div>
        </div>

        {/* Knowledge Base Section */}
        <div>
           <div className="px-3 mb-2 flex items-center justify-between group cursor-pointer" onClick={() => fileInputRef.current?.click()}>
             <span className="text-[11px] font-medium text-zinc-500 uppercase tracking-wider">Sources</span>
             <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-white/5 rounded">
                <span className="text-[10px] text-zinc-500">Add</span>
                <Upload size={10} className="text-zinc-500" />
             </div>
           </div>
           
           <div className="px-1">
             {/* 1. Render Active Upload First (if any) */}
             {(uploadStatus.state === 'uploading' || uploadStatus.state === 'error') && (
               <PipelineStatus 
                 currentStep={uploadStatus.state === 'error' ? 'error' : (uploadStatus.message as any)}
                 fileName={getActiveFileName()}
                 onDelete={onClearUpload} // Allow canceling upload
               />
             )}

             {/* 2. Render Existing Documents using PipelineStatus for consistent UI */}
             {documents.map((doc, i) => (
               <PipelineStatus 
                 key={i}
                 fileName={doc}
                 currentStep="complete" 
                 onDelete={onDeleteFile ? () => onDeleteFile(doc) : undefined}
               />
             ))}

             {/* 3. Empty State / Upload Trigger */}
             {uploadStatus.state !== 'uploading' && (
                <div onClick={() => fileInputRef.current?.click()} className="mx-2 px-3 py-6 border border-dashed border-white/5 rounded-lg flex flex-col items-center justify-center gap-2 cursor-pointer hover:bg-white/5 hover:border-white/10 transition-all group mt-2">
                   <div className="w-8 h-8 rounded-full bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform">
                      <Upload size={14} className="text-zinc-600 group-hover:text-zinc-400" />
                   </div>
                   <span className="text-[11px] text-zinc-600 group-hover:text-zinc-500">Upload PDF Context</span>
                </div>
             )}
           </div>
        </div>
      </div>

      {/* 3. Footer */}
      <div className="p-3 border-t border-white/5 flex items-center gap-2">
         {/* Reset Button */}
         <button 
            onClick={onClear} 
            className="flex-1 flex items-center gap-2 px-3 py-2 rounded-md hover:bg-[#111] text-zinc-500 hover:text-zinc-300 transition-colors text-[13px] border border-transparent hover:border-white/5"
         >
            <Settings size={14} className="group-hover:rotate-45 transition-transform duration-500" />
            <span>Reset App</span>
         </button>

         {/* System Status */}
         <div className="flex items-center gap-2 px-2.5 py-2 bg-black/20 border border-white/5 rounded-md shrink-0">
             <div 
               className={`w-1.5 h-1.5 rounded-full transition-colors duration-500 ${
                 isBackendOnline ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]'
               }`} 
             />
             <span className={`text-[10px] font-medium transition-colors duration-500 ${isBackendOnline ? 'text-emerald-500/80' : 'text-red-500/80'}`}>
               {isBackendOnline ? 'Online' : 'Offline'}
             </span>
         </div>
      </div>
    </aside>
  );
};