import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { motion } from 'framer-motion';
import { Copy, Check, RefreshCw, Cpu, BookOpen, Clock } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  latency?: number;
  mode?: 'fast' | 'deep';
  thoughts?: string;
}

export const MessageBubble = ({ msg, onRegenerate }: { msg: Message, onRegenerate?: () => void }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // --- USER MESSAGE ---
  if (msg.role === 'user') {
    return (
      <motion.div 
        initial={{ opacity: 0, y: 10 }} 
        animate={{ opacity: 1, y: 0 }} 
        className="flex justify-end group"
      >
         <div className="bg-[#1A1A1A] border border-white/5 text-zinc-100 px-5 py-3 rounded-2xl rounded-tr-sm max-w-[85%] text-[15px] leading-relaxed shadow-sm">
            {msg.content}
         </div>
      </motion.div>
    );
  }

  // --- ASSISTANT MESSAGE ---
  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      className="flex gap-4 max-w-full group"
    >
      {/* Left Gutter (Optional decorative line or icon) */}
      <div className="w-8 shrink-0 flex flex-col items-center pt-1">
         <div className={`w-4 h-4 rounded-sm flex items-center justify-center shadow-lg shadow-white/5 ${msg.mode === 'deep' ? 'bg-purple-500/20' : 'bg-emerald-500/20'}`}>
            <div className={`w-1.5 h-1.5 rounded-full ${msg.mode === 'deep' ? 'bg-purple-400' : 'bg-emerald-400'}`} />
         </div>
         {/* Vertical line connector */}
         <div className="w-px h-full bg-white/5 mt-3 mb-2" />
      </div>

      <div className="flex-1 space-y-3 min-w-0">
        
        {/* Thoughts Block (Linear Style) */}
        {msg.thoughts && (
            <div className="bg-[#0D0D0D] border border-white/5 rounded-lg p-3 my-2">
                <div className="flex items-center gap-2 mb-2">
                    <Cpu size={12} className="text-purple-400" />
                    <span className="text-[11px] font-bold text-zinc-500 uppercase tracking-wider">Reasoning Trace</span>
                </div>
                <div className="text-[13px] text-zinc-400 leading-relaxed font-mono whitespace-pre-wrap pl-5 border-l border-white/5">
                    {msg.thoughts}
                </div>
            </div>
        )}

        {/* Markdown Content */}
        <div className="prose prose-invert prose-sm max-w-none text-zinc-200 prose-p:leading-7 prose-headings:font-semibold prose-headings:text-zinc-100 prose-pre:bg-[#0D0D0D] prose-pre:border prose-pre:border-white/5 prose-code:text-blue-300 prose-a:text-blue-400 hover:prose-a:text-blue-300">
          <ReactMarkdown components={{
             code({node, inline, className, children, ...props}: any) {
               const match = /language-(\w+)/.exec(className || '')
               return !inline && match ? (
                 <div className="relative group/code my-4">
                   <div className="absolute right-3 top-3 opacity-0 group-hover/code:opacity-100 transition-opacity z-10">
                      <button onClick={() => handleCopy(String(children))} className="p-1.5 bg-white/10 rounded hover:bg-white/20 transition-colors">
                         {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} className="text-zinc-400" />}
                      </button>
                   </div>
                   <div className="text-[10px] uppercase text-zinc-500 bg-[#151515] border-x border-t border-white/5 px-4 py-1.5 rounded-t-lg w-fit font-mono tracking-wider">
                      {match[1]}
                   </div>
                   <SyntaxHighlighter 
                      style={vscDarkPlus} 
                      language={match[1]} 
                      PreTag="div" 
                      {...props} 
                      customStyle={{
                        background:'#0D0D0D', 
                        fontSize:'13px', 
                        borderRadius:'0 8px 8px 8px', 
                        border:'1px solid rgba(255,255,255,0.05)',
                        margin: 0
                      }}
                   >
                     {String(children).replace(/\n$/, '')}
                   </SyntaxHighlighter>
                 </div>
               ) : ( 
                 <code className="bg-white/5 border border-white/5 px-1.5 py-0.5 rounded text-[13px] font-mono text-zinc-300" {...props}>
                    {children}
                 </code> 
               )
             }
          }}>
            {msg.content}
          </ReactMarkdown>
        </div>

        {/* Footer: Metadata */}
        <div className="flex flex-wrap items-center gap-3 mt-4 pt-1">
            {msg.sources && msg.sources.length > 0 && (
               <div className="flex flex-wrap gap-2">
                  {msg.sources.map((src, i) => (
                      <div key={i} className="flex items-center gap-1.5 px-2 py-1 bg-[#111] border border-white/5 rounded text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors cursor-default">
                          <BookOpen size={10} />
                          <span className="truncate max-w-[150px]">{src}</span>
                      </div>
                  ))}
               </div>
            )}
            
            <div className="ml-auto flex items-center gap-3">
               {msg.latency && (
                  <div className="flex items-center gap-1 text-[10px] text-zinc-600 font-mono">
                     <Clock size={10} />
                     <span>{msg.latency.toFixed(2)}s</span>
                  </div>
               )}
               <button 
                  onClick={onRegenerate} 
                  className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-white/5 rounded text-zinc-500 hover:text-zinc-300 transition-all"
                  title="Regenerate response"
               >
                  <RefreshCw size={12} />
               </button>
            </div>
        </div>
      </div>
    </motion.div>
  );
};