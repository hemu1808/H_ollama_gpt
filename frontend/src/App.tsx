import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Zap } from 'lucide-react';

// --- COMPONENTS ---
import { Sidebar } from './components/Sidebar';
import { MessageBubble, type Message } from './components/MessageBubble';
import { ChatInput } from './components/ChatArea'; // Now Modularized
import { ParticleBackground } from './components/ParticleBackground';
import { StatsRadar } from './components/StatsRadar';

const API_URL = "http://localhost:8000";

interface Session {
  id: string;
  title: string;
  messages: Message[];
  date: string;
}

const App: React.FC = () => {
  // --- STATE MANAGEMENT ---
  const [sessions, setSessions] = useState<Session[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState("");
  const [input, setInput] = useState('');
  const [deepMode, setDeepMode] = useState(false);
  const [documents, setDocuments] = useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = useState({ state: 'idle', message: '' });
  const [isBackendOnline, setIsBackendOnline] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // --- INITIALIZATION & PERSISTENCE ---
  useEffect(() => {
    // 1. Recover Session
    const saved = localStorage.getItem('chat_sessions');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSessions(parsed);
        if (parsed.length > 0) setCurrentSessionId(parsed[0].id);
        else createNewSession();
      } catch (e) {
        console.error("Session Corrupt:", e);
        createNewSession();
      }
    } else {
      createNewSession();
    }
    // 2. Fetch Docs
    fetchDocuments();
  }, []);

  useEffect(() => {
    if (sessions.length > 0) {
      localStorage.setItem('chat_sessions', JSON.stringify(sessions));
    }
  }, [sessions]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [sessions, currentSessionId, loading]);

  //connection status checker
  useEffect(() => {
    const checkStatus = async () => {
      try {
        // Replace with your actual backend URL
        const response = await fetch("http://localhost:8000/health"); 
        if (response.ok) {
          setIsBackendOnline(true);
        } else {
          setIsBackendOnline(false);
        }
      } catch (error) {
        setIsBackendOnline(false);
      }
    };

    checkStatus(); // Check immediately
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  // --- LOGIC HANDLERS ---
  const getCurrentMessages = () => {
    return sessions.find(s => s.id === currentSessionId)?.messages || [];
  };

  const createNewSession = () => {
    const newId = Date.now().toString();
    const newSession: Session = {
      id: newId,
      title: "New Conversation",
      messages: [{ id: Date.now(), role: 'assistant', content: "Systems ready. How can I help you analyze your data?" }],
      date: new Date().toLocaleDateString()
    };
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newId);
  };

  const updateCurrentSessionMessages = (newMsg: Message) => {
    setSessions(prev => prev.map(s => {
      if (s.id === currentSessionId) {
        let newTitle = s.title;
        // Auto-rename on first user message
        if (s.title === "New Conversation" && newMsg.role === 'user') {
          newTitle = newMsg.content.slice(0, 30) + (newMsg.content.length > 30 ? "..." : "");
        }
        return { ...s, title: newTitle, messages: [...s.messages, newMsg] };
      }
      return s;
    }));
  };

  const clearUploadStatus = () => {
    setUploadStatus({ state: 'idle', message: '' });
  };

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_URL}/documents`);
      setDocuments(res.data);
    } catch (e) { console.error("Backend Offline"); }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    
    // Init Abort Controller
    abortControllerRef.current = new AbortController();

    const userMsg: Message = { id: Date.now(), role: 'user', content: input };
    updateCurrentSessionMessages(userMsg);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          question: userMsg.content, 
          top_k: 4, 
          mode: deepMode ? 'deep' : 'fast' 
        }),
        signal: abortControllerRef.current.signal,
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.replace('data: ', ''));
              
              if (data.type === 'status') {
                setLoadingStep(data.content);
              } else if (data.type === 'result') {
                const aiMsg: Message = {
                  id: Date.now() + 1,
                  role: 'assistant',
                  content: data.answer,
                  sources: data.sources,
                  mode: deepMode ? 'deep' : 'fast',
                  thoughts: data.thoughts,
                  latency: data.processing_time
                };
                updateCurrentSessionMessages(aiMsg);
              }
            } catch (e) { console.error("SSE Error", e); }
          }
        }
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        updateCurrentSessionMessages({ 
          id: Date.now(), role: 'assistant', content: "**Network Error**: Connection lost." 
        });
      }
    }
    setLoading(false);
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const file = e.target.files[0];
    let isExplicitlyComplete = false;
    setUploadStatus({ state: 'uploading', message: 'extract' });
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.body) throw new Error("No response body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || ''; 
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.replace('data: ', '');
            try {
              const data = JSON.parse(jsonStr);
              if (data.error) {
                console.error("Stream Error:", data.error);
                setUploadStatus({ state: 'error', message: 'Failed' });
                return;
              }
              if (data.step) {
                setUploadStatus({ state: 'uploading', message: data.step });                
                if (data.step === 'complete') {
                  setUploadStatus({ state: 'success', message: 'complete' });
                  await fetchDocuments(); 
                }
              }
            } catch (err) {
              console.error("JSON Parse Error", err);
            }
          }
        }
      }
      if (!isExplicitlyComplete) {
         setUploadStatus({ state: 'success', message: 'complete' });
         await fetchDocuments();
      }
    } catch (error) {
      console.error("Upload failed", error);
      setUploadStatus({ state: 'error', message: 'Upload Failed' });
    } finally {
        e.target.value = '';
    }
  };

  const handleDeleteFile = async (fileName: string) => {
    setDocuments((prevDocs) => prevDocs.filter((doc) => doc !== fileName));
    try {
      const response = await fetch(`${API_URL}/documents/delete_file/${fileName}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Backend delete failed');
      }
    } catch (error) {
      console.error("Error deleting file:", error);
      setDocuments((prevDocs) => [...prevDocs, fileName]);
      alert("Failed to delete file from server");
    }
  };


  const handleDeleteSession = (id: string) => {
    if (window.confirm("Delete this chat?")) {
      const newSessions = sessions.filter(s => s.id !== id);
      setSessions(newSessions);
      if (newSessions.length === 0) createNewSession();
      else if (currentSessionId === id) setCurrentSessionId(newSessions[0].id);
    }
  };

  const handleClear = () => {
    if (window.confirm("Factory Reset: Clear all chats and history?")) {
      localStorage.removeItem('chat_sessions');
      setSessions([]);
      createNewSession();
    }
  };

  // --- RENDER ---
  const currentSession = sessions.find(s => s.id === currentSessionId);

  return (
    <div className="flex h-screen bg-[#050505] text-white font-sans overflow-hidden">
      {/* 1. LAYOUT BACKGROUNDS */}
      <div className="grid-bg" /> {/* Defined in index.css */}
      <ParticleBackground />
      
      {/* 2. SIDEBAR */}
      <Sidebar 
        sessions={sessions}
        currentSessionId={currentSessionId}
        documents={documents} 
        uploadStatus={uploadStatus} 
        onUpload={handleUpload} 
        onDeleteFile={handleDeleteFile}
        onClear={handleClear} 
        onNewChat={createNewSession}
        onSwitchSession={setCurrentSessionId}
        onDeleteSession={handleDeleteSession}
        isBackendOnline={isBackendOnline}
        //onClear={handleResetChat}
        onClearUpload={clearUploadStatus}
      />

      {/* 3. MAIN CONTENT */}
      <main className="flex-1 flex flex-col relative min-w-0 z-10">
        
        {/* HEADER */}
        <header className="h-14 border-b border-white/5 bg-[#050505]/60 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-30">
          <div className="flex items-center gap-2">
            <span className="text-zinc-500 text-sm">Workspace /</span>
            <span className="font-medium text-sm text-zinc-200">{currentSession?.title || "Draft"}</span>
          </div>

          <div className="flex items-center p-1 bg-white/5 border border-white/5 rounded-full">
          {/* Fast Mode Button */}
          <button
          onClick={() => setDeepMode(false)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300 ${
            !deepMode
            ? 'bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/50 shadow-[0_0_12px_rgba(16,185,129,0.3)]'
            : 'text-zinc-500 hover:text-zinc-300'
          }`}
          >
            <Activity size={13} />
            <span>Fast</span>
            </button>
            {/* Deep Mode Button */}
            <button
            onClick={() => setDeepMode(true)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300 ${
              deepMode
              ? 'bg-purple-500/20 text-purple-400 ring-1 ring-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.3)]'
              : 'text-zinc-500 hover:text-zinc-300'
            }`}
            >
              <Zap size={13} />
              <span>Deep</span>
              </button>
              </div>
        </header>

        {/* CHAT MESSAGES */}
        <div className="flex-1 overflow-y-auto relative scroll-smooth no-scrollbar">
          <div className="max-w-5xl mx-auto w-full px-4 py-12 space-y-10">
            <AnimatePresence mode="popLayout">
              {getCurrentMessages().map(msg => (
                <MessageBubble 
                  key={msg.id} 
                  msg={msg} 
                  onRegenerate={handleSend} 
                />
              ))}
            </AnimatePresence>
            
            {/* Thinking Step Indicator */}
            {loading && (
               <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-3 pl-1">
                  <div className="w-4 h-4 rounded-full border-2 border-zinc-700 border-t-zinc-300 animate-spin" />
                  <span className="text-xs text-zinc-500 font-mono animate-pulse">{loadingStep || "Processing..."}</span>
               </motion.div>
            )}
            
            {/* Invisble div for auto-scroll */}
            <div ref={messagesEndRef} className="h-4" />
          </div>

          {/* Background Flair: Radar Chart */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] pointer-events-none -z-10 opacity-[0.03] blur-sm">
             <StatsRadar />
          </div>
        </div>
        {/* 
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-0">
            <GlowCard>
               <div className="w-[300px] h-[200px]" /> 
            </GlowCard>
        </div> */}

        {/* INPUT AREA */}
        <ChatInput 
          input={input}
          setInput={setInput}
          onSend={handleSend}
          loading={loading}
          onStop={handleStop}
          deepMode={deepMode}
          setDeepMode={setDeepMode}
          onUpload={handleUpload}
        />
      </main>
    </div>
  );
};

export default App;