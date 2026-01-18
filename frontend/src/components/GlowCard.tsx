import React, { useRef, useState } from 'react';

export const GlowCard = ({ children, className = "" }: { children: React.ReactNode, className?: string }) => {
  const divRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setOpacity(1);
  };

  const handleMouseLeave = () => setOpacity(0);

  return (
    <div 
      ref={divRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className={`relative rounded-xl border border-white/10 bg-[#0A0A0A] overflow-hidden ${className}`}
    >
      {/* The Glow Overlay */}
      <div 
        className="pointer-events-none absolute -inset-px transition-opacity duration-300 z-0"
        style={{ 
          opacity,
          background: `radial-gradient(600px circle at ${position.x}px ${position.y}px, rgba(255,255,255,0.06), transparent 40%)` 
        }}
      />
      {/* Content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
};