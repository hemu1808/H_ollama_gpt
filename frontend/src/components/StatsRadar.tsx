import React from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

const data = [
  { subject: 'Relevance', A: 120, fullMark: 150 },
  { subject: 'Accuracy', A: 98, fullMark: 150 },
  { subject: 'Latency', A: 86, fullMark: 150 },
  { subject: 'Depth', A: 99, fullMark: 150 },
  { subject: 'Context', A: 85, fullMark: 150 },
];

export const StatsRadar = () => {
  return (
    <div className="h-64 w-full relative">
      <div className="absolute inset-0 bg-accent/5 blur-3xl rounded-full" />
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: '#A1A1AA', fontSize: 10 }} />
          <PolarRadiusAxis angle={30} domain={[0, 150]} tick={false} axisLine={false} />
          <Radar
            name="Mike"
            dataKey="A"
            stroke="#3B82F6"
            strokeWidth={2}
            fill="#3B82F6"
            fillOpacity={0.2}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};