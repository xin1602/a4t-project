import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface RiskHistogramProps {
  data: Array<{ bin: string; count: number }>;
}

const RiskHistogram: React.FC<RiskHistogramProps> = ({ data }) => {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
          <XAxis
            dataKey="bin"
            tick={{ fontSize: 12, fill: '#64748b' }}
            label={{ value: 'Risk Score', position: 'insideBottom', offset: -4, fontSize: 12, fill: '#94a3b8' }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 12, fill: '#64748b' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            formatter={(value: number) => [value, '用戶數']}
            labelFormatter={(label: string) => `區間：${label}`}
            contentStyle={{
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              boxShadow: '0 4px 12px -2px rgb(0 0 0 / 0.08)',
              fontSize: '13px',
            }}
          />
          <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} name="用戶數" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default RiskHistogram;
