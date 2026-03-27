import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  LabelList,
} from 'recharts';

interface ShapChartProps {
  shapValues: Record<string, number>;
  topFeatures: string[];
  featureValues: Record<string, number>;
  /** Number of remaining features not shown */
  maxFeatures?: number;
}

/** Format large numbers for compact display */
function fmtVal(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 1e12) return (v / 1e12).toFixed(1) + 'T';
  if (abs >= 1e9) return (v / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (v / 1e3).toFixed(1) + 'K';
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(3);
}

const ShapChart: React.FC<ShapChartProps> = ({
  shapValues,
  topFeatures,
  featureValues,
  maxFeatures = 15,
}) => {
  const shown = topFeatures.slice(0, maxFeatures);
  const remaining = Object.keys(shapValues).length - shown.length;

  const chartData = shown.map((feature) => {
    const shap = shapValues[feature] ?? 0;
    const raw = featureValues[feature];
    const rawLabel = raw !== undefined ? fmtVal(raw) : '?';
    return {
      label: `${rawLabel} = ${feature}`,
      feature,
      value: shap,
      rawValue: raw,
    };
  });

  // Add "N other features" row if there are remaining features
  if (remaining > 0) {
    const otherShap = Object.entries(shapValues)
      .filter(([k]) => !shown.includes(k))
      .reduce((sum, [, v]) => sum + v, 0);
    chartData.push({
      label: `${remaining} other features`,
      feature: '__other__',
      value: otherShap,
      rawValue: 0,
    });
  }

  const barHeight = 28;
  const chartHeight = Math.max(280, chartData.length * barHeight + 60);

  return (
    <div>
      <div className="mb-2 flex gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-red-500" />
          推向詐欺（正值）
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-blue-500" />
          推向正常（負值）
        </span>
      </div>
      <div style={{ width: '100%', height: chartHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 8, right: 64, left: 8, bottom: 8 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: '#64748b' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={220}
              tick={{ fontSize: 11, fill: '#64748b' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              formatter={(value: number, _name: string, props: { payload: { feature: string; rawValue: number } }) => {
                const raw = props.payload.rawValue;
                return [
                  `SHAP: ${value >= 0 ? '+' : ''}${value.toFixed(4)}　原始值: ${raw !== undefined ? fmtVal(raw) : 'N/A'}`,
                  props.payload.feature,
                ];
              }}
              contentStyle={{
                borderRadius: '8px',
                border: '1px solid #e2e8f0',
                boxShadow: '0 4px 12px -2px rgb(0 0 0 / 0.08)',
                fontSize: '13px',
              }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} name="SHAP">
              <LabelList
                dataKey="value"
                position="right"
                formatter={(v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`}
                style={{ fontSize: 11, fontWeight: 600 }}
                fill="#64748b"
              />
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.value >= 0 ? '#ef4444' : '#3b82f6'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ShapChart;
