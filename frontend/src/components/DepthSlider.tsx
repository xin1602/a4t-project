import React from 'react';

interface DepthSliderProps {
  value: number;
  onChange: (value: number) => void;
}

const DepthSlider: React.FC<DepthSliderProps> = ({ value, onChange }) => {
  return (
    <div>
      <div className="flex items-center gap-3">
        <span className="whitespace-nowrap text-sm font-medium text-slate-600">展開深度</span>
        <input
          type="range"
          min={1}
          max={5}
          step={1}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-surface-200 accent-brand-600"
        />
        <span className="min-w-[40px] text-center text-sm font-bold text-brand-600">
          {value} hop
        </span>
      </div>
      <div className="mt-1 flex justify-between px-[72px] text-[11px] text-slate-400">
        {[1, 2, 3, 4, 5].map((n) => (
          <span key={n}>{n}</span>
        ))}
      </div>
    </div>
  );
};

export default DepthSlider;
