import React, { useMemo, useState } from 'react';

interface HeatmapChartProps {
  data: Array<{ hour: number; day: number; count: number }>;
}

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const HOURS = Array.from({ length: 24 }, (_, i) => i);
const CELL = 26;
const GAP = 3;
const LEFT = 40;
const TOP = 22;

/** Night-time hours relevant to fraud detection (22:00–05:59) */
function isNightHour(h: number): boolean {
  return h >= 22 || h < 6;
}

/** Interpolate from slate-100 → amber-200 → red-500 */
function cellColor(count: number, max: number): string {
  if (count === 0) return '#f8fafc'; // slate-50
  const t = Math.min(count / max, 1);
  if (t <= 0.5) {
    // slate-100 → amber-200
    const s = t * 2;
    const r = Math.round(241 + s * (253 - 241));
    const g = Math.round(245 + s * (230 - 245));
    const b = Math.round(249 + s * (138 - 249));
    return `rgb(${r},${g},${b})`;
  }
  // amber-200 → red-500
  const s = (t - 0.5) * 2;
  const r = Math.round(253 + s * (239 - 253));
  const g = Math.round(230 + s * (68 - 230));
  const b = Math.round(138 + s * (68 - 138));
  return `rgb(${r},${g},${b})`;
}

function textColor(count: number, max: number): string {
  if (count === 0) return 'transparent';
  return count / max > 0.55 ? '#fff' : '#64748b';
}

const HeatmapChart: React.FC<HeatmapChartProps> = ({ data }) => {
  const [hover, setHover] = useState<{ day: number; hour: number; count: number; x: number; y: number } | null>(null);

  const countMap = useMemo(() => {
    const map: Record<string, number> = {};
    for (const e of data) map[`${e.day}-${e.hour}`] = e.count;
    return map;
  }, [data]);

  const maxCount = useMemo(() => Math.max(1, ...data.map((d) => d.count)), [data]);

  const totalW = LEFT + HOURS.length * (CELL + GAP);
  const totalH = TOP + DAYS.length * (CELL + GAP) + 36; // extra for legend

  // Night-time total
  const nightTotal = useMemo(
    () => data.filter((d) => isNightHour(d.hour)).reduce((s, d) => s + d.count, 0),
    [data],
  );
  const allTotal = useMemo(() => data.reduce((s, d) => s + d.count, 0), [data]);

  return (
    <div className="relative">
      {/* Night ratio badge */}
      {allTotal > 0 && (
        <div className="mb-2 flex items-center gap-2 text-xs text-slate-500">
          <span className="inline-block h-2.5 w-2.5 rounded-sm border border-red-400 bg-red-50" />
          夜間時段（22–06）：{nightTotal} 筆（{Math.round((nightTotal / allTotal) * 100)}%）
        </div>
      )}

      <svg width={totalW} height={totalH} className="select-none">
        {/* Hour labels */}
        {HOURS.map((h) => (
          <text
            key={`h-${h}`}
            x={LEFT + h * (CELL + GAP) + CELL / 2}
            y={14}
            textAnchor="middle"
            className="fill-slate-400"
            style={{ fontSize: 10 }}
          >
            {h % 3 === 0 ? `${h}` : ''}
          </text>
        ))}

        {/* Day rows */}
        {DAYS.map((dayLabel, dayIdx) => (
          <g key={dayIdx}>
            {/* Day label */}
            <text
              x={LEFT - 6}
              y={TOP + dayIdx * (CELL + GAP) + CELL / 2 + 4}
              textAnchor="end"
              className="fill-slate-500"
              style={{ fontSize: 11 }}
            >
              {dayLabel}
            </text>

            {/* Cells */}
            {HOURS.map((hour) => {
              const count = countMap[`${dayIdx}-${hour}`] ?? 0;
              const x = LEFT + hour * (CELL + GAP);
              const y = TOP + dayIdx * (CELL + GAP);
              const night = isNightHour(hour);

              return (
                <g
                  key={hour}
                  onMouseEnter={() => {
                    setHover({
                      day: dayIdx, hour, count,
                      x: x + CELL / 2,
                      y: y - 4,
                    });
                  }}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: 'default' }}
                >
                  <rect
                    x={x} y={y}
                    width={CELL} height={CELL}
                    rx={4} ry={4}
                    fill={cellColor(count, maxCount)}
                    stroke={night ? '#f87171' : '#e2e8f0'}
                    strokeWidth={night ? 1.5 : 0.5}
                    strokeDasharray={night && count === 0 ? '2,2' : undefined}
                  />
                  {count > 0 && (
                    <text
                      x={x + CELL / 2}
                      y={y + CELL / 2 + 4}
                      textAnchor="middle"
                      fill={textColor(count, maxCount)}
                      style={{ fontSize: 9, fontWeight: 600, pointerEvents: 'none' }}
                    >
                      {count}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        ))}

        {/* Legend */}
        {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
          <rect
            key={i}
            x={LEFT + i * 20}
            y={TOP + DAYS.length * (CELL + GAP) + 10}
            width={16} height={12} rx={3}
            fill={cellColor(Math.round(t * maxCount), maxCount)}
            stroke="#e2e8f0" strokeWidth={0.5}
          />
        ))}
        <text
          x={LEFT - 2}
          y={TOP + DAYS.length * (CELL + GAP) + 20}
          textAnchor="end"
          className="fill-slate-400"
          style={{ fontSize: 10 }}
        >少</text>
        <text
          x={LEFT + 5 * 20 + 4}
          y={TOP + DAYS.length * (CELL + GAP) + 20}
          className="fill-slate-400"
          style={{ fontSize: 10 }}
        >多</text>
      </svg>

      {/* Tooltip */}
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-lg"
          style={{
            left: hover.x,
            top: hover.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <span className="font-semibold text-slate-700">{DAYS[hover.day]} {hover.hour}:00–{hover.hour}:59</span>
          <br />
          <span className="text-slate-500">{hover.count} 筆交易</span>
          {isNightHour(hover.hour) && (
            <span className="ml-1.5 rounded bg-red-50 px-1 py-0.5 text-[10px] text-red-500">夜間</span>
          )}
        </div>
      )}
    </div>
  );
};

export default HeatmapChart;
