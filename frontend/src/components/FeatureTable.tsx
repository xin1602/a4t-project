import React from 'react';

interface FeatureStat {
  type: 'continuous' | 'categorical';
  mean?: number; std?: number; p25?: number; p50?: number; p75?: number;
  distribution?: Record<string, number>; total?: number;
}

interface FeatureTableProps {
  topFeatures: string[];
  shapValues: Record<string, number>;
  featureValues: Record<string, number>;
  featureStats: Record<string, FeatureStat>;
  maxRows?: number;
}

const FEATURE_LABELS: Record<string, string> = {
  sex: '性別',
  age: '年齡',
  career: '職業代碼',
  twd_inflow_sum: 'TWD 流入總額',
  twd_outflow_sum: 'TWD 流出總額',
  twd_amount_std: 'TWD 金額標準差',
  twd_night_tx_ratio: 'TWD 夜間交易比例',
  twd_to_crypto_ratio: 'TWD→加密比例',
  crypto_inflow_sum: '加密貨幣流入總額',
  crypto_outflow_sum: '加密貨幣流出總額',
  crypto_to_fiat_ratio: '加密→法幣比例',
  crypto_external_ratio: '外部轉帳比例',
  swap_total_twd_amount: '兌換 TWD 總額',
  swap_avg_twd_amount: '兌換 TWD 平均',
  swap_max_twd_amount: '兌換 TWD 最大',
  swap_to_crypto_ratio: '兌換→加密比例',
  trading_count: '交易筆數',
  trading_total_amount: '交易總額',
};

const SEX_MAP: Record<string, string> = { '1': '男', '2': '女' };

function displayValue(feature: string, val: number): string {
  if (feature === 'sex') return SEX_MAP[String(Math.round(val))] ?? String(val);
  const abs = Math.abs(val);
  if (abs >= 1e12) return (val / 1e12).toFixed(1) + 'T';
  if (abs >= 1e9) return (val / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (val / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (val / 1e3).toFixed(1) + 'K';
  if (Number.isInteger(val)) return val.toString();
  if (abs < 0.01 && abs > 0) return val.toFixed(4);
  return val.toFixed(2);
}

function comparisonLabel(
  feature: string,
  val: number,
  stat: FeatureStat,
): { text: string; color: string } {
  if (stat.type === 'categorical' && stat.distribution) {
    const key = String(Math.round(val));
    const frac = stat.distribution[key];
    if (frac !== undefined) {
      const pct = Math.round(frac * 100);
      const labelStr = feature === 'sex' ? (SEX_MAP[key] ?? key) : key;
      return {
        text: `${labelStr} 佔全體 ${pct}%`,
        color: pct < 30 ? 'text-amber-600' : 'text-slate-500',
      };
    }
    return { text: '—', color: 'text-slate-400' };
  }

  // Continuous
  const p25 = stat.p25 ?? 0;
  const p50 = stat.p50 ?? 0;
  const p75 = stat.p75 ?? 0;
  if (val > p75) return { text: '高於 75% 用戶', color: 'text-red-600' };
  if (val > p50) return { text: '高於中位數', color: 'text-amber-600' };
  if (val < p25) return { text: '低於 25% 用戶', color: 'text-blue-600' };
  return { text: '正常範圍', color: 'text-slate-500' };
}

const FeatureTable: React.FC<FeatureTableProps> = ({
  topFeatures,
  shapValues,
  featureValues,
  featureStats,
  maxRows = 10,
}) => {
  const rows = topFeatures.slice(0, maxRows);

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-surface-200 text-left">
            <th className="py-2 pr-2 font-semibold text-slate-500">特徵</th>
            <th className="py-2 px-2 font-semibold text-slate-500">數值</th>
            <th className="py-2 px-2 font-semibold text-slate-500">vs 全體</th>
            <th className="py-2 px-2 font-semibold text-slate-500">SHAP</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((f) => {
            const shap = shapValues[f] ?? 0;
            const raw = featureValues[f];
            const stat = featureStats[f];
            const label = FEATURE_LABELS[f] || f;
            const valStr = raw !== undefined ? displayValue(f, raw) : '—';

            const cmp = raw !== undefined && stat
              ? comparisonLabel(f, raw, stat)
              : null;

            return (
              <tr key={f} className="border-b border-surface-100">
                <td className="py-2 pr-2">
                  <span className="text-slate-700">{label}</span>
                  {label !== f && (
                    <span className="ml-1 text-[10px] text-slate-400">{f}</span>
                  )}
                </td>
                <td className="py-2 px-2 font-mono text-slate-600">{valStr}</td>
                <td className="py-2 px-2">
                  {cmp ? (
                    <span className={`text-[11px] font-medium ${cmp.color}`}>{cmp.text}</span>
                  ) : (
                    <span className="text-slate-400">—</span>
                  )}
                </td>
                <td className={`py-2 px-2 font-mono font-semibold ${shap >= 0 ? 'text-red-500' : 'text-blue-500'}`}>
                  {shap >= 0 ? '+' : ''}{shap.toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default FeatureTable;
