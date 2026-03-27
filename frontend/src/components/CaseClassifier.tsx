import React, { useState } from 'react';
import { updateCase } from '../api/client';

interface CaseClassifierProps {
  caseId: string;
  currentStatus: string;
  onStatusChange: (newStatus: string) => void;
}

const STATUS_BUTTONS = [
  { label: '加入黑名單', value: 'blacklisted', bg: 'bg-red-500 hover:bg-red-600', ring: 'focus:ring-red-400' },
  { label: '持續觀察', value: 'watching', bg: 'bg-amber-500 hover:bg-amber-600', ring: 'focus:ring-amber-400' },
  { label: '正常用戶', value: 'normal', bg: 'bg-emerald-500 hover:bg-emerald-600', ring: 'focus:ring-emerald-400' },
];

const STATUS_LABELS: Record<string, string> = {
  pending: '待審查',
  blacklisted: '黑名單',
  watching: '持續觀察',
  normal: '正常用戶',
};

const CaseClassifier: React.FC<CaseClassifierProps> = ({
  caseId,
  currentStatus,
  onStatusChange,
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClassify = async (newStatus: string) => {
    if (newStatus === currentStatus) return;
    setLoading(true);
    setError(null);
    try {
      await updateCase(caseId, newStatus, 'analyst');
      onStatusChange(newStatus);
    } catch {
      setError('分類儲存失敗，請重試');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed right-6 top-1/2 z-50 w-44 -translate-y-1/2 rounded-2xl border border-surface-200 bg-white p-5 shadow-panel">
      <p className="mb-2 text-sm font-semibold text-slate-700">案件分類</p>
      <div className="mb-4 rounded-lg bg-surface-50 px-3 py-1.5 text-center text-xs text-slate-500">
        目前：{STATUS_LABELS[currentStatus] ?? currentStatus}
      </div>

      <div className="flex flex-col gap-2">
        {STATUS_BUTTONS.map((btn) => {
          const disabled = loading || btn.value === currentStatus;
          return (
            <button
              key={btn.value}
              className={`w-full rounded-lg px-3 py-2 text-sm font-medium text-white shadow-sm transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-offset-2 ${btn.bg} ${btn.ring} ${disabled ? 'cursor-not-allowed opacity-40' : ''}`}
              disabled={disabled}
              onClick={() => handleClassify(btn.value)}
            >
              {btn.label}
            </button>
          );
        })}
      </div>

      {loading && (
        <p className="mt-3 text-center text-xs text-slate-400">儲存中…</p>
      )}
      {error && (
        <p className="mt-3 text-center text-xs text-red-500">{error}</p>
      )}
    </div>
  );
};

export default CaseClassifier;
