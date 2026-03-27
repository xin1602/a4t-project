import React from 'react';

interface KpiCardProps {
  title: string;
  value: number | null | undefined;
  unit?: string;
  status?: string;
}

const KpiCard: React.FC<KpiCardProps> = ({ title, value, unit, status }) => {
  const isEmpty = value === null || value === undefined;

  return (
    <div className="card min-w-[160px] flex-1">
      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-400">
        {title}
      </p>
      <p className={`text-2xl font-bold ${isEmpty ? 'text-slate-300' : 'text-slate-800'}`}>
        {isEmpty ? (
          '尚未執行'
        ) : (
          <>
            {value}
            {unit && <span className="ml-1 text-sm font-normal text-slate-400">{unit}</span>}
          </>
        )}
      </p>
      {status && !isEmpty && (
        <p className="mt-1.5 text-xs text-slate-400">{status}</p>
      )}
    </div>
  );
};

export default KpiCard;
