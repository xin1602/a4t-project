import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getOverview } from '../api/client';
import KpiCard from '../components/KpiCard';
import RiskHistogram from '../components/RiskHistogram';
import type { OverviewData } from '../types';

const Overview: React.FC = () => {
  const navigate = useNavigate();
  const [data, setData] = useState<OverviewData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getOverview()
      .then((res) => setData(res))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <div className="h-4 w-4 animate-spin-slow rounded-full border-2 border-surface-300 border-t-brand-600" />
          載入中…
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-10">
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
          載入失敗：{error}
        </div>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="mx-auto max-w-5xl px-6 py-8">
      <h2 className="mb-6 text-xl font-bold text-slate-800">風險總覽</h2>

      {/* KPI Cards */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-4">
        <KpiCard title="總用戶數" value={data.total_users} unit="人" />
        <KpiCard title="詐欺用戶" value={data.fraud_count} unit="人" />
        <KpiCard title="正常用戶" value={data.normal_count} unit="人" />
        <KpiCard title="高風險用戶" value={data.high_risk_count} unit="人" />
      </div>

      {/* Risk Histogram */}
      <div className="card mb-8">
        <p className="card-title">風險分數分布（Ensemble OOF）</p>
        <RiskHistogram data={data.risk_histogram} />
      </div>

      {/* High Risk User List */}
      <div className="card">
        <p className="card-title">高風險用戶列表（risk_score &gt; 0.35）</p>
        {data.high_risk_list.length === 0 ? (
          <p className="py-4 text-center text-sm text-slate-400">無高風險用戶</p>
        ) : (
          <div className="overflow-x-auto" style={{ maxHeight: '480px' }}>
            <table className="w-full">
              <thead className="sticky top-0 bg-white">
                <tr>
                  <th className="table-header">User ID</th>
                  <th className="table-header">Risk Score</th>
                  <th className="table-header">Risk Level</th>
                  <th className="table-header">真實標籤</th>
                </tr>
              </thead>
              <tbody>
                {data.high_risk_list.map((user) => (
                  <tr
                    key={user.user_id}
                    className="table-row-hover"
                    onClick={() => navigate(`/users/${user.user_id}`)}
                  >
                    <td className="table-cell font-medium text-brand-600">{user.user_id}</td>
                    <td className="table-cell font-mono">{user.risk_score.toFixed(3)}</td>
                    <td className="table-cell">
                      <span className={`badge ${
                        user.risk_level === 'critical' ? 'badge-red' :
                        user.risk_level === 'high' ? 'badge-orange' :
                        user.risk_level === 'medium' ? 'badge-yellow' : 'badge-green'
                      }`}>
                        {user.risk_level}
                      </span>
                    </td>
                    <td className="table-cell">
                      {user.fraud_label === 1 ? (
                        <span className="badge badge-red">詐欺</span>
                      ) : user.fraud_label === 0 ? (
                        <span className="badge badge-green">正常</span>
                      ) : (
                        <span className="text-xs text-slate-400">未知</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Overview;
