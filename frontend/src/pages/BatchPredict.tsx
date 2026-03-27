import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { batchPredict, downloadBatchCsv } from '../api/client';
import type { PredictionResult } from '../types';

type SourceType = 'api' | 'csv';

const BatchPredict: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [source, setSource] = useState<SourceType>('api');
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<PredictionResult[] | null>(null);
  const [totalUsers, setTotalUsers] = useState(0);
  const [highRiskCount, setHighRiskCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);

  const handleExecute = async () => {
    setLoading(true);
    setError(null);
    setCsvError(null);
    setResults(null);

    try {
      const res = await batchPredict(source, csvFile ?? undefined);
      setResults(res.results);
      setTotalUsers(res.total_users);
      setHighRiskCount(res.high_risk_count);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes('422')) {
        setCsvError(msg);
      } else {
        setError('批次預測失敗');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const blob = await downloadBatchCsv();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'batch_predict_results.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch {
      setError('下載 CSV 失敗');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    setCsvFile(file);
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <h2 className="mb-6 text-xl font-bold text-slate-800">批次預測</h2>

      {/* Source Selection */}
      <div className="mb-5">
        <p className="mb-2 text-sm font-medium text-slate-600">資料來源</p>
        <div className="flex gap-3">
          {(['api', 'csv'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className={`rounded-lg border-2 px-5 py-2.5 text-sm font-medium transition-all duration-150 ${
                source === s
                  ? 'border-brand-500 bg-brand-50 text-brand-700'
                  : 'border-surface-200 bg-white text-slate-500 hover:border-surface-300 hover:text-slate-700'
              }`}
            >
              {s === 'api' ? 'API 自動拉取' : '手動上傳 CSV'}
            </button>
          ))}
        </div>
      </div>

      {/* CSV Upload */}
      {source === 'csv' && (
        <div className="mb-5">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="text-sm text-slate-500 file:mr-3 file:rounded-lg file:border-0 file:bg-brand-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-brand-600 hover:file:bg-brand-100"
          />
          {csvFile && (
            <span className="ml-2 text-sm text-slate-400">{csvFile.name}</span>
          )}
        </div>
      )}

      {/* Execute Button */}
      <div className="mb-6 flex items-center gap-3">
        <button
          className="btn-primary"
          disabled={loading || (source === 'csv' && !csvFile)}
          onClick={handleExecute}
        >
          {loading ? '執行中…' : '執行批次預測'}
        </button>
        {loading && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <div className="h-3.5 w-3.5 animate-spin-slow rounded-full border-2 border-surface-300 border-t-brand-600" />
            處理中，請稍候…
          </div>
        )}
      </div>

      {/* CSV Validation Error */}
      {csvError && (
        <div className="mb-5 rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
          CSV 欄位驗證失敗：{csvError}
        </div>
      )}

      {/* General Error + Retry */}
      {error && (
        <div className="mb-5 flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
          <span>{error}</span>
          <button onClick={handleExecute} className="btn-danger text-xs">
            重試
          </button>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="card">
          <div className="mb-4 flex items-center justify-between">
            <p className="card-title !mb-0">
              預測結果（高風險：{highRiskCount} / 總計：{totalUsers}）
            </p>
            <button className="btn-secondary text-xs" onClick={handleDownload}>
              下載結果 CSV
            </button>
          </div>

          {results.length === 0 ? (
            <p className="py-4 text-center text-sm text-slate-400">無高風險用戶</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="table-header">User ID</th>
                    <th className="table-header">Risk Score</th>
                    <th className="table-header">Risk Level</th>
                    <th className="table-header text-center">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r) => (
                    <tr key={r.user_id} className="table-row-hover">
                      <td className="table-cell font-mono">{r.user_id}</td>
                      <td className="table-cell font-mono">{r.risk_score.toFixed(3)}</td>
                      <td className="table-cell">
                        <span className={`badge ${
                          r.risk_level === 'critical' ? 'badge-red' :
                          r.risk_level === 'high' ? 'badge-orange' :
                          r.risk_level === 'medium' ? 'badge-yellow' : 'badge-green'
                        }`}>
                          {r.risk_level}
                        </span>
                      </td>
                      <td className="table-cell text-center">
                        <button
                          onClick={() => navigate(`/users/${r.user_id}`)}
                          className="rounded-lg bg-brand-50 px-3 py-1.5 text-xs font-medium text-brand-600 transition-colors hover:bg-brand-100"
                        >
                          查看
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default BatchPredict;
