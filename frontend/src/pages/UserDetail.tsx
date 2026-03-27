import React, { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getUserDetail, getGnnExplanation } from '../api/client';
import KpiCard from '../components/KpiCard';
import ShapChart from '../components/ShapChart';
import FeatureTable from '../components/FeatureTable';
import HeatmapChart from '../components/HeatmapChart';
import GraphRenderer from '../components/GraphRenderer';
import LlmPanel from '../components/LlmPanel';
import type { UserDetail as UserDetailType } from '../types';

const RISK_LEVEL_BADGE: Record<string, string> = {
  low: 'badge-green',
  medium: 'badge-yellow',
  high: 'badge-orange',
  critical: 'badge-red',
};

type AttrTab = 'lgbm' | 'graphsage';

const TAB_LABELS: Record<AttrTab, string> = {
  lgbm: 'LightGBM (SHAP)',
  graphsage: 'GraphSAGE (GNNExplainer)',
};

// Feature name -> Chinese label
const FEATURE_LABELS: Record<string, string> = {
  crypto_night_tx_ratio: '夜間交易比例',
  crypto_to_wallet_hhi: '轉出錢包集中度',
  crypto_net_flow: '加密貨幣淨流量',
  graph_degree: '圖連接度',
  crypto_external_ratio: '外部轉帳比例',
  community_size: '社群大小',
  crypto_amount_cv: '金額變異係數',
  twd_outflow_sum: '法幣總流出',
  crypto_total_amount: '加密貨幣總額',
  velocity_1h: '1小時交易速度',
  burst_count: '交易爆發次數',
  pagerank_score: 'PageRank 分數',
  weighted_degree: '時間衰減度數',
};

interface FeatureAttributionTabsProps {
  shapValues: Record<string, number>;
  topFeatures: string[];
  featureValues: Record<string, number>;
  gnnNodeMask: Array<[string, number]>;
  gnnLoading: boolean;
  hasGnn: boolean;
}

const FeatureAttributionTabs: React.FC<FeatureAttributionTabsProps> = ({
  shapValues,
  topFeatures,
  featureValues,
  gnnNodeMask,
  gnnLoading,
  hasGnn,
}) => {
  const [tab, setTab] = useState<AttrTab>('lgbm');

  // Convert GNN node_mask to ShapChart-compatible format
  const gnnAsShap: Record<string, number> = {};
  const gnnTopFeatures: string[] = [];
  for (const [name, val] of gnnNodeMask) {
    gnnAsShap[name] = val;
    gnnTopFeatures.push(name);
  }

  return (
    <div className="card mb-5">
      <div className="mb-4 flex items-center gap-1 border-b border-slate-200">
        {(Object.keys(TAB_LABELS) as AttrTab[]).map((key) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === key
                ? 'border-b-2 border-brand-600 text-brand-700'
                : 'text-slate-400 hover:text-slate-600'
            }`}
          >
            {TAB_LABELS[key]}
          </button>
        ))}
        {hasGnn && (
          <span className="ml-auto flex items-center gap-1.5 rounded-full bg-red-50 px-3 py-1 text-xs font-medium text-red-600">
            GNNExplainer 可用
          </span>
        )}
      </div>

      {tab === 'lgbm' && (
        <ShapChart
          shapValues={shapValues}
          topFeatures={topFeatures}
          featureValues={featureValues}
        />
      )}

      {tab === 'graphsage' && (
        <>
          {gnnLoading ? (
            <div className="flex items-center gap-2 py-8 justify-center text-sm text-slate-400">
              <div className="h-4 w-4 animate-spin-slow rounded-full border-2 border-surface-300 border-t-brand-600" />
              載入 GNNExplainer…
            </div>
          ) : gnnNodeMask.length === 0 ? (
            <div className="py-5 text-center text-sm text-slate-400">
              <p>此用戶無預計算的 GNNExplainer 資料</p>
              <p className="mt-1 text-xs">GNNExplainer 僅對 TOP-K 最高風險用戶預計算</p>
            </div>
          ) : (
            <>
              <p className="mb-3 text-xs text-slate-500">
                GNNExplainer node_mask：圖傳播過程中對黑名單判定貢獻最大的節點特徵
              </p>
              <div className="space-y-2">
                {gnnNodeMask.map(([name, val], i) => {
                  const maxAbs = Math.max(...gnnNodeMask.map(([, v]) => Math.abs(v)), 1e-6);
                  const pct = Math.abs(val) / maxAbs;
                  const isPositive = val >= 0;
                  const label = FEATURE_LABELS[name] ?? name;
                  return (
                    <div key={i} className="flex items-center gap-3">
                      <span className="w-36 truncate text-xs text-slate-600" title={name}>
                        {label}
                      </span>
                      <div className="relative h-4 flex-1 rounded-full bg-surface-100">
                        <div
                          className={`absolute top-0 h-4 rounded-full transition-all ${
                            isPositive ? 'bg-red-400' : 'bg-indigo-400'
                          }`}
                          style={{ width: `${Math.max(pct * 100, 3)}%` }}
                        />
                      </div>
                      <span className={`w-16 text-right text-xs font-mono ${
                        isPositive ? 'text-red-600' : 'text-indigo-600'
                      }`}>
                        {val >= 0 ? '+' : ''}{val.toFixed(4)}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 flex gap-4 text-[10px] text-slate-400">
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-sm bg-red-400" /> 推向詐欺
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block h-2 w-2 rounded-sm bg-indigo-400" /> 推向正常
                </span>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
};

const UserDetail: React.FC = () => {
  const { userId: urlUserId } = useParams<{ userId: string }>();
  const navigate = useNavigate();

  const [searchInput, setSearchInput] = useState(urlUserId ?? '');
  const [activeUserId, setActiveUserId] = useState(urlUserId ?? '');
  const [data, setData] = useState<UserDetailType | null>(null);
  const [gnnNodeMask, setGnnNodeMask] = useState<Array<[string, number]>>([]);
  const [gnnLoading, setGnnLoading] = useState(false);
  const [hasGnn, setHasGnn] = useState(false);
  const [loading, setLoading] = useState(false);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchUser = useCallback((uid: string) => {
    if (!uid.trim()) return;
    setLoading(true);
    setNotFound(false);
    setError(null);
    setData(null);
    setGnnNodeMask([]);
    setHasGnn(false);

    getUserDetail(uid.trim())
      .then((res) => {
        setData(res);
        setActiveUserId(uid.trim());

        // Fetch GNNExplainer node_mask asynchronously
        setGnnLoading(true);
        getGnnExplanation(uid.trim())
          .then((gnnRes) => {
            setGnnNodeMask(gnnRes.node_mask_top10);
            setHasGnn(true);
          })
          .catch(() => {
            setGnnNodeMask([]);
            setHasGnn(false);
          })
          .finally(() => setGnnLoading(false));
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes('404')) {
          setNotFound(true);
        } else {
          setError(msg);
        }
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (urlUserId) {
      setSearchInput(urlUserId);
      fetchUser(urlUserId);
    }
  }, [urlUserId, fetchUser]);

  const handleSearch = () => fetchUser(searchInput);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleSearch();
  };
  const handleNodeClick = (uid: string) => navigate(`/users/${uid}`);

  const heatmapData = data ? data.heatmap_data : [];

  return (
    <div className="mx-auto max-w-5xl px-6 py-8 pr-52">
      {/* Search Bar */}
      <div className="mb-6 flex gap-2">
        <input
          type="text"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="輸入 user_id 搜尋"
          className="input-field"
        />
        <button onClick={handleSearch} className="btn-primary whitespace-nowrap">
          搜尋
        </button>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <div className="h-4 w-4 animate-spin-slow rounded-full border-2 border-surface-300 border-t-brand-600" />
          載入中…
        </div>
      )}
      {notFound && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm font-medium text-red-600">
          查無此用戶
        </div>
      )}
      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
          載入失敗：{error}
        </div>
      )}

      {data && (
        <>
          {/* Data Source Label */}
          <div className={`mb-6 flex items-center gap-3 rounded-xl border px-5 py-4 text-sm font-medium ${
            data.fraud_label === 1
              ? 'border-red-200 bg-red-50 text-red-700'
              : data.fraud_label === 0
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-blue-200 bg-blue-50 text-blue-700'
          }`}>
            <span className="text-lg">{data.fraud_label === 1 ? '⚠' : data.fraud_label === 0 ? '✓' : '🔍'}</span>
            <span>
              {data.fraud_label === 1
                ? '已標記詐欺（黑名單）'
                : data.fraud_label === 0
                  ? '已標記正常'
                  : '待預測用戶（模型即時推論）'}
            </span>
          </div>

          {/* Risk Score + Key Metrics */}
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-4">
            <div className="card">
              <p className="mb-1 text-xs font-medium uppercase tracking-wider text-slate-400">Risk Score</p>
              <p className="text-3xl font-bold text-slate-800">{data.risk_score.toFixed(3)}</p>
              <span className={`badge mt-2 ${RISK_LEVEL_BADGE[data.risk_level] ?? 'badge-gray'}`}>
                {data.risk_level.toUpperCase()}
              </span>
            </div>
            <KpiCard title="交易量" value={data.tx_volume} />
            <KpiCard title="夜間交易比例" value={Math.round(data.night_tx_ratio * 100)} unit="%" />
            <KpiCard title="交易對手數" value={data.counterparty_count} />
          </div>

          {/* Feature Attribution — Tabbed: LightGBM SHAP / GraphSAGE GNNExplainer */}
          <FeatureAttributionTabs
            shapValues={data.shap_values}
            topFeatures={data.top_features}
            featureValues={data.feature_values}
            gnnNodeMask={gnnNodeMask}
            gnnLoading={gnnLoading}
            hasGnn={hasGnn}
          />

          {/* Feature Detail Table */}
          <div className="card mb-5">
            <p className="card-title">特徵明細（vs 全體統計）</p>
            <FeatureTable
              topFeatures={data.top_features}
              shapValues={data.shap_values}
              featureValues={data.feature_values}
              featureStats={data.feature_stats}
            />
          </div>

          {/* Heatmap */}
          <div className="card mb-5">
            <p className="card-title">交易時間熱力圖</p>
            {heatmapData.length === 0 ? (
              <p className="py-5 text-center text-sm text-slate-400">尚無交易時間資料</p>
            ) : (
              <HeatmapChart data={heatmapData} />
            )}
          </div>

          {/* Graph Renderer with GNNExplainer */}
          <div className="card mb-5 !p-4 overflow-hidden" style={{ height: '480px' }}>
            <div className="mb-2 flex items-center justify-between">
              <p className="card-title">關聯圖網絡（1 hop）</p>
              <span className="text-[10px] text-slate-400">
                紅色加粗邊 = GNNExplainer 識別的關鍵交易路徑
              </span>
            </div>
            <div style={{ height: 'calc(100% - 32px)' }}>
              <GraphRenderer rootUserId={activeUserId} hopDepth={1} onNodeClick={handleNodeClick} />
            </div>
          </div>

          {/* LLM Panel */}
          <LlmPanel
            userId={activeUserId}
            riskScore={data.risk_score}
            shapTop5={data.top_features.slice(0, 5).map(f => ({
              feature: f,
              value: data.shap_values[f] ?? 0,
              shap: data.shap_values[f] ?? 0,
            }))}
          />
        </>
      )}
    </div>
  );
};

export default UserDetail;
