import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getSubgraph } from '../api/client';
import DepthSlider from '../components/DepthSlider';
import GraphRenderer from '../components/GraphRenderer';
import LlmPanel from '../components/LlmPanel';
import type { SubgraphData } from '../types';

const GraphAnalysis: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const navigate = useNavigate();
  const [hopDepth, setHopDepth] = useState(1);
  const [graphData, setGraphData] = useState<SubgraphData | null>(null);
  const initialFetchDone = useRef(false);

  useEffect(() => {
    if (!userId) return;
    getSubgraph(userId, hopDepth)
      .then((data) => {
        setGraphData(data);
        initialFetchDone.current = true;
      })
      .catch(() => {
        setGraphData(null);
      });
  }, [userId, hopDepth]);

  const handleNodeClick = useCallback(
    (uid: string) => navigate(`/users/${uid}`),
    [navigate],
  );

  if (!userId) {
    return (
      <div className="p-10">
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-sm text-red-600">
          缺少 user_id 參數
        </div>
      </div>
    );
  }

  const hasGnn = graphData?.hasGnnExplanation ?? false;
  const highRiskNodes = graphData?.nodes.filter((n) => n.riskScore >= 0.6) ?? [];
  const totalNodes = graphData?.nodes.length ?? 0;
  const totalEdges = graphData?.edges.length ?? 0;
  const gnnEdges = graphData?.edges.filter((e) => (e.edgeMask ?? 0) > 0.3).length ?? 0;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <h2 className="mb-4 text-xl font-bold text-slate-800">
        圖網絡分析 — <span className="text-brand-600">{userId}</span>
      </h2>

      {/* Stats bar */}
      <div className="mb-4 flex flex-wrap gap-3">
        <span className="rounded-lg bg-surface-100 px-3 py-1.5 text-xs text-slate-600">
          節點 {totalNodes}
        </span>
        <span className="rounded-lg bg-surface-100 px-3 py-1.5 text-xs text-slate-600">
          邊 {totalEdges}
        </span>
        <span className="rounded-lg bg-red-50 px-3 py-1.5 text-xs text-red-600">
          高風險 {highRiskNodes.length}
        </span>
        {hasGnn && (
          <span className="rounded-lg bg-red-50 px-3 py-1.5 text-xs font-medium text-red-700">
            GNNExplainer 關鍵邊 {gnnEdges}
          </span>
        )}
      </div>

      {/* Depth Slider */}
      <div className="mb-5 max-w-md">
        <DepthSlider value={hopDepth} onChange={setHopDepth} />
      </div>

      {/* Full-page Graph */}
      <div className="card mb-5 !p-0 overflow-hidden" style={{ height: 'calc(100vh - 380px)', minHeight: '400px' }}>
        <GraphRenderer
          rootUserId={userId}
          hopDepth={hopDepth}
          onNodeClick={handleNodeClick}
        />
      </div>

      {/* LLM Cluster Summary */}
      <LlmPanel
        userId={userId}
        riskScore={0}
        shapTop5={[]}
      />
    </div>
  );
};

export default GraphAnalysis;
