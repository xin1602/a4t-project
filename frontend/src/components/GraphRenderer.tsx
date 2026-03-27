import React, { useEffect, useState, useCallback, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { getSubgraph } from '../api/client';
import type { GraphRendererProps, GraphNode, GraphEdge, GnnNodeImportance } from '../types';

// --- Color: blacklist (red) vs normal (gray) ---
export function getNodeColor(node: GraphNode): string {
  if (node.isBlacklist) return '#ef4444';
  return '#94a3b8';
}

interface FGNode extends GraphNode {
  x?: number; y?: number;
  vx?: number; vy?: number;
  fx?: number | undefined; fy?: number | undefined;
}
interface FGLink extends Omit<GraphEdge, 'source' | 'target'> {
  source: string | FGNode;
  target: string | FGNode;
}
interface FGGraphData { nodes: FGNode[]; links: FGLink[]; }

const FEATURE_LABELS: Record<string, string> = {
  crypto_night_tx_ratio: '夜間交易比例',
  crypto_to_wallet_hhi: '轉出錢包集中度',
  crypto_net_flow: '加密貨幣淨流量',
  graph_degree: '圖連接度',
  crypto_external_ratio: '外部轉帳比例',
  community_size: '社群大小',
  velocity_1h: '1小時交易速度',
  burst_count: '交易爆發次數',
  pagerank_score: 'PageRank',
  weighted_degree: '時間衰減度數',
};
function getFeatureLabel(n: string) { return FEATURE_LABELS[n] ?? n; }

/* ── GNN Node Mask Panel ── */
const GnnNodeMaskPanel: React.FC<{ items: GnnNodeImportance[] }> = ({ items }) => {
  if (!items || items.length === 0) return null;
  const maxAbs = Math.max(...items.map(d => Math.abs(d.importance)), 1e-6);
  return (
    <div className="absolute left-2 top-2 z-10 w-52 rounded-lg border border-surface-200 bg-white/95 p-2.5 shadow-md backdrop-blur-sm">
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">GNNExplainer 關鍵特徵</p>
      <div className="space-y-1">
        {items.map((item, i) => {
          const pct = Math.abs(item.importance) / maxAbs;
          const pos = item.importance >= 0;
          return (
            <div key={i} className="flex items-center gap-1.5">
              <span className="w-20 truncate text-[9px] text-slate-600" title={item.feature}>{getFeatureLabel(item.feature)}</span>
              <div className="relative h-2.5 flex-1 rounded-full bg-surface-100">
                <div className={`absolute top-0 h-2.5 rounded-full ${pos ? 'bg-red-400' : 'bg-indigo-400'}`} style={{ width: `${Math.max(pct * 100, 4)}%` }} />
              </div>
              <span className={`w-10 text-right text-[9px] font-mono ${pos ? 'text-red-600' : 'text-indigo-600'}`}>{item.importance >= 0 ? '+' : ''}{item.importance.toFixed(3)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ── Legend (inside the graph container) ── */
const GraphLegend: React.FC<{ hasGnn: boolean }> = ({ hasGnn }) => (
  <div className="absolute bottom-2 left-2 z-10 rounded-lg border border-surface-200 bg-white/95 px-2.5 py-1.5 shadow-sm backdrop-blur-sm">
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[9px] text-slate-500">
      <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-[#ef4444]" />黑名單</span>
      <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-full bg-[#94a3b8]" />非黑名單</span>
      <span className="flex items-center gap-1"><span className="inline-block h-3 w-3 rounded-full border-2 border-amber-400" />查詢目標</span>
      {hasGnn && <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-3 bg-red-500" />GNN 關鍵邊</span>}
    </div>
  </div>
);

/* ── Main Component ── */
const GraphRenderer: React.FC<GraphRendererProps> = ({ rootUserId, hopDepth, onNodeClick }) => {
  const [graphData, setGraphData] = useState<FGGraphData>({ nodes: [], links: [] });
  const [hasGnn, setHasGnn] = useState(false);
  const [gnnNodeMask, setGnnNodeMask] = useState<GnnNodeImportance[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dimensions, setDimensions] = useState<{ w: number; h: number }>({ w: 600, h: 400 });
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>();

  // Track container size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setDimensions({ w: Math.floor(width), h: Math.floor(height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getSubgraph(rootUserId, hopDepth)
      .then(data => {
        if (cancelled) return;
        setGraphData({ nodes: data.nodes.map(n => ({ ...n })), links: data.edges.map(e => ({ ...e })) });
        setHasGnn(data.hasGnnExplanation ?? false);
        setGnnNodeMask(data.gnnNodeMaskTop10 ?? []);
      })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [rootUserId, hopDepth]);

  /* ── Paint node: root = amber ring; blacklist = red; normal = gray ── */
  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const n = node as FGNode;
    const isRoot = n.id === rootUserId;
    const radius = isRoot ? 6 : 4; // fixed small size in world units
    const x = n.x ?? 0, y = n.y ?? 0;
    const color = getNodeColor(n);

    // Node fill
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    // Root: amber ring (tight around node)
    if (isRoot) {
      ctx.beginPath();
      ctx.arc(x, y, radius + 1.5, 0, 2 * Math.PI);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Label — only show when zoomed in enough
    const fontSize = 12 / globalScale;
    if (fontSize > 2 && fontSize < 20) {
      ctx.font = `${isRoot ? '600' : '400'} ${fontSize}px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = isRoot ? '#92400e' : '#475569';
      ctx.fillText(String(n.id), x, y + radius + 2);
    }
  }, [rootUserId]);

  const paintNodePointerArea = useCallback((node: any, color: string, ctx: CanvasRenderingContext2D) => {
    const n = node as FGNode;
    ctx.beginPath();
    ctx.arc(n.x ?? 0, n.y ?? 0, 8, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }, []);

  /* ── Paint link: GNN highlighted = red, others = light gray ── */
  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const l = link as FGLink;
    const src = l.source as FGNode, tgt = l.target as FGNode;
    if (src.x == null || tgt.x == null) return;

    const mask = l.edgeMask ?? 0;
    const isHigh = mask > 0.3;
    const w = isHigh ? 1.5 : 0.5; // fixed world-unit width
    const col = isHigh ? `rgba(239,68,68,${Math.min(0.5 + mask * 0.5, 1)})` : '#cbd5e1';

    ctx.beginPath();
    ctx.moveTo(src.x, src.y!);
    ctx.lineTo(tgt.x, tgt.y!);
    ctx.strokeStyle = col;
    ctx.lineWidth = w;
    ctx.stroke();

    // Arrow — filled triangle near target
    const dx = tgt.x - src.x, dy = tgt.y! - src.y!;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len > 10) {
      const tgtRadius = 5; // approximate target node radius
      const ang = Math.atan2(dy, dx);
      // Arrow tip sits just outside target node
      const tipX = tgt.x - Math.cos(ang) * (tgtRadius + 1);
      const tipY = tgt.y! - Math.sin(ang) * (tgtRadius + 1);
      const aLen = 4;
      const aHalf = Math.PI / 7;

      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - aLen * Math.cos(ang - aHalf), tipY - aLen * Math.sin(ang - aHalf));
      ctx.lineTo(tipX - aLen * Math.cos(ang + aHalf), tipY - aLen * Math.sin(ang + aHalf));
      ctx.closePath();
      ctx.fillStyle = col;
      ctx.fill();
    }

    // GNN edge mask label only (amount moved to hover tooltip)
    if (isHigh) {
      const labelSize = 8 / globalScale;
      if (labelSize > 2 && labelSize < 16) {
        const mx = (src.x + tgt.x) / 2, my = (src.y! + tgt.y!) / 2;
        ctx.font = `600 ${labelSize}px Inter, system-ui, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ef4444';
        ctx.fillText(mask.toFixed(2), mx, my - 2);
      }
    }
  }, []);

  const getNodeLabel = useCallback((node: any) => {
    const n = node as FGNode;
    const isRoot = n.id === rootUserId;
    const rootTag = isRoot ? '<div style="color:#f59e0b;font-weight:700;margin-bottom:2px;">★ 查詢目標</div>' : '';
    const blTag = n.isBlacklist ? '<span style="color:#ef4444;">● 黑名單</span>' : '<span style="color:#94a3b8;">● 非黑名單</span>';
    const srcTag = n.fraudLabel != null
      ? '<span style="color:#64748b;font-size:10px;">(ground truth)</span>'
      : '<span style="color:#818cf8;font-size:10px;">(模型預測)</span>';
    return `<div style="background:#0f172a;color:#f1f5f9;padding:10px 14px;border-radius:10px;font-size:12px;line-height:1.7;max-width:260px;font-family:Inter,system-ui,sans-serif;box-shadow:0 8px 24px -4px rgb(0 0 0/0.2);">
      ${rootTag}<div style="font-weight:600;margin-bottom:2px;">${n.id} ${blTag} ${srcTag}</div>
      <div><span style="color:#94a3b8;">Risk Score:</span> ${(n.riskScore ?? 0).toFixed(3)}</div>
      <div><span style="color:#94a3b8;">加密轉帳總額:</span> ${(n.cryptoTotalAmount ?? 0).toLocaleString()}</div>
      <div><span style="color:#94a3b8;">夜間交易比例:</span> ${((n.nightTxRatio ?? 0) * 100).toFixed(1)}%</div>
      <div style="margin-top:6px;color:#818cf8;font-size:11px;">點擊進入用戶查詢頁</div></div>`.trim();
  }, [rootUserId]);

  const getLinkLabel = useCallback((link: any) => {
    const l = link as FGLink;
    const amt = l.amount ?? 0;
    const mask = l.edgeMask ?? 0;
    const amtStr = amt > 0 ? `$${amt.toLocaleString()} TWD` : '共享 IP 連結';
    const maskStr = mask > 0.1 ? `<div><span style="color:#94a3b8;">GNN edge_mask:</span> <span style="color:#ef4444;font-weight:600;">${mask.toFixed(4)}</span></div>` : '';
    return `<div style="background:#0f172a;color:#f1f5f9;padding:8px 12px;border-radius:8px;font-size:12px;line-height:1.6;font-family:Inter,system-ui,sans-serif;box-shadow:0 4px 12px -2px rgb(0 0 0/0.2);">
      <div><span style="color:#94a3b8;">交易金額:</span> ${amtStr}</div>
      ${maskStr}
    </div>`.trim();
  }, []);

  const handleNodeClick = useCallback((node: any) => { const n = node as FGNode; if (n.id) onNodeClick(String(n.id)); }, [onNodeClick]);
  const handleNodeDragEnd = useCallback((node: any) => { const n = node as FGNode; n.fx = n.x; n.fy = n.y; }, []);

  // Auto zoom-to-fit after layout stabilizes
  const zoomedRef = useRef(false);
  useEffect(() => { zoomedRef.current = false; }, [rootUserId, hopDepth]);
  const handleEngineStop = useCallback(() => {
    if (!zoomedRef.current && fgRef.current) {
      fgRef.current.zoomToFit(400, 40);
      zoomedRef.current = true;
    }
  }, []);

  if (error) return (
    <div className="flex h-full min-h-[400px] w-full items-center justify-center rounded-xl bg-surface-50">
      <span className="text-sm text-red-500">載入圖網絡失敗：{error}</span>
    </div>
  );

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden rounded-xl bg-surface-50" style={{ minHeight: 300 }}>
      {loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-surface-50/80 backdrop-blur-sm">
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <div className="h-4 w-4 animate-spin-slow rounded-full border-2 border-surface-300 border-t-brand-600" />載入中…
          </div>
        </div>
      )}
      {hasGnn && <GnnNodeMaskPanel items={gnnNodeMask} />}
      <GraphLegend hasGnn={hasGnn} />
      {hasGnn && (
        <div className="absolute right-2 top-2 z-10 rounded-full bg-red-50 px-2.5 py-1 text-[9px] font-medium text-red-600 shadow-sm border border-red-200">GNNExplainer</div>
      )}
      <ForceGraph2D
        ref={fgRef} graphData={graphData} nodeId="id" linkSource="source" linkTarget="target"
        width={dimensions.w} height={dimensions.h}
        nodeCanvasObject={paintNode} nodeCanvasObjectMode={() => 'replace'}
        nodePointerAreaPaint={paintNodePointerArea} nodeLabel={getNodeLabel}
        linkCanvasObject={paintLink} linkCanvasObjectMode={() => 'replace'}
        linkLabel={getLinkLabel}
        onNodeClick={handleNodeClick} onNodeDragEnd={handleNodeDragEnd}
        onEngineStop={handleEngineStop}
        enableZoomInteraction enablePanInteraction enableNodeDrag cooldownTicks={100}
      />
    </div>
  );
};

export default GraphRenderer;
