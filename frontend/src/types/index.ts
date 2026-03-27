// Graph network node
export interface GraphNode {
  id: string;
  riskScore: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  caseStatus: 'pending' | 'blacklisted' | 'watching' | 'normal';
  graphDegree: number;
  cryptoTotalAmount: number;
  nightTxRatio: number;
  isBlacklist: boolean;
  fraudLabel: number | null;
}

// Graph network edge
export interface GraphEdge {
  source: string;
  target: string;
  amount: number;
  edgeMask?: number;
}

// GraphRenderer component props
export interface GraphRendererProps {
  rootUserId: string;
  hopDepth: number;
  onNodeClick: (userId: string) => void;
}

// Prediction result
export interface PredictionResult {
  user_id: string;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  is_blacklist: boolean;
}

// Case record
export interface CaseRecord {
  case_id: string;
  user_id: string;
  status: 'pending' | 'blacklisted' | 'watching' | 'normal';
  risk_score: number;
  created_at: string;
  updated_at: string;
  updated_by: string;
}

// User detail
export interface UserDetail {
  user_id: string;
  risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  shap_values: Record<string, number>;
  top_features: string[];
  feature_values: Record<string, number>;
  feature_stats: Record<string, {
    type: 'continuous' | 'categorical';
    mean?: number; std?: number; p25?: number; p50?: number; p75?: number;
    distribution?: Record<string, number>; total?: number;
  }>;
  ig_attributions: Record<string, number>;
  tx_volume: number;
  night_tx_ratio: number;
  counterparty_count: number;
  case_status: string;
  fraud_label: number | null;
  heatmap_data: Array<{ hour: number; day: number; count: number }>;
}

// Overview data
export interface OverviewData {
  total_users: number;
  fraud_count: number;
  normal_count: number;
  high_risk_count: number;
  risk_histogram: Array<{ bin: string; count: number }>;
  high_risk_list: Array<{
    user_id: string;
    risk_score: number;
    risk_level: string;
    fraud_label: number | null;
  }>;
}

// GNN node feature importance from GNNExplainer
export interface GnnNodeImportance {
  feature: string;
  importance: number;
}

// Subgraph data (with optional GNNExplainer overlay)
export interface SubgraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  hasGnnExplanation?: boolean;
  gnnNodeMaskTop10?: GnnNodeImportance[];
}

// GNN explanation
export interface GnnExplanation {
  user_id: string;
  edge_mask: Record<string, number>;
  node_mask_top10: Array<[string, number]>;
  computed_at: string;
}

// IG explanation
export interface IgExplanation {
  user_id: string;
  attributions: Record<string, number>;
  convergence_delta: number;
  computed_at: string;
}

// Batch predict response
export interface BatchPredictResponse {
  total_users: number;
  high_risk_count: number;
  results: PredictionResult[];
  status: string;
}

// Case list response
export interface CaseListResponse {
  cases: CaseRecord[];
}
