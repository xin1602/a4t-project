import type { Config } from "@netlify/functions";
import { getFeatureStats, getUserSummary } from "./_shared/data";
import { getCaseState } from "./_shared/state";

function jsonResponse(body: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init?.headers ?? {}),
    },
  });
}

export default async function users(req: Request, context: { params: { userId?: string } }) {
  if (req.method !== "GET") {
    return jsonResponse({ detail: "Method not allowed" }, { status: 405 });
  }

  const userId = context.params.userId ?? "";
  const summary = await getUserSummary(userId);
  if (!summary) {
    return jsonResponse({ detail: `User '${userId}' not found in the system.` }, { status: 404 });
  }

  const [featureStats, caseState] = await Promise.all([getFeatureStats(), getCaseState()]);
  const currentCase = Object.values(caseState.cases).find((c: any) => c.user_id === userId) as any | undefined;
  const topFeatures = summary.top_features ?? [];
  const subsetStats: Record<string, any> = {};
  for (const feature of topFeatures) {
    if (featureStats[feature]) subsetStats[feature] = featureStats[feature];
  }

  return jsonResponse({
    user_id: userId,
    risk_score: summary.risk_score,
    risk_level: summary.risk_level,
    shap_values: summary.shap_values ?? {},
    top_features: topFeatures,
    feature_values: summary.feature_values ?? {},
    feature_stats: subsetStats,
    ig_attributions: {},
    tx_volume: summary.tx_volume ?? 0,
    night_tx_ratio: summary.night_tx_ratio ?? 0,
    counterparty_count: summary.counterparty_count ?? 0,
    case_status: currentCase?.status ?? summary.case_status ?? "pending",
    fraud_label: summary.fraud_label ?? null,
    heatmap_data: summary.heatmap_data ?? [],
  });
}

export const config: Config = {
  path: "/api/users/:userId",
};

