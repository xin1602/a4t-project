import type { Config } from "@netlify/functions";
import { getMeta, getUserSummary } from "./_shared/data";
import { getLastBatchState, setLastBatchState } from "./_shared/state";

function jsonResponse(body: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init?.headers ?? {}),
    },
  });
}

function csvResponse(text: string) {
  return new Response(text, {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": "attachment; filename=batch_predict_results.csv",
    },
  });
}

function riskLevel(score: number): "low" | "medium" | "high" | "critical" {
  if (score >= 0.8) return "critical";
  if (score >= 0.6) return "high";
  if (score >= 0.35) return "medium";
  return "low";
}

function generateCsv(results: any[]): string {
  const rows = [
    [
      "user_id",
      "risk_score",
      "is_blacklist",
      "top_feature_1",
      "top_feature_1_shap",
      "top_feature_2",
      "top_feature_2_shap",
      "top_feature_3",
      "top_feature_3_shap",
    ].join(","),
  ];
  for (const result of results) {
    const top = result.top_features ?? [];
    const shap = result.shap_values ?? {};
    const cols = [result.user_id, result.risk_score, result.is_blacklist];
    for (let i = 0; i < 3; i += 1) {
      const feature = top[i] ?? "";
      cols.push(feature);
      cols.push(feature ? (shap[feature] ?? 0) : "");
    }
    rows.push(cols.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(","));
  }
  return rows.join("\n");
}

export default async function predict(req: Request) {
  const url = new URL(req.url);
  const pathname = url.pathname;

  if (req.method === "GET" && pathname === "/api/predict/download") {
    const results = (await getLastBatchState()) ?? [];
    if (results.length === 0) {
      return jsonResponse({ detail: "No batch prediction results available. Run POST /api/predict/batch first." }, { status: 404 });
    }
    return csvResponse(generateCsv(results));
  }

  if (req.method !== "POST" || pathname !== "/api/predict/batch") {
    return jsonResponse({ detail: "Method not allowed" }, { status: 405 });
  }

  const form = await req.formData();
  const source = String(form.get("source") ?? "api");
  const { feature_names, representative_user_ids = [] } = await getMeta();
  const representativeSet = new Set(representative_user_ids.map(String));
  let userIds: string[] = [];

  if (source === "api") {
    userIds = representative_user_ids.map(String);
  } else if (source === "csv") {
    const file = form.get("file");
    if (!(file instanceof File)) {
      return jsonResponse({ detail: "source='csv' requires a file upload" }, { status: 422 });
    }
    const text = await file.text();
    const lines = text.trim().split(/\r?\n/);
    const header = lines[0]?.split(",").map((s) => s.trim()) ?? [];
    const missing = feature_names.filter((name) => !header.includes(name));
    if (missing.length > 0) {
      return jsonResponse({
        message: "CSV is missing required feature columns",
        missing_fields: missing.sort(),
      }, { status: 422 });
    }
    const userIdIdx = header.indexOf("user_id");
    if (userIdIdx < 0) {
      return jsonResponse({ detail: "CSV must contain a 'user_id' column" }, { status: 422 });
    }
    userIds = lines.slice(1).filter(Boolean).map((line) => line.split(",")[userIdIdx]).filter(Boolean);
    const missingDemoUsers = userIds.filter((uid) => !representativeSet.has(String(uid)));
    if (missingDemoUsers.length > 0) {
      return jsonResponse(
        {
          detail: "CSV batch is limited to the 5 representative users deployed on Netlify.",
          missing_users: missingDemoUsers,
        },
        { status: 404 }
      );
    }
  } else {
    return jsonResponse({ detail: `Invalid source '${source}'. Must be 'api' or 'csv'.` }, { status: 422 });
  }

  const allResults: any[] = [];
  for (const userId of userIds) {
    const summary = await getUserSummary(String(userId));
    if (!summary) {
      continue;
    }
    allResults.push({
      user_id: String(userId),
      risk_score: Number(summary.risk_score ?? 0),
      risk_level: riskLevel(Number(summary.risk_score ?? 0)),
      is_blacklist: Boolean(summary.isBlacklist ?? false),
      top_features: summary.top_features ?? [],
      shap_values: summary.shap_values ?? {},
    });
  }

  await setLastBatchState(allResults);
  const highRisk = allResults.filter((row) => row.risk_score > 0.35);
  return jsonResponse({
    total_users: allResults.length,
    high_risk_count: highRisk.length,
    results: highRisk.map(({ user_id, risk_score, risk_level, is_blacklist }) => ({
      user_id,
      risk_score,
      risk_level,
      is_blacklist,
    })),
    status: "completed",
  });
}

export const config: Config = {
  path: ["/api/predict/batch", "/api/predict/download"],
  method: ["GET", "POST"],
};
