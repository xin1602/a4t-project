import type { Config } from "@netlify/functions";
import { getGraphSnapshot, getUserSummary } from "./_shared/data";

function sse(text: string) {
  const payload = `data: ${text.replace(/\n/g, "\ndata: ")}\n\n`;
  return new Response(payload, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  });
}

function summarizeShap(shap: Record<string, number>, limit = 5): string {
  const entries = Object.entries(shap).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, limit);
  return entries.map(([name, value]) => `- ${name}: ${value >= 0 ? "+" : ""}${value.toFixed(4)}`).join("\n");
}

export default async function llm(req: Request) {
  const pathname = new URL(req.url).pathname;
  const body = await req.json().catch(() => ({} as any));

  if (req.method !== "POST") {
    return sse("[ERROR] Method not allowed");
  }

  if (pathname === "/api/llm/investigate") {
    const userId = String(body.user_id ?? body.userId ?? "");
    const summary = await getUserSummary(userId);
    if (!summary) return sse(`[ERROR] User '${userId}' not found`);
    const text = [
      `## Investigation Summary`,
      `User: ${userId}`,
      `Risk score: ${Number(summary.risk_score ?? 0).toFixed(4)} (${summary.risk_level ?? "low"})`,
      ``,
      `### Top Signals`,
      summarizeShap(summary.shap_values ?? {}),
      ``,
      `### Quick Take`,
      summary.risk_score >= 0.8
        ? "This user is in the critical range and should be reviewed first."
        : summary.risk_score >= 0.6
          ? "This user is elevated and deserves closer inspection."
          : "This user is not at the top of the queue, but the listed features still deserve a look.",
      ``,
      `## Suggested Questions`,
      `- Which feature contributed most strongly?`,
      `- Are there suspicious transaction patterns in the heatmap?`,
      `- How does this compare with the user's local graph neighborhood?`,
    ].join("\n");
    return sse(text);
  }

  if (pathname === "/api/llm/cluster-summary") {
    const graph = await getGraphSnapshot();
    const nodes = Array.isArray(body.nodes) ? body.nodes : [];
    const edges = Array.isArray(body.edges) ? body.edges : [];
    const highRisk = nodes.filter((n: any) => Number(n.riskScore ?? 0) >= 0.6).length;
    const gnnEdges = edges.filter((e: any) => Number(e.edgeMask ?? 0) > 0.3).length;
    const text = [
      `## Cluster Summary`,
      `Nodes: ${nodes.length}`,
      `Edges: ${edges.length}`,
      `High-risk nodes: ${highRisk}`,
      `GNN-highlighted edges: ${gnnEdges}`,
      ``,
      `### Notes`,
      `- This cluster is being summarized from the Netlify snapshot.`,
      `- Graph adjacency includes ${Object.keys(graph.adjacency).length} known users.`,
    ].join("\n");
    return sse(text);
  }

  if (pathname === "/api/llm/chat") {
    const userId = String(body.user_id ?? "");
    const question = String(body.question ?? "");
    const summary = await getUserSummary(userId);
    const risk = Number(body.risk_score ?? summary?.risk_score ?? 0);
    const top5 = Array.isArray(body.shap_top5) ? body.shap_top5 : [];
    const text = [
      `## Answer`,
      `Question: ${question}`,
      `User: ${userId}`,
      `Risk score: ${risk.toFixed(4)}`,
      ``,
      `### Top drivers`,
      top5.length
        ? top5.map((item: any) => `- ${item.feature}: ${Number(item.shap ?? 0) >= 0 ? "+" : ""}${Number(item.shap ?? 0).toFixed(4)}`).join("\n")
        : summarizeShap(summary?.shap_values ?? {}),
      ``,
      `### Response`,
      `This is a rule-based fallback running inside Netlify Functions. It keeps the UI usable even when a dedicated LLM backend is not attached.`,
    ].join("\n");
    return sse(text);
  }

  return sse("[ERROR] Unknown LLM action");
}

export const config: Config = {
  path: ["/api/llm/investigate", "/api/llm/cluster-summary", "/api/llm/chat"],
};
