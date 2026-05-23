import type { Config } from "@netlify/functions";
import { getGraphSnapshot, getUserSummary } from "./_shared/data";

function jsonResponse(body: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init?.headers ?? {}),
    },
  });
}

function clampHop(hop: number): number {
  return Math.max(1, Math.min(5, hop));
}

export default async function graph(req: Request, context: { params: { userId?: string } }) {
  if (req.method !== "GET") {
    return jsonResponse({ detail: "Method not allowed" }, { status: 405 });
  }

  const userId = context.params.userId ?? "";
  const summary = await getUserSummary(userId);
  if (!summary) {
    return jsonResponse({ detail: `User '${userId}' not found in the graph.` }, { status: 404 });
  }

  const url = new URL(req.url);
  const hop = clampHop(Number(url.searchParams.get("hop") ?? "1"));
  const snapshots = await getGraphSnapshot();
  const userSnapshots = snapshots[userId];
  const snapshot = userSnapshots?.[String(hop)];
  if (!snapshot) {
    return jsonResponse(
      { detail: `No precomputed graph snapshot for user '${userId}' at hop ${hop}.` },
      { status: 404 }
    );
  }

  return jsonResponse(snapshot);
}

export const config: Config = {
  path: "/api/graph/:userId",
};
