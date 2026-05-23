import type { Config } from "@netlify/functions";
import { getGnnIndex } from "./_shared/data";

function jsonResponse(body: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init?.headers ?? {}),
    },
  });
}

export default async function explainGnn(_req: Request, context: { params: { userId?: string } }) {
  const userId = context.params.userId ?? "";
  const gnnIndex = await getGnnIndex();
  const gnn = gnnIndex[userId];
  if (!gnn) {
    return jsonResponse({ detail: `No pre-computed GNN explanation for user '${userId}'.` }, { status: 404 });
  }
  return jsonResponse(gnn);
}

export const config: Config = {
  path: "/api/explain/gnn/:userId",
};

