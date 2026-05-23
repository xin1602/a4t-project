import type { Config } from "@netlify/functions";
import { getAuditState, getCaseState, setAuditState, setCaseState } from "./_shared/state";

function jsonResponse(body: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(body), {
    ...init,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...(init?.headers ?? {}),
    },
  });
}

export default async function cases(req: Request) {
  const url = new URL(req.url);
  const caseId = url.pathname.split("/").pop() ?? "";

  if (req.method === "GET" && url.pathname === "/api/cases") {
    const state = await getCaseState();
    return jsonResponse({ cases: Object.values(state.cases) });
  }

  if (req.method === "PATCH" && caseId && caseId !== "cases") {
    const payload = (await req.json()) as { status?: string; operator?: string };
    const nextStatus = payload.status ?? "";
    const operator = payload.operator ?? "system";
    const caseState = await getCaseState();
    const auditState = await getAuditState();
    const target = caseState.cases[caseId];
    if (!target) {
      return jsonResponse({ detail: `Case '${caseId}' not found` }, { status: 404 });
    }

    const now = new Date().toISOString();
    const updated = {
      ...target,
      status: nextStatus,
      updated_at: now,
      updated_by: operator,
    };
    caseState.cases[caseId] = updated;
    auditState.entries.push({
      entry_id: `AUD-${crypto.randomUUID().replace(/-/g, "").slice(0, 8)}`,
      case_id: caseId,
      user_id: target.user_id,
      operator,
      timestamp_utc: now,
      old_status: target.status,
      new_status: nextStatus,
    });
    await Promise.all([setCaseState(caseState), setAuditState(auditState)]);
    return jsonResponse(updated);
  }

  return jsonResponse({ detail: "Method not allowed" }, { status: 405 });
}

export const config: Config = {
  path: ["/api/cases", "/api/cases/:caseId"],
  method: ["GET", "PATCH"],
};
