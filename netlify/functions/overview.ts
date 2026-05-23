import type { Config } from "@netlify/functions";
import { getOverview } from "./_shared/data";
import { getCaseState } from "./_shared/state";

export default async function overview() {
  const [base, caseState] = await Promise.all([
    getOverview(),
    getCaseState().catch(() => ({ cases: {} })),
  ]);
  const cases = Object.values(caseState.cases);
  const pending = cases.filter((c: any) => c.status === "pending").length;
  return Response.json({
    ...base,
    pending_case_count: pending,
  });
}

export const config: Config = {
  path: "/api/overview",
};
