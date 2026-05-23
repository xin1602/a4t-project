import { getStore } from "@netlify/blobs";
import { getAuditSeed, getCaseSeed } from "./data";

const CASE_STORE = getStore({ name: "a4t-case-state", consistency: "strong" });
const BATCH_STORE = getStore({ name: "a4t-batch-state", consistency: "strong" });

let seeded = false;

async function ensureSeeded(): Promise<void> {
  if (seeded) return;
  const [caseSeed, auditSeed] = await Promise.all([getCaseSeed(), getAuditSeed()]);
  const existing = await CASE_STORE.get("case_status", { type: "json" }).catch(() => null);
  if (!existing) {
    await CASE_STORE.setJSON("case_status", caseSeed);
  }
  const existingAudit = await CASE_STORE.get("audit_log", { type: "json" }).catch(() => null);
  if (!existingAudit) {
    await CASE_STORE.setJSON("audit_log", auditSeed);
  }
  seeded = true;
}

export async function getCaseState(): Promise<{ cases: Record<string, any> }> {
  await ensureSeeded();
  const state = await CASE_STORE.get("case_status", { type: "json" });
  return (state as { cases: Record<string, any> }) ?? { cases: {} };
}

export async function getAuditState(): Promise<{ entries: any[] }> {
  await ensureSeeded();
  const state = await CASE_STORE.get("audit_log", { type: "json" });
  return (state as { entries: any[] }) ?? { entries: [] };
}

export async function setCaseState(state: { cases: Record<string, any> }): Promise<void> {
  await ensureSeeded();
  await CASE_STORE.setJSON("case_status", state);
}

export async function setAuditState(state: { entries: any[] }): Promise<void> {
  await ensureSeeded();
  await CASE_STORE.setJSON("audit_log", state);
}

export async function getLastBatchState(): Promise<any[] | null> {
  return (await BATCH_STORE.get("last_batch_results", { type: "json" }).catch(() => null)) as any[] | null;
}

export async function setLastBatchState(results: any[]): Promise<void> {
  await BATCH_STORE.setJSON("last_batch_results", results);
}

