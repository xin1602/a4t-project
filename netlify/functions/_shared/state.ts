import { getStore } from "@netlify/blobs";
import { getAuditSeed, getCaseSeed } from "./data";

type CaseState = { cases: Record<string, any> };
type AuditState = { entries: any[] };

let seeded = false;
let blobsAvailable = true;
let caseStateMemory: CaseState | null = null;
let auditStateMemory: AuditState | null = null;
let batchStateMemory: any[] | null = null;
let seedCache: Promise<{ caseSeed: CaseState; auditSeed: AuditState }> | null = null;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

async function getSeeds(): Promise<{ caseSeed: CaseState; auditSeed: AuditState }> {
  if (!seedCache) {
    seedCache = Promise.all([getCaseSeed(), getAuditSeed()]).then(([caseSeed, auditSeed]) => ({
      caseSeed: clone(caseSeed),
      auditSeed: clone(auditSeed),
    }));
  }
  return seedCache;
}

async function ensureMemoryState(): Promise<{ cases: CaseState; audit: AuditState }> {
  if (!caseStateMemory || !auditStateMemory) {
    const { caseSeed, auditSeed } = await getSeeds();
    caseStateMemory = clone(caseSeed);
    auditStateMemory = clone(auditSeed);
  }
  if (!batchStateMemory) {
    batchStateMemory = [];
  }
  return { cases: caseStateMemory, audit: auditStateMemory };
}

async function ensureSeeded(): Promise<void> {
  if (seeded) return;
  const { cases, audit } = await ensureMemoryState();
  if (!blobsAvailable) {
    seeded = true;
    return;
  }

  try {
    const caseStore = getStore({ name: "a4t-case-state", consistency: "strong" });
    const existing = await caseStore.get("case_status", { type: "json" }).catch(() => null);
    if (!existing) {
      await caseStore.setJSON("case_status", cases);
    }
    const existingAudit = await caseStore.get("audit_log", { type: "json" }).catch(() => null);
    if (!existingAudit) {
      await caseStore.setJSON("audit_log", audit);
    }
  } catch {
    blobsAvailable = false;
  }
  seeded = true;
}

export async function getCaseState(): Promise<{ cases: Record<string, any> }> {
  await ensureSeeded();
  if (blobsAvailable) {
    try {
      const caseStore = getStore({ name: "a4t-case-state", consistency: "strong" });
      const state = await caseStore.get("case_status", { type: "json" });
      if (state) {
        caseStateMemory = clone(state as { cases: Record<string, any> });
        return caseStateMemory;
      }
    } catch {
      blobsAvailable = false;
    }
  }

  const { cases } = await ensureMemoryState();
  return cases;
}

export async function getAuditState(): Promise<{ entries: any[] }> {
  await ensureSeeded();
  if (blobsAvailable) {
    try {
      const caseStore = getStore({ name: "a4t-case-state", consistency: "strong" });
      const state = await caseStore.get("audit_log", { type: "json" });
      if (state) {
        auditStateMemory = clone(state as { entries: any[] });
        return auditStateMemory;
      }
    } catch {
      blobsAvailable = false;
    }
  }

  const { audit } = await ensureMemoryState();
  return audit;
}

export async function setCaseState(state: { cases: Record<string, any> }): Promise<void> {
  await ensureSeeded();
  caseStateMemory = clone(state);
  if (!blobsAvailable) return;

  try {
    const caseStore = getStore({ name: "a4t-case-state", consistency: "strong" });
    await caseStore.setJSON("case_status", state);
  } catch {
    blobsAvailable = false;
  }
}

export async function setAuditState(state: { entries: any[] }): Promise<void> {
  await ensureSeeded();
  auditStateMemory = clone(state);
  if (!blobsAvailable) return;

  try {
    const caseStore = getStore({ name: "a4t-case-state", consistency: "strong" });
    await caseStore.setJSON("audit_log", state);
  } catch {
    blobsAvailable = false;
  }
}

export async function getLastBatchState(): Promise<any[] | null> {
  if (blobsAvailable) {
    try {
      const batchStore = getStore({ name: "a4t-batch-state", consistency: "strong" });
      const state = await batchStore.get("last_batch_results", { type: "json" });
      if (Array.isArray(state)) {
        batchStateMemory = clone(state);
        return batchStateMemory;
      }
    } catch {
      blobsAvailable = false;
    }
  }

  return batchStateMemory;
}

export async function setLastBatchState(results: any[]): Promise<void> {
  batchStateMemory = clone(results);
  if (!blobsAvailable) return;

  try {
    const batchStore = getStore({ name: "a4t-batch-state", consistency: "strong" });
    await batchStore.setJSON("last_batch_results", results);
  } catch {
    blobsAvailable = false;
  }
}
