import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const cache = new Map<string, unknown>();

async function readJsonAsset<T>(filename: string): Promise<T> {
  if (cache.has(filename)) {
    return cache.get(filename) as T;
  }

  const url = new URL(`./_generated/${filename}`, import.meta.url);
  const text = await readFile(fileURLToPath(url), "utf8");
  const parsed = JSON.parse(text) as T;
  cache.set(filename, parsed);
  return parsed;
}

async function readJsonlIndex<T extends { user_id: string }>(filename: string): Promise<Map<string, T>> {
  const cacheKey = `jsonl:${filename}`;
  if (cache.has(cacheKey)) {
    return cache.get(cacheKey) as Map<string, T>;
  }

  const url = new URL(`./_generated/${filename}`, import.meta.url);
  const text = await readFile(fileURLToPath(url), "utf8");
  const index = new Map<string, T>();
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as T;
    index.set(row.user_id, row);
  }
  cache.set(cacheKey, index);
  return index;
}

export async function getMeta(): Promise<{ feature_names: string[]; high_risk_threshold: number }> {
  return readJsonAsset("meta.json");
}

export async function getOverview(): Promise<any> {
  return readJsonAsset("overview.json");
}

export async function getFeatureStats(): Promise<Record<string, any>> {
  return readJsonAsset("feature_stats.json");
}

export async function getUsersIndex(): Promise<Map<string, any>> {
  return readJsonlIndex("users.jsonl");
}

export async function getGraphSnapshot(): Promise<Record<string, Record<string, any>>> {
  return readJsonAsset("graph.json");
}

export async function getGnnIndex(): Promise<Record<string, any>> {
  return readJsonAsset("gnn_explanations.json");
}

export async function getCaseSeed(): Promise<any> {
  return readJsonAsset("case_status.json");
}

export async function getAuditSeed(): Promise<any> {
  return readJsonAsset("audit_log.json");
}

export async function getUserSummary(userId: string): Promise<any | null> {
  const idx = await getUsersIndex();
  return idx.get(userId) ?? null;
}
