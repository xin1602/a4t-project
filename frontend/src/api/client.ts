import type {
  OverviewData,
  UserDetail,
  SubgraphData,
  GnnExplanation,
  IgExplanation,
  CaseListResponse,
  CaseRecord,
  BatchPredictResponse,
} from '../types';

const BASE_URL = '/api';

// Throw an error with status code for non-2xx responses
async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// GET /api/overview
export async function getOverview(): Promise<OverviewData> {
  const res = await fetch(`${BASE_URL}/overview`);
  return handleResponse<OverviewData>(res);
}

// GET /api/users/{user_id}
export async function getUserDetail(userId: string): Promise<UserDetail> {
  const res = await fetch(`${BASE_URL}/users/${encodeURIComponent(userId)}`);
  return handleResponse<UserDetail>(res);
}

// GET /api/graph/{user_id}?hop={hop}
export async function getSubgraph(userId: string, hop: number): Promise<SubgraphData> {
  const res = await fetch(
    `${BASE_URL}/graph/${encodeURIComponent(userId)}?hop=${hop}`
  );
  return handleResponse<SubgraphData>(res);
}

// GET /api/explain/gnn/{user_id}
export async function getGnnExplanation(userId: string): Promise<GnnExplanation> {
  const res = await fetch(`${BASE_URL}/explain/gnn/${encodeURIComponent(userId)}`);
  return handleResponse<GnnExplanation>(res);
}

// GET /api/explain/ig/{user_id}
export async function getIgExplanation(userId: string): Promise<IgExplanation> {
  const res = await fetch(`${BASE_URL}/explain/ig/${encodeURIComponent(userId)}`);
  return handleResponse<IgExplanation>(res);
}

// GET /api/cases
export async function getCases(): Promise<CaseListResponse> {
  const res = await fetch(`${BASE_URL}/cases`);
  return handleResponse<CaseListResponse>(res);
}

// PATCH /api/cases/{case_id}
export async function updateCase(
  caseId: string,
  status: string,
  operator: string
): Promise<CaseRecord> {
  const res = await fetch(`${BASE_URL}/cases/${encodeURIComponent(caseId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, operator }),
  });
  return handleResponse<CaseRecord>(res);
}

// POST /api/predict/batch
export async function batchPredict(
  source: 'api' | 'csv',
  file?: File
): Promise<BatchPredictResponse> {
  const formData = new FormData();
  formData.append('source', source);
  if (file) {
    formData.append('file', file);
  }
  const res = await fetch(`${BASE_URL}/predict/batch`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<BatchPredictResponse>(res);
}

// GET /api/predict/download — returns a Blob for CSV download
export async function downloadBatchCsv(): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/predict/download`);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }
  return res.blob();
}

// SSE helper: reads a fetch response stream and calls onChunk for each data line
async function readSseStream(
  res: Response,
  onChunk: (text: string) => void,
  onError: (error: Error) => void
): Promise<void> {
  if (!res.ok) {
    onError(new Error(`HTTP ${res.status}: ${res.statusText}`));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError(new Error('Response body is not readable'));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE format: "data: <text>\n\n"
      const parts = buffer.split('\n\n');
      // Keep the last incomplete chunk in the buffer
      buffer = parts.pop() ?? '';

      for (const part of parts) {
        // SSE multi-line: collect all "data: " lines and join with \n
        const dataLines: string[] = [];
        for (const line of part.split('\n')) {
          if (line.startsWith('data: ')) {
            dataLines.push(line.slice(6));
          } else if (line === 'data:') {
            dataLines.push('');
          }
        }
        if (dataLines.length > 0) {
          onChunk(dataLines.join('\n'));
        }
      }
    }
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
  } finally {
    reader.releaseLock();
  }
}

// POST /api/llm/investigate — SSE stream
export async function streamInvestigation(
  userId: string,
  context: Record<string, unknown>,
  onChunk: (text: string) => void,
  onError: (error: Error) => void
): Promise<void> {
  let res: Response;
  try {
    // Merge userId into context and send as flat body
    const body = { user_id: userId, ...context };
    res = await fetch(`${BASE_URL}/llm/investigate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }
  await readSseStream(res, onChunk, onError);
}

// POST /api/llm/cluster-summary — SSE stream
export async function streamClusterSummary(
  graphData: SubgraphData,
  onChunk: (text: string) => void,
  onError: (error: Error) => void
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/llm/cluster-summary`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(graphData),
    });
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }
  await readSseStream(res, onChunk, onError);
}

// POST /api/llm/chat — SSE stream for free-form Q&A
export async function streamChat(
  userId: string,
  question: string,
  riskScore: number,
  shapTop5: Array<{ feature: string; value: number; shap: number }>,
  onChunk: (text: string) => void,
  onError: (error: Error) => void
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/llm/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        question,
        risk_score: riskScore,
        shap_top5: shapTop5,
      }),
    });
  } catch (err) {
    onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }
  await readSseStream(res, onChunk, onError);
}
