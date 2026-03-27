import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { streamInvestigation, streamChat } from '../api/client';

interface LlmPanelProps {
  userId: string;
  riskScore: number;
  shapTop5: Array<{ feature: string; value: number; shap: number }>;
}

/** Parse LLM output to extract suggested questions from "## 建議提問" section */
function extractSuggestions(text: string): string[] {
  const lines = text.split('\n');
  let inSuggestions = false;
  const suggestions: string[] = [];
  for (const line of lines) {
    if (/^##\s*建議提問/.test(line)) {
      inSuggestions = true;
      continue;
    }
    if (inSuggestions && /^##\s/.test(line)) break; // next section
    if (inSuggestions && /^-\s+/.test(line)) {
      suggestions.push(line.replace(/^-\s+/, '').trim());
    }
  }
  return suggestions;
}

/** Remove the "## 建議提問" section from the report text */
function stripSuggestions(text: string): string {
  const lines = text.split('\n');
  const out: string[] = [];
  let inSuggestions = false;
  for (const line of lines) {
    if (/^##\s*建議提問/.test(line)) {
      inSuggestions = true;
      continue;
    }
    if (inSuggestions && /^##\s/.test(line)) {
      inSuggestions = false;
    }
    if (!inSuggestions) out.push(line);
  }
  return out.join('\n').trim();
}

const LlmPanel: React.FC<LlmPanelProps> = ({ userId, riskScore, shapTop5 }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [rawAnalysis, setRawAnalysis] = useState('');
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'ai'; text: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const prevUserIdRef = useRef('');

  // Auto-fetch analysis when userId changes
  useEffect(() => {
    if (!userId || userId === prevUserIdRef.current) return;
    prevUserIdRef.current = userId;
    setRawAnalysis('');
    setChatMessages([]);
    setAnalysisLoading(true);

    let raw = '';
    streamInvestigation(
      userId,
      { user_id: userId, risk_score: riskScore, shap_top5: shapTop5 },
      (chunk) => { raw += chunk; setRawAnalysis(raw); },
      () => setAnalysisLoading(false),
    ).finally(() => setAnalysisLoading(false));
  }, [userId, riskScore, shapTop5]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  const handleSendChat = useCallback((question?: string) => {
    const q = (question ?? chatInput).trim();
    if (!q || chatLoading) return;
    setChatInput('');
    setChatMessages((prev) => [...prev, { role: 'user', text: q }]);
    setChatLoading(true);

    let aiText = '';
    setChatMessages((prev) => [...prev, { role: 'ai', text: '' }]);

    streamChat(
      userId, q, riskScore, shapTop5,
      (chunk) => {
        aiText += chunk;
        setChatMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: 'ai', text: aiText };
          return copy;
        });
      },
      () => setChatLoading(false),
    ).finally(() => setChatLoading(false));
  }, [chatInput, chatLoading, userId, riskScore, shapTop5]);

  const reportText = stripSuggestions(rawAnalysis);
  const suggestions = extractSuggestions(rawAnalysis);

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="fixed right-0 top-1/2 z-30 -translate-y-1/2 rounded-l-lg bg-brand-600 px-2 py-4 text-xs text-white shadow-lg hover:bg-brand-700"
        style={{ writingMode: 'vertical-rl' }}
      >
        LLM 分析
      </button>
    );
  }

  return (
    <div className="fixed right-0 top-0 z-20 flex h-full w-80 flex-col border-l border-surface-200 bg-white shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-surface-200 bg-surface-50 px-4 py-3">
        <span className="text-sm font-semibold text-slate-700">LLM 分析</span>
        <button onClick={() => setCollapsed(true)} className="text-slate-400 hover:text-slate-600 text-lg leading-none">&times;</button>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">
        {/* ── Report Section ── */}
        <div className="border-b border-surface-200 px-4 py-3">
          {analysisLoading && !rawAnalysis && (
            <div className="flex items-center gap-2 py-4 text-xs text-slate-400">
              <div className="h-3 w-3 animate-spin-slow rounded-full border-2 border-surface-300 border-t-brand-600" />
              分析中…
            </div>
          )}
          {reportText && (
            <div className="llm-markdown text-xs leading-relaxed text-slate-600">
              <ReactMarkdown>{reportText}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* ── Suggested Questions ── */}
        {suggestions.length > 0 && (
          <div className="border-b border-surface-200 px-4 py-3">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-slate-400">建議追問</p>
            <div className="flex flex-col gap-1.5">
              {suggestions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSendChat(q)}
                  disabled={chatLoading}
                  className="rounded-lg border border-surface-200 bg-surface-50 px-3 py-2 text-left text-xs text-slate-600 transition-colors hover:border-brand-200 hover:bg-brand-50 hover:text-brand-700 disabled:opacity-40"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Chat Section ── */}
        <div className="px-4 py-3">
          {chatMessages.length > 0 && (
            <div className="mb-2">
              {chatMessages.map((msg, i) => (
                <div key={i} className={`mb-2 rounded-lg px-3 py-2 text-xs ${
                  msg.role === 'user'
                    ? 'ml-6 bg-brand-50 text-brand-700'
                    : 'mr-2 bg-surface-50 text-slate-600'
                }`}>
                  {msg.role === 'ai' ? (
                    <div className="llm-markdown"><ReactMarkdown>{msg.text || '…'}</ReactMarkdown></div>
                  ) : msg.text}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Chat input */}
      <div className="border-t border-surface-200 px-3 py-2">
        <div className="flex gap-2">
          <input
            type="text"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendChat()}
            placeholder="對此用戶提問…"
            className="flex-1 rounded-lg border border-surface-200 px-3 py-1.5 text-xs outline-none focus:border-brand-400"
            disabled={chatLoading}
          />
          <button
            onClick={() => handleSendChat()}
            disabled={chatLoading || !chatInput.trim()}
            className="rounded-lg bg-brand-600 px-3 py-1.5 text-xs text-white hover:bg-brand-700 disabled:opacity-40"
          >
            送出
          </button>
        </div>
      </div>
    </div>
  );
};

export default LlmPanel;
