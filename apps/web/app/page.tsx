"use client";

import { useEffect, useState } from "react";

type RunItem = {
  id: number;
  created_at_utc: string;
  input_type: string;
  domain?: string | null;
  url?: string | null;
  mode?: string | null;
  verifier?: string | null;
};

function labelFromScore(score: number): "HIGH" | "MEDIUM" | "LOW" {
  if (score >= 80) return "HIGH";
  if (score >= 50) return "MEDIUM";
  return "LOW";
}

function badgeClass(label: string) {
  if (label === "HIGH") return "bg-green-100 text-green-800 border-green-300";
  if (label === "MEDIUM") return "bg-yellow-100 text-yellow-800 border-yellow-300";
  return "bg-red-100 text-red-800 border-red-300";
}

function SourceBadge({ score }: { score: number }) {
  const label = labelFromScore(score);
  return (
    <span className={`text-[11px] px-2 py-[2px] border rounded ${badgeClass(label)}`}>
      {label} {score}
    </span>
  );
}

function VerdictPill({ v, c }: { v: string; c?: number }) {
  const color =
    v === "SUPPORTED"
      ? "bg-green-100 text-green-800 border-green-300"
      : v === "REFUTED"
      ? "bg-red-100 text-red-800 border-red-300"
      : "bg-gray-100 text-gray-800 border-gray-300";
  return (
    <span className={`text-xs px-2 py-1 border rounded ${color}`}>
      {v}{typeof c === "number" ? ` (${c})` : ""}
    </span>
  );
}

export default function Home() {
  const API = "http://127.0.0.1:8000";

  const [url, setUrl] = useState("");
  const [text, setText] = useState("");
  const [mode, setMode] = useState<"live" | "snapshot">("live");
  const [verifier, setVerifier] = useState<"baseline" | "debate">("baseline");

  // Step 11 controls
  const [minSourceScore, setMinSourceScore] = useState<number>(50);
  const [requireIndependent, setRequireIndependent] = useState<boolean>(true);
  const [minOverlap, setMinOverlap] = useState<number>(6);
  const [maxPerBase, setMaxPerBase] = useState<number>(2);
  const [enableWeighted, setEnableWeighted] = useState<boolean>(true);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const [runs, setRuns] = useState<RunItem[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);

  async function fetchRuns() {
    setRunsLoading(true);
    try {
      const res = await fetch(`${API}/runs?limit=10`);
      if (!res.ok) throw new Error(`Runs API error: ${res.status}`);
      const data = await res.json();
      setRuns(data.items ?? []);
    } catch {
      // ignore
    } finally {
      setRunsLoading(false);
    }
  }

  useEffect(() => {
    fetchRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = {
        mode,
        verifier,
        max_claims: 6,
        max_evidence_per_claim: 5,

        // Step 11 payload
        min_source_score: minSourceScore,
        require_independent_domains: requireIndependent,
        min_overlap: minOverlap,
        max_sources_per_base_domain: maxPerBase,
        enable_weighted_confidence: enableWeighted,
      };

      if (url.trim()) payload.url = url.trim();
      if (text.trim()) payload.text = text.trim();

      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const msg = await res.text();
        throw new Error(`API error ${res.status}: ${msg}`);
      }

      const data = await res.json();
      setResult(data);
      fetchRuns();
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  async function exportRuns() {
    const res = await fetch(`${API}/runs-export?limit=200`);
    if (!res.ok) {
      alert(`Export failed: ${res.status}`);
      return;
    }
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "fact-validator-runs.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const overallLabel = result?.domain_label ?? null;

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Fact Validator (Working Demo)</h1>
          <p className="text-sm text-gray-600 mt-2">
            Step 11 enabled: credibility-weighted aggregation + independence constraints (showing adjusted verdicts).
          </p>
        </div>

        <a className="underline text-sm mt-1" href="/source">
          Source Credibility Checker →
        </a>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        {/* Input */}
        <div className="border rounded p-4">
          <label className="block text-sm font-medium">URL (optional)</label>
          <input
            className="w-full mt-2 p-2 border rounded"
            placeholder="https://example.com/article"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />

          <label className="block text-sm font-medium mt-4">Text (optional)</label>
          <textarea
            className="w-full mt-2 p-2 border rounded min-h-[140px]"
            placeholder="Paste a claim or article text..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />

          <div className="flex items-center gap-3 mt-4 flex-wrap">
            <label className="text-sm font-medium">Mode</label>
            <select className="border rounded p-2" value={mode} onChange={(e) => setMode(e.target.value as any)}>
              <option value="live">live</option>
              <option value="snapshot">snapshot</option>
            </select>

            <label className="text-sm font-medium ml-0 lg:ml-2">Verifier</label>
            <select className="border rounded p-2" value={verifier} onChange={(e) => setVerifier(e.target.value as any)}>
              <option value="baseline">baseline</option>
              <option value="debate">debate (LLM PROVER/SKEPTIC/JUDGE)</option>
            </select>
          </div>

          {/* Step 11 controls */}
          <div className="mt-5 border rounded p-3 bg-gray-50">
            <div className="font-medium text-sm">Step 11 Controls</div>

            <div className="mt-3 grid gap-3">
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm">Min source score: <b>{minSourceScore}</b></label>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={minSourceScore}
                  onChange={(e) => setMinSourceScore(Number(e.target.value))}
                  className="w-full"
                />
                <div className="text-xs text-gray-600">Higher = stricter filtering of evidence sources.</div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={requireIndependent}
                  onChange={(e) => setRequireIndependent(e.target.checked)}
                />
                <label className="text-sm">Require 2 independent base domains to mark SUPPORTED</label>
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <label className="text-sm">Min overlap</label>
                <input
                  className="border rounded p-2 w-24"
                  type="number"
                  value={minOverlap}
                  min={1}
                  max={20}
                  onChange={(e) => setMinOverlap(Number(e.target.value))}
                />

                <label className="text-sm">Max sources per base domain</label>
                <input
                  className="border rounded p-2 w-24"
                  type="number"
                  value={maxPerBase}
                  min={1}
                  max={10}
                  onChange={(e) => setMaxPerBase(Number(e.target.value))}
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={enableWeighted}
                  onChange={(e) => setEnableWeighted(e.target.checked)}
                />
                <label className="text-sm">Enable credibility-weighted confidence</label>
              </div>
            </div>
          </div>

          <button
            className="mt-4 px-4 py-2 rounded bg-black text-white disabled:opacity-50"
            disabled={loading || (!url.trim() && !text.trim())}
            onClick={analyze}
          >
            {loading ? "Analyzing..." : "Analyze"}
          </button>

          {error && (
            <div className="mt-4 p-3 rounded border border-red-300 bg-red-50 text-red-700">{error}</div>
          )}
        </div>

        {/* Result */}
        <div className="border rounded p-4">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <h2 className="text-lg font-semibold">Result</h2>
            {overallLabel && (
              <span className={`text-xs px-2 py-1 border rounded ${badgeClass(overallLabel)}`}>
                Domain: {overallLabel} ({result.domain_score})
              </span>
            )}
          </div>

          {!result && <p className="text-sm text-gray-600 mt-2">No result yet.</p>}

          {result && (
            <div className="mt-4 text-sm">
              <div><b>Input Type:</b> {result.input_type}</div>
              <div><b>Domain:</b> {result.domain ?? "-"}</div>
              <div><b>Domain Score:</b> {result.domain_score} ({result.domain_label})</div>
              <div><b>Misinformation Likelihood:</b> {result.final_misinformation_likelihood}</div>

              <div className="mt-3">
                <b>Extracted Text Preview</b>
                <div className="mt-1 p-2 border rounded bg-gray-50">{result.extracted_text_preview}</div>
              </div>

              <div className="mt-4">
                <b>Claims</b>
                <div className="mt-2 grid gap-3">
                  {(result.claims ?? []).map((c: any, idx: number) => (
                    <div key={idx} className="border rounded p-3">
                      <div className="font-medium">{c.claim_text}</div>

                      <div className="mt-2 flex items-center gap-2 flex-wrap">
                        <span className="text-xs text-gray-600">Baseline:</span>
                        <VerdictPill v={c.verdict} c={c.confidence} />

                        <span className="text-xs text-gray-600 ml-2">Adjusted (Step 11):</span>
                        <VerdictPill v={c.adjusted_verdict ?? c.verdict} c={c.adjusted_confidence ?? c.confidence} />
                      </div>

                      <div className="mt-2 text-xs text-gray-700">
                        {c.debate_summary}
                      </div>

                      {/* Step 11 Evidence Summary */}
                      {c.evidence_summary && (
                        <div className="mt-3 p-2 border rounded bg-gray-50 text-xs">
                          <div><b>Policy:</b> min_score={c.evidence_summary.policy?.min_source_score}, min_overlap={c.evidence_summary.policy?.min_overlap}, independent={String(c.evidence_summary.policy?.require_independent_domains)}</div>
                          <div><b>Distinct base domains:</b> {c.evidence_summary.distinct_base_domains}</div>
                          {Array.isArray(c.evidence_summary.strong_base_domains) && c.evidence_summary.strong_base_domains.length > 0 && (
                            <div><b>Strong domains:</b> {c.evidence_summary.strong_base_domains.join(", ")}</div>
                          )}
                          <div className="mt-1"><b>Reason:</b> {c.evidence_summary.reason}</div>
                        </div>
                      )}

                      <div className="mt-3">
                        <b>Evidence</b>
                        <ul className="list-disc pl-6 mt-1">
                          {(c.evidence ?? []).slice(0, 8).map((e: any, j: number) => (
                            <li key={j} className="mt-1">
                              <span className="font-medium">{e.domain}</span>{" "}
                              <SourceBadge score={Number(e.domain_score ?? 50)} />{" "}
                              <span className="text-xs text-gray-600">overlap={e.overlap}</span>{" "}
                              —{" "}
                              <a className="underline" href={e.url} target="_blank" rel="noreferrer">
                                open
                              </a>{" "}
                              |{" "}
                              <a className="underline" href={`/source?d=${encodeURIComponent(e.domain)}`}>
                                check source
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <details className="mt-3">
                        <summary className="cursor-pointer text-xs font-medium">Top sources (Step 11)</summary>
                        <pre className="mt-2 p-2 bg-gray-100 rounded overflow-auto text-[11px]">
                          {JSON.stringify(c.evidence_summary?.top_sources ?? [], null, 2)}
                        </pre>
                      </details>
                    </div>
                  ))}
                </div>
              </div>

              <details className="mt-4">
                <summary className="cursor-pointer font-medium">Raw JSON</summary>
                <pre className="mt-2 p-3 bg-gray-100 rounded overflow-auto text-xs">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>

      {/* Run History */}
      <div className="mt-8 border rounded p-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <h2 className="text-lg font-semibold">Run History (last 10)</h2>
          <div className="flex gap-2">
            <button className="px-3 py-2 border rounded" onClick={fetchRuns} disabled={runsLoading}>
              {runsLoading ? "Refreshing..." : "Refresh"}
            </button>
            <button className="px-3 py-2 border rounded" onClick={exportRuns}>
              Export JSON
            </button>
          </div>
        </div>

        <div className="mt-3 overflow-auto">
          <table className="w-full text-sm border">
            <thead className="bg-gray-50">
              <tr>
                <th className="text-left p-2 border">ID</th>
                <th className="text-left p-2 border">Time (UTC)</th>
                <th className="text-left p-2 border">Type</th>
                <th className="text-left p-2 border">Domain</th>
                <th className="text-left p-2 border">URL</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id}>
                  <td className="p-2 border">{r.id}</td>
                  <td className="p-2 border">{r.created_at_utc}</td>
                  <td className="p-2 border">{r.input_type}</td>
                  <td className="p-2 border">{r.domain ?? "-"}</td>
                  <td className="p-2 border truncate max-w-[420px]">
                    {r.url ? (
                      <a className="underline" href={r.url} target="_blank" rel="noreferrer">
                        {r.url}
                      </a>
                    ) : (
                      "-"
                    )}
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td className="p-2 border" colSpan={5}>
                    No runs yet. Click Analyze at least once.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
