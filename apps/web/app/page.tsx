"use client";

import { useEffect, useMemo, useState } from "react";
import { cx } from "../components/ui";
import { API_BASE } from "../components/api";

type EvidenceItem = {
  url: string;
  snippet?: string;
  domain?: string;
  domain_score?: number;
  overlap?: number;
  title?: string;
};

type ToulminArg = {
  role?: string;
  attack_type?: string;
  datum?: string;
  warrant?: string;
  claim?: string;
  confidence?: number;
};

type DebateInfo = {
  prover?: ToulminArg | null;
  skeptic?: ToulminArg | null;
  judge?: any;
  accepted_arguments?: string[];
  defeated_arguments?: string[];
  model?: string;
  error?: string;
};

type ClaimItem = {
  claim_text: string;
  verdict: "SUPPORTED" | "REFUTED" | "NEI" | string;
  confidence: number;
  verdict_source?: string;
  debate_summary?: string;
  evidence: EvidenceItem[];
  excluded_evidence?: { domain: string; url: string; reason: string }[];
  debate?: DebateInfo | null;
  adjusted_verdict?: string | null;
  adjusted_confidence?: number | null;
  evidence_summary?: { reason?: string; distinct_base_domains?: number } | null;
};

type AnalyzeResponse = {
  input_type: "url" | "text" | string;
  domain?: string;
  extracted_text_chars?: number;
  extracted_text_preview?: string;
  domain_score?: number;
  domain_label?: string;
  final_misinformation_likelihood?: number;
  claims?: ClaimItem[];
  timestamp_utc?: string;
  metadata?: any;
};

type RunRow = {
  id: number | string;
  created_utc?: string;
  input_type?: string;
  verifier?: string;
  domain?: string;
  url?: string;
};

function verdictClass(v?: string | null) {
  const u = (v || "").toUpperCase();
  if (u === "SUPPORTED") return "bg-emerald-400/15 border-emerald-300/30 text-emerald-200";
  if (u === "REFUTED") return "bg-rose-400/15 border-rose-300/30 text-rose-200";
  return "bg-amber-400/15 border-amber-300/30 text-amber-200";
}

function ArgBox({ title, arg }: { title: string; arg?: ToulminArg | null }) {
  if (!arg || typeof arg !== "object") return null;
  return (
    <div className="rounded-xl border border-white/10 bg-white/5 p-3 text-sm">
      <div className="text-xs font-semibold text-white/70">
        {title}
        {arg.attack_type ? ` · ${arg.attack_type}` : ""}
        {typeof arg.confidence === "number" ? ` · ${Math.round(arg.confidence * 100)}%` : ""}
      </div>
      {arg.datum && (
        <div className="mt-1">
          <span className="text-white/50">Datum:</span> {arg.datum}
        </div>
      )}
      {arg.warrant && (
        <div className="mt-1">
          <span className="text-white/50">Warrant:</span> {arg.warrant}
        </div>
      )}
      {arg.claim && (
        <div className="mt-1">
          <span className="text-white/50">Claim:</span> {arg.claim}
        </div>
      )}
    </div>
  );
}

function fmtPct(x?: number) {
  if (typeof x !== "number") return "—";
  const v = Math.max(0, Math.min(1, x));
  return `${Math.round(v * 100)}%`;
}

function safeHostFromUrl(u?: string) {
  try {
    if (!u) return "—";
    return new URL(u).hostname;
  } catch {
    return "—";
  }
}

function sortEvidence(evs: EvidenceItem[]) {
  return [...(evs || [])].sort((a, b) => {
    const sa = typeof a.domain_score === "number" ? a.domain_score : 0;
    const sb = typeof b.domain_score === "number" ? b.domain_score : 0;
    if (sb !== sa) return sb - sa;
    const da = (a.domain ?? safeHostFromUrl(a.url)).length;
    const db = (b.domain ?? safeHostFromUrl(b.url)).length;
    return da - db;
  });
}

export default function Page() {
  const [tab, setTab] = useState<"url" | "text">("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");

  const [strictPolicy, setStrictPolicy] = useState(false);
  const [verifier, setVerifier] = useState<"baseline" | "debate">("baseline");

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxClaims, setMaxClaims] = useState(6);
  const [maxEvidence, setMaxEvidence] = useState(5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);

  const [openClaimIdx, setOpenClaimIdx] = useState<number | null>(0);

  const canAnalyze = useMemo(() => {
    if (tab === "url") return url.trim().length > 8;
    return text.trim().length > 10;
  }, [tab, url, text]);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = {
        verifier,
        enable_weighted_confidence: strictPolicy,
        max_claims: maxClaims,
        max_evidence_per_claim: maxEvidence,
      };
      if (tab === "url") payload.url = url.trim();
      else payload.text = text.trim();

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const t = await res.text();
        let detail = t;
        try {
          detail = JSON.parse(t)?.detail ?? t;
        } catch {}
        throw new Error(`API error ${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
      }

      const json = (await res.json()) as AnalyzeResponse;
      if (json.claims) {
        json.claims = json.claims.map((c) => ({ ...c, evidence: sortEvidence(c.evidence || []) }));
      }
      setResult(json);
      setOpenClaimIdx(0);

      fetchRuns();
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  async function fetchRuns() {
    setRunsLoading(true);
    setRunsError(null);

    try {
      const res = await fetch(`${API_BASE}/runs?limit=10`);
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Runs API error ${res.status}: ${t}`);
      }
      const json = await res.json();
      const items: RunRow[] = Array.isArray(json) ? json : (json?.items ?? []);
      setRuns(items);
    } catch (e: any) {
      setRuns([]);
      setRunsError(e?.message ?? "Could not load run history.");
    } finally {
      setRunsLoading(false);
    }
  }

  function exportCSV() {
    window.open(`${API_BASE}/runs-export?limit=500&format=csv`, "_blank", "noreferrer");
  }

  function exportPDF() {
    window.print();
  }

  useEffect(() => {
    fetchRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const examples = [
    "https://ourworldindata.org/co2-and-greenhouse-gas-emissions",
    "https://www.bbc.com/news",
    "https://en.wikipedia.org/wiki/Naples",
    "https://www.who.int",
  ];

  return (
    <main className="min-h-screen bg-[#070A15] text-white">
      {/* Background glow */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-24 -left-24 h-[360px] w-[360px] rounded-full blur-3xl opacity-40 bg-fuchsia-500" />
        <div className="absolute top-24 right-0 h-[420px] w-[420px] rounded-full blur-3xl opacity-35 bg-cyan-400" />
        <div className="absolute bottom-0 left-1/3 h-[420px] w-[420px] rounded-full blur-3xl opacity-25 bg-emerald-400" />
      </div>

      {/* Hero */}
      <div className="relative border-b border-white/10">
        <div className="mx-auto max-w-6xl px-4 py-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-4 py-2 text-xs text-white/80">
            <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_18px_#34d399]" />
            Live credibility-aware fact checking
          </div>

          <div className="mt-5 grid lg:grid-cols-2 gap-8 items-start">
            <div>
              <h1 className="text-3xl md:text-4xl font-semibold leading-tight">
                Fact Validator
                <span className="block text-white/70 text-lg md:text-xl mt-3">
                  Rate sources • extract claims • retrieve evidence • produce verdicts
                </span>
              </h1>

              <div className="mt-6 grid sm:grid-cols-3 gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="text-sm font-semibold">Source Credibility</div>
                  <div className="text-xs text-white/70 mt-1">Domain score + reasons</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="text-sm font-semibold">Evidence Retrieval</div>
                  <div className="text-xs text-white/70 mt-1">SERP-based lookup</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="text-sm font-semibold">Verification</div>
                  <div className="text-xs text-white/70 mt-1">Baseline / Debate mode</div>
                </div>
              </div>

              <div className="mt-6 flex gap-2 flex-wrap">
                <a className="text-sm px-3 py-2 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10" href="/source">
                  Source Credibility Checker →
                </a>
                <a
                  className="text-sm px-3 py-2 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10"
                  href={`${API_BASE}/docs`}
                  target="_blank"
                  rel="noreferrer"
                >
                  API Docs
                </a>
                <button
                  className="text-sm px-3 py-2 rounded-lg border border-white/15 bg-white/5 hover:bg-white/10"
                  onClick={exportPDF}
                  disabled={!result}
                  title="Print / Save as PDF"
                >
                  Export PDF
                </button>
              </div>
            </div>

            {/* Analyze panel */}
            <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-[0_0_40px_rgba(0,0,0,0.4)]">
              <div className="flex items-center gap-2">
                <button
                  className={cx(
                    "px-4 py-2 rounded-xl text-sm font-medium border transition",
                    tab === "url" ? "bg-white text-black border-white" : "bg-white/0 border-white/15 hover:bg-white/10"
                  )}
                  onClick={() => setTab("url")}
                >
                  URL
                </button>
                <button
                  className={cx(
                    "px-4 py-2 rounded-xl text-sm font-medium border transition",
                    tab === "text" ? "bg-white text-black border-white" : "bg-white/0 border-white/15 hover:bg-white/10"
                  )}
                  onClick={() => setTab("text")}
                >
                  Text
                </button>

                <div className="flex-1" />

                <button
                  className="text-xs px-3 py-2 rounded-xl border border-white/15 hover:bg-white/10"
                  onClick={() => setShowAdvanced((v) => !v)}
                >
                  {showAdvanced ? "Hide Advanced" : "Advanced"}
                </button>
              </div>

              <div className="mt-4 grid gap-3">
                <div className="grid sm:grid-cols-2 gap-3">
                  <label className="text-xs text-white/70 flex flex-col">
                    Evidence policy
                    <span className="mt-1 flex items-center gap-2 bg-black/30 border border-white/15 rounded-xl px-3 py-2 text-sm">
                      <input
                        type="checkbox"
                        checked={strictPolicy}
                        onChange={(e) => setStrictPolicy(e.target.checked)}
                      />
                      Strict (credible + independent sources)
                    </span>
                  </label>

                  <label className="text-xs text-white/70">
                    Verifier
                    <select
                      className="mt-1 w-full bg-black/30 border border-white/15 rounded-xl px-3 py-2 text-sm outline-none"
                      value={verifier}
                      onChange={(e) => setVerifier(e.target.value as any)}
                    >
                      <option value="baseline">baseline</option>
                      <option value="debate">debate</option>
                    </select>
                  </label>
                </div>

                {showAdvanced && (
                  <div className="grid sm:grid-cols-2 gap-3">
                    <label className="text-xs text-white/70">
                      Max Claims
                      <input
                        type="number"
                        min={1}
                        max={12}
                        className="mt-1 w-full bg-black/30 border border-white/15 rounded-xl px-3 py-2 text-sm outline-none"
                        value={maxClaims}
                        onChange={(e) => setMaxClaims(parseInt(e.target.value || "6", 10))}
                      />
                    </label>
                    <label className="text-xs text-white/70">
                      Max Evidence / Claim
                      <input
                        type="number"
                        min={1}
                        max={10}
                        className="mt-1 w-full bg-black/30 border border-white/15 rounded-xl px-3 py-2 text-sm outline-none"
                        value={maxEvidence}
                        onChange={(e) => setMaxEvidence(parseInt(e.target.value || "5", 10))}
                      />
                    </label>
                  </div>
                )}

                {tab === "url" ? (
                  <div>
                    <div className="text-xs text-white/70">URL</div>
                    <input
                      className="mt-1 w-full bg-black/30 border border-white/15 rounded-xl px-3 py-3 text-sm outline-none"
                      placeholder="https://example.com/article"
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                    />
                    <div className="mt-2 flex gap-2 flex-wrap">
                      {examples.map((x) => (
                        <button
                          key={x}
                          className="text-xs px-3 py-1.5 rounded-full border border-white/15 hover:bg-white/10"
                          onClick={() => setUrl(x)}
                        >
                          Use example
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="text-xs text-white/70">Text</div>
                    <textarea
                      className="mt-1 w-full bg-black/30 border border-white/15 rounded-xl px-3 py-3 text-sm min-h-[140px] outline-none"
                      placeholder="Paste a claim or paragraph..."
                      value={text}
                      onChange={(e) => setText(e.target.value)}
                    />
                  </div>
                )}

                <button
                  className={cx(
                    "w-full rounded-xl px-4 py-3 text-sm font-semibold transition",
                    !canAnalyze || loading
                      ? "bg-white/15 text-white/50 cursor-not-allowed"
                      : "bg-gradient-to-r from-fuchsia-500 to-cyan-400 text-black hover:opacity-95"
                  )}
                  onClick={analyze}
                  disabled={!canAnalyze || loading}
                >
                  {loading ? "Analyzing…" : "Analyze"}
                </button>

                {error && (
                  <div className="rounded-xl border border-rose-300/30 bg-rose-500/10 text-rose-200 p-3 text-sm">
                    <b>Error:</b> {error}
                    <div className="mt-2 text-xs text-rose-200/80">
                      If “Failed to fetch”, check backend is running at <span className="font-mono">{API_BASE}</span>.
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="relative mx-auto max-w-6xl px-4 py-8 grid gap-6">
        {result && (
          <div className="grid md:grid-cols-3 gap-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="text-xs text-white/70">Input</div>
              <div className="mt-2 text-sm">
                Type: <span className="font-mono">{String(result.input_type).toUpperCase()}</span>
              </div>
              <div className="mt-1 text-sm">
                Domain: <span className="font-mono">{result.domain ?? "—"}</span>
              </div>
              <div className="mt-2 text-xs text-white/50">{result.timestamp_utc ? `UTC: ${result.timestamp_utc}` : ""}</div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="text-xs text-white/70">Source Credibility</div>
              <div className="mt-3 inline-flex items-center gap-2">
                <span className="px-3 py-1 rounded-full text-xs font-semibold border border-white/10 bg-white/10">
                  {typeof result.domain_score === "number" ? result.domain_score : "—"} {result.domain_label ?? ""}
                </span>
              </div>
              <div className="mt-3 text-xs text-white/60">Heuristic score — a risk signal, not ground truth.</div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="text-xs text-white/70">Misinformation Likelihood</div>
              <div className="mt-2 text-3xl font-semibold">{fmtPct(result.final_misinformation_likelihood)}</div>
              <div className="mt-3 h-2 rounded-full bg-white/10 overflow-hidden">
                <div
                  className="h-2 bg-gradient-to-r from-emerald-400 via-amber-300 to-rose-400"
                  style={{
                    width:
                      typeof result.final_misinformation_likelihood === "number"
                        ? `${Math.round(Math.max(0, Math.min(1, result.final_misinformation_likelihood)) * 100)}%`
                        : "0%",
                  }}
                />
              </div>
              <div className="mt-3 text-xs text-white/60">
                Mean probability that a claim is false (REFUTED = confidence, SUPPORTED = 1 − confidence, NEI = 50%).
              </div>
            </div>
          </div>
        )}

        {result && (
          <div className="grid lg:grid-cols-3 gap-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="text-sm font-semibold">Extracted Text Preview</div>
              <div className="mt-2 text-xs text-white/60">chars: {result.extracted_text_chars ?? "—"}</div>
              <div className="mt-3 text-sm text-white/80 leading-relaxed">{result.extracted_text_preview ?? "—"}</div>
            </div>

            <div className="lg:col-span-2 rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="text-sm font-semibold">Claims</div>
              <div className="mt-2 text-xs text-white/60">Click a claim to expand evidence.</div>
              {(result.claims ?? []).length === 0 && (
                <div className="mt-4 rounded-xl border border-amber-300/30 bg-amber-500/10 text-amber-100 p-3 text-sm">
                  {result.metadata?.warning ?? "No checkable claim was found in the input."} Try a full sentence stating a
                  fact, e.g. “Paris is the capital of France.”
                </div>
              )}

              <div className="mt-4 grid gap-3">
                {(result.claims ?? []).map((c, idx) => {
                  const open = openClaimIdx === idx;
                  return (
                    <div key={idx} className="rounded-2xl border border-white/10 overflow-hidden bg-black/20">
                      <button
                        className="w-full text-left p-4 hover:bg-white/5 transition"
                        onClick={() => setOpenClaimIdx(open ? null : idx)}
                      >
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-xs text-white/60">Claim #{idx + 1}</div>
                            <div className="mt-1 text-sm font-medium text-white">{c.claim_text}</div>
                            <div className="mt-2 flex items-center gap-2 flex-wrap">
                              <span className={cx("px-3 py-1 rounded-full text-xs font-semibold border", verdictClass(c.verdict))}>
                                {(c.verdict || "NEI").toUpperCase()} • {fmtPct(c.confidence)}
                              </span>
                              {c.adjusted_verdict && (
                                <span
                                  className={cx("px-3 py-1 rounded-full text-xs font-semibold border", verdictClass(c.adjusted_verdict))}
                                  title="Verdict under the strict evidence policy"
                                >
                                  Strict: {c.adjusted_verdict} • {fmtPct(c.adjusted_confidence ?? undefined)}
                                </span>
                              )}
                              <span className="text-xs text-white/60">Evidence: {(c.evidence || []).length}</span>
                              {(c.excluded_evidence || []).length > 0 && (
                                <span className="text-xs text-white/40">
                                  ({(c.excluded_evidence || []).length} excluded as non-independent)
                                </span>
                              )}
                              {c.verdict_source && <span className="text-xs text-white/40">via {c.verdict_source}</span>}
                            </div>
                          </div>
                          <div className="text-xs text-white/60">{open ? "▲" : "▼"}</div>
                        </div>
                      </button>

                      {open && (
                        <div className="p-4 border-t border-white/10 bg-black/10">
                          {c.debate_summary && (
                            <div className="text-sm text-white/80">
                              <span className="text-white/60">Explanation:</span> {c.debate_summary}
                            </div>
                          )}

                          {c.debate && !c.debate.error && (
                            <div className="mt-4">
                              <div className="text-sm font-semibold">Debate (Toulmin arguments)</div>
                              <div className="mt-2 grid md:grid-cols-2 gap-2">
                                <ArgBox title="PROVER (support)" arg={c.debate.prover} />
                                <ArgBox title="SKEPTIC (attack)" arg={c.debate.skeptic} />
                              </div>
                              <div className="mt-2 text-xs text-white/60">
                                Judge accepted: {(c.debate.accepted_arguments || []).join(", ") || "—"} · defeated:{" "}
                                {(c.debate.defeated_arguments || []).join(", ") || "—"}
                                {c.debate.model ? ` · model: ${c.debate.model}` : ""}
                              </div>
                            </div>
                          )}
                          {c.debate?.error && (
                            <div className="mt-3 text-xs text-rose-200/80">Debate failed, heuristic verdict used: {c.debate.error}</div>
                          )}

                          <div className="mt-4 text-sm font-semibold">Evidence</div>

                          {(c.evidence || []).length === 0 ? (
                            <div className="mt-2 text-sm text-white/60">No evidence items returned.</div>
                          ) : (
                            <ul className="mt-2 grid gap-2">
                              {sortEvidence(c.evidence || []).map((ev, j) => {
                                const d = ev.domain ?? safeHostFromUrl(ev.url);
                                const s = typeof ev.domain_score === "number" ? ev.domain_score : undefined;

                                return (
                                  <li key={j} className="rounded-2xl border border-white/10 bg-white/5 p-3">
                                    <div className="flex items-center justify-between gap-3">
                                      <div className="min-w-0">
                                        <div className="flex items-center gap-2 flex-wrap">
                                          <span className="text-sm font-semibold truncate">{d}</span>
                                          {typeof s === "number" && (
                                            <span
                                              className={cx(
                                                "px-2 py-1 rounded-full text-xs font-semibold border",
                                                s >= 80
                                                  ? "bg-emerald-400/15 border-emerald-300/30 text-emerald-200"
                                                  : s >= 50
                                                  ? "bg-amber-400/15 border-amber-300/30 text-amber-200"
                                                  : "bg-rose-400/15 border-rose-300/30 text-rose-200"
                                              )}
                                            >
                                              {s}
                                            </span>
                                          )}
                                        </div>
                                      </div>

                                      <a
                                        className="text-xs px-3 py-1.5 rounded-full border border-white/15 hover:bg-white/10"
                                        href={ev.url}
                                        target="_blank"
                                        rel="noreferrer"
                                      >
                                        Open
                                      </a>
                                    </div>

                                    {ev.snippet && <div className="mt-2 text-sm text-white/80 leading-snug">{ev.snippet}</div>}
                                  </li>
                                );
                              })}
                            </ul>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <details className="mt-5 rounded-2xl border border-white/10 bg-black/20 p-4">
                <summary className="cursor-pointer text-sm font-semibold text-white/90">Raw JSON (debug)</summary>
                <pre className="mt-4 text-xs overflow-auto bg-black/50 text-white rounded-xl p-4">
{JSON.stringify(result, null, 2)}
                </pre>
              </details>
            </div>
          </div>
        )}

        <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div>
              <div className="text-sm font-semibold">Run History (last 10)</div>
              <div className="text-xs text-white/60">Stored runs for reproducibility (exportable).</div>
            </div>
            <div className="flex items-center gap-2">
              <button
                className="text-xs px-3 py-2 rounded-xl border border-white/15 hover:bg-white/10"
                onClick={fetchRuns}
                disabled={runsLoading}
              >
                {runsLoading ? "Refreshing…" : "Refresh"}
              </button>
              <button className="text-xs px-3 py-2 rounded-xl border border-white/15 hover:bg-white/10" onClick={exportCSV}>
                Export CSV
              </button>
            </div>
          </div>

          {runsError ? (
            <div className="mt-4 text-sm text-white/70">
              <b>Run history not available.</b> {runsError}
            </div>
          ) : runs.length === 0 ? (
            <div className="mt-4 text-sm text-white/70">No runs found yet.</div>
          ) : (
            <div className="mt-4 overflow-auto border border-white/10 rounded-xl">
              <table className="min-w-full text-sm">
                <thead className="bg-white/5 text-white/70">
                  <tr>
                    <th className="text-left font-medium px-3 py-2">ID</th>
                    <th className="text-left font-medium px-3 py-2">Time (UTC)</th>
                    <th className="text-left font-medium px-3 py-2">Type</th>
                    <th className="text-left font-medium px-3 py-2">Verifier</th>
                    <th className="text-left font-medium px-3 py-2">Domain</th>
                    <th className="text-left font-medium px-3 py-2">Report</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r, i) => (
                    <tr key={String(r.id) + "_" + i} className="border-t border-white/10">
                      <td className="px-3 py-2">{r.id}</td>
                      <td className="px-3 py-2 font-mono text-xs text-white/70">{r.created_utc ?? "—"}</td>
                      <td className="px-3 py-2 text-white/80">{r.input_type ?? "—"}</td>
                      <td className="px-3 py-2 text-white/80">{r.verifier ?? "—"}</td>
                      <td className="px-3 py-2 text-white/80">{r.domain ?? safeHostFromUrl(r.url)}</td>
                      <td className="px-3 py-2">
                        <a
                          className="text-xs px-3 py-1.5 rounded-full border border-white/15 hover:bg-white/10"
                          href={`/run/${r.id}`}
                        >
                          Open report
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="text-xs text-white/50 text-center pb-8">
          Credibility score is heuristic and should be interpreted as a risk signal, not ground truth.
        </div>
      </div>

      {/* Print styling */}
      <style jsx global>{`
        @media print {
          body {
            background: white !important;
            color: black !important;
          }
          a {
            color: black !important;
          }
          button,
          select,
          input,
          textarea,
          summary {
            display: none !important;
          }
        }
      `}</style>
    </main>
  );
}
