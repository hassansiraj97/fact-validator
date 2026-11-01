"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { API_BASE } from "../../../components/api";

function downloadJson(filename: string, obj: any) {
  const json = JSON.stringify(obj, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function Arg({ title, arg }: { title: string; arg: any }) {
  if (!arg || typeof arg !== "object") return null;
  return (
    <div className="text-sm border rounded p-2">
      <div className="text-xs font-semibold text-gray-600">
        {title}
        {arg.attack_type ? ` · ${arg.attack_type}` : ""}
      </div>
      {arg.datum && (
        <div>
          <span className="text-gray-500">Datum:</span> {arg.datum}
        </div>
      )}
      {arg.warrant && (
        <div>
          <span className="text-gray-500">Warrant:</span> {arg.warrant}
        </div>
      )}
      {arg.claim && (
        <div>
          <span className="text-gray-500">Claim:</span> {arg.claim}
        </div>
      )}
    </div>
  );
}

export default function RunPage() {
  const params = useParams<{ id: string }>();
  const id = params?.id;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<any>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/runs/${id}`);
        if (!res.ok) throw new Error(`Run load error: ${res.status}`);
        setRun(await res.json());
      } catch (e: any) {
        setError(e?.message ?? "Failed to load run");
      } finally {
        setLoading(false);
      }
    }
    if (id) load();
  }, [id]);

  // The stored row wraps the analysis result in `response`
  const result = run ? run.response ?? run : null;

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto bg-white text-gray-900">
      <style jsx global>{`
        @media print {
          .no-print {
            display: none !important;
          }
        }
      `}</style>

      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-semibold">Saved Report</h1>
          <p className="text-sm text-gray-600 mt-1">
            Run ID: {id} {run?.created_utc ? `· ${run.created_utc}` : ""} {run?.verifier ? `· verifier: ${run.verifier}` : ""}
          </p>
          {run?.url && <p className="text-sm text-gray-600 break-all">{run.url}</p>}
        </div>

        <div className="no-print flex gap-2">
          <a className="px-3 py-2 border rounded text-sm underline" href="/">
            Back
          </a>
          <button className="px-3 py-2 border rounded text-sm" onClick={() => window.print()}>
            Print / Save PDF
          </button>
          {run && (
            <button className="px-3 py-2 border rounded text-sm" onClick={() => downloadJson(`fact-validator-run-${id}.json`, run)}>
              Download JSON
            </button>
          )}
        </div>
      </div>

      {loading && <div className="mt-6 text-sm text-gray-700">Loading…</div>}
      {error && <div className="mt-6 text-sm text-red-700">{error}</div>}

      {result && (
        <div className="mt-6 p-4 border rounded bg-white">
          <div className="flex flex-wrap gap-6 text-sm">
            <div>
              <div className="text-gray-500">Input Type</div>
              <div className="font-medium">{result.input_type}</div>
            </div>
            <div>
              <div className="text-gray-500">Domain</div>
              <div className="font-medium">{result.domain ?? "-"}</div>
            </div>
            <div>
              <div className="text-gray-500">Extracted Text Chars</div>
              <div className="font-medium">{result.extracted_text_chars}</div>
            </div>
            <div>
              <div className="text-gray-500">Domain Score</div>
              <div className="font-medium">
                {result.domain_score ?? "-"} {result.domain_label ? `(${result.domain_label})` : ""}
              </div>
            </div>
            <div>
              <div className="text-gray-500">Misinformation Likelihood</div>
              <div className="font-medium">
                {typeof result.final_misinformation_likelihood === "number"
                  ? `${Math.round(result.final_misinformation_likelihood * 100)}%`
                  : "-"}
              </div>
            </div>
          </div>

          <div className="mt-4">
            <div className="text-gray-500 text-sm">Extracted Text Preview</div>
            <div className="text-sm mt-1">{result.extracted_text_preview}</div>
          </div>

          <div className="mt-6">
            <div className="text-gray-800 font-semibold">Claims</div>
            <div className="mt-2 space-y-3">
              {(result.claims ?? []).map((c: any, idx: number) => (
                <div key={idx} className="p-3 border rounded">
                  <div className="text-sm">{c.claim_text}</div>
                  <div className="mt-2 text-sm">
                    <span className="font-medium">{c.verdict}</span>{" "}
                    <span className="text-gray-500">({Math.round((c.confidence ?? 0) * 100)}%)</span>
                    {c.verdict_source && <span className="text-gray-500"> · via {c.verdict_source}</span>}
                    {c.adjusted_verdict && <span className="text-gray-500"> · strict policy: {c.adjusted_verdict}</span>}
                  </div>

                  {c.debate_summary && (
                    <div className="mt-2 text-sm text-gray-700">
                      <span className="text-gray-500">Explanation:</span> {c.debate_summary}
                    </div>
                  )}

                  {c.debate && !c.debate.error && (
                    <div className="mt-2 grid md:grid-cols-2 gap-2">
                      <Arg title="PROVER" arg={c.debate.prover} />
                      <Arg title="SKEPTIC" arg={c.debate.skeptic} />
                    </div>
                  )}

                  <div className="mt-3 text-sm text-gray-800 font-medium">Evidence</div>
                  <ul className="mt-1 space-y-2">
                    {(c.evidence ?? []).map((e: any, j: number) => (
                      <li key={j} className="text-sm">
                        <div className="font-medium">
                          {e.domain} <span className="text-gray-500">(credibility: {e.domain_score})</span>
                        </div>
                        <div className="text-gray-700">{e.snippet}</div>
                        <div className="no-print">
                          <a className="text-blue-700 underline" href={e.url} target="_blank" rel="noreferrer">
                            Open source
                          </a>
                        </div>
                        <div className="hidden print:block text-xs text-gray-700">Source URL: {e.url}</div>
                      </li>
                    ))}
                    {(c.evidence ?? []).length === 0 && <li className="text-sm text-gray-500">No evidence retrieved.</li>}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          <details className="mt-6 no-print">
            <summary className="cursor-pointer text-sm text-gray-700">Raw JSON (debug)</summary>
            <pre className="mt-2 text-xs overflow-auto p-3 border rounded bg-gray-50">{JSON.stringify(run, null, 2)}</pre>
          </details>
        </div>
      )}
    </main>
  );
}
