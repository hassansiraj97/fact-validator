"use client";

import { useMemo, useState } from "react";
import { Button, Card, Divider, cx } from "../../components/ui";
import { API_BASE } from "../../components/api";

type SourceScore = {
  domain: string;
  base_domain?: string;
  score: number;
  label: string;
  reasons?: Record<string, string>;
  timestamp_utc?: string;
  disclaimer?: string;
};

export default function SourcePage() {

  const [domain, setDomain] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SourceScore | null>(null);

  const canCheck = useMemo(() => domain.trim().length >= 3, [domain]);

  async function check() {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const d = domain.trim().replace(/^https?:\/\//, "").replace(/\/.*$/, "");
      const res = await fetch(`${API_BASE}/source/${encodeURIComponent(d)}`);

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`API error ${res.status}: ${t}`);
      }

      const json = (await res.json()) as SourceScore;
      setData(json);
    } catch (e: any) {
      setError(e?.message ?? "Failed to fetch");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-900">
      <div className="border-b bg-white">
        <div className="mx-auto max-w-4xl px-4 py-5 flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Source Credibility Checker</h1>
            <p className="text-sm text-zinc-600">Enter a domain to get a heuristic credibility score and reasons</p>
          </div>
          <a className="text-sm px-3 py-2 rounded-lg border hover:bg-zinc-50" href="/">
            ← Back
          </a>
        </div>
      </div>

      <div className="mx-auto max-w-4xl px-4 py-8 grid gap-6">
        <Card title="Check a domain" subtitle="Examples: bbc.com, who.int, wikipedia.org">
          <div>
            <div className="text-sm font-medium">Domain</div>
            <input
              className="mt-2 w-full border rounded-xl px-3 py-3 text-sm outline-none focus:ring-2 focus:ring-zinc-200"
              placeholder="bbc.com"
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
            />

            <div className="mt-4 flex items-center gap-2">
              <Button onClick={check} disabled={!canCheck || loading}>
                {loading ? "Checking…" : "Check"}
              </Button>

              <a className="text-sm px-3 py-2 rounded-lg border hover:bg-zinc-50" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
                API Docs
              </a>
            </div>

            {error && (
              <div className="mt-4 rounded-xl border border-rose-200 bg-rose-50 text-rose-700 p-4 text-sm">
                <b>Error:</b> {error}
              </div>
            )}
          </div>
        </Card>

        {data && (
          <Card title="Result" subtitle={data.timestamp_utc ? `UTC: ${data.timestamp_utc}` : undefined}>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-3 py-1 rounded-full text-xs font-semibold border bg-zinc-50">
                {data.score} {data.label}
              </span>
              <span className="text-sm text-zinc-700">
                Domain: <span className="font-mono">{data.domain}</span>
              </span>
            </div>

            <Divider />

            <div className="text-sm font-semibold">Reasons</div>
            {data.reasons && Object.keys(data.reasons).length > 0 ? (
              <ul className="mt-2 grid gap-2">
                {Object.entries(data.reasons).map(([k, v]) => (
                  <li key={k} className="border rounded-xl p-3 text-sm text-zinc-700">
                    <div className="text-xs text-zinc-500">{k}</div>
                    <div className="mt-1">{v}</div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="mt-2 text-sm text-zinc-500">No reasons provided.</div>
            )}

            {data.disclaimer && (
              <>
                <Divider />
                <div className="text-xs text-zinc-500">{data.disclaimer}</div>
              </>
            )}
          </Card>
        )}
      </div>

      <footer className="py-10 text-center text-xs text-zinc-500">
        Credibility score is heuristic and should be interpreted as a risk signal.
      </footer>
    </main>
  );
}
