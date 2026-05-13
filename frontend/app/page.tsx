"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FileText,
  Loader2,
  UploadCloud,
} from "lucide-react";

import { apiConfig, uploadCard } from "@/lib/api";
import type { IngestionResult } from "@/lib/types";
import { formatDate } from "@/lib/format";

export default function UploadPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<IngestionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(apiConfig.MOCK);

  const handleFile = useCallback(async (file: File) => {
    setBusy(true);
    setError(null);
    setResult(null);
    setFilename(file.name);
    try {
      const res = await uploadCard(file);
      setResult(res);
      if (!res.success) {
        setError(res.errors.join("; ") || "Parsing failed");
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setBusy(false);
    }
  }, []);

  const handleDemo = useCallback(async () => {
    // Synthesise a "file" upload using a tiny placeholder PDF-ish blob.
    const blob = new Blob(["demo"], { type: "application/pdf" });
    const file = new File([blob], "demo-CD-2026-05-10.pdf", {
      type: "application/pdf",
    });
    setDemoMode(true);
    await handleFile(file);
  }, [handleFile]);

  const cardId =
    result?.card?.card_id ??
    result?.card_id ??
    // Backend persists card to DB but currently doesn't echo the ID — fall
    // back to a synthetic identifier so navigation still works in demo mode.
    (result?.card ? "latest" : null);

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50">
          Pari-Mutuel Analytics
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload a Brisnet UP, Equibase, or DRF race-card PDF. The pipeline
          extracts every horse, prices each bet, and proposes a 1/4-Kelly,
          CVaR-constrained portfolio.
        </p>
        <div className="mt-2 flex items-center gap-3 text-xxs text-slate-500">
          <span>
            API: <span className="mono">{apiConfig.API_BASE}</span>
          </span>
          <span>·</span>
          <span>
            mode:{" "}
            <span className={demoMode ? "text-amber-400" : "text-emerald-400"}>
              {demoMode ? "demo / mock" : "live"}
            </span>
          </span>
        </div>
      </div>

      <div
        className={[
          "relative rounded-lg border-2 border-dashed p-10 transition-colors",
          dragOver
            ? "border-indigo-400 bg-indigo-500/10"
            : "border-slate-700 bg-slate-900/40 hover:border-slate-500",
        ].join(" ")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
      >
        <div className="flex flex-col items-center text-center">
          <UploadCloud size={36} className="text-slate-500" />
          <p className="mt-3 text-sm text-slate-200">
            Drag and drop a PDF here, or
          </p>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="mt-2 inline-flex items-center gap-1.5 rounded bg-indigo-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-400"
          >
            <FileText size={14} />
            Choose file
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
          />
          <p className="mt-3 text-xxs text-slate-500">
            Max upload size set by backend.
          </p>
          <button
            type="button"
            onClick={handleDemo}
            className="mt-4 text-xs text-indigo-300 underline-offset-2 hover:underline"
          >
            or load a synthetic demo card
          </button>
        </div>

        {busy && (
          <div className="absolute inset-0 flex items-center justify-center rounded-lg bg-slate-950/70 backdrop-blur-sm">
            <div className="flex items-center gap-2 text-sm text-slate-200">
              <Loader2 size={16} className="animate-spin" />
              Parsing {filename ?? "PDF"} …
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-6 flex items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-200">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" />
          <div>
            <div className="font-medium">Ingestion failed</div>
            <div className="text-xs text-rose-300/80">{error}</div>
          </div>
        </div>
      )}

      {result?.success && result.card && (
        <div className="mt-6 rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-4">
          <div className="flex items-start gap-2">
            <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-emerald-400" />
            <div className="flex-1">
              <div className="text-sm font-medium text-slate-100">
                Parsed {result.card.source_filename}
              </div>
              <dl className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-xxs sm:grid-cols-4">
                <div>
                  <dt className="text-slate-500">Track</dt>
                  <dd className="mono text-slate-200">{result.card.track_code}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">Date</dt>
                  <dd className="text-slate-200">{formatDate(result.card.card_date)}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">Races</dt>
                  <dd className="mono text-slate-200">{result.card.races.length}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">Source</dt>
                  <dd className="mono text-slate-200">{result.card.source_format}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">Processing</dt>
                  <dd className="mono text-slate-200">
                    {result.processing_ms != null
                      ? `${result.processing_ms.toFixed(0)} ms`
                      : "—"}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Pages</dt>
                  <dd className="mono text-slate-200">{result.card.total_pages}</dd>
                </div>
              </dl>
            </div>
          </div>
          {cardId && (
            <div className="mt-4 flex items-center justify-end">
              <Link
                href={`/card/${cardId}`}
                className="inline-flex items-center gap-1 rounded bg-indigo-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-400"
                onClick={() => router.prefetch(`/card/${cardId}`)}
              >
                Analyze card →
              </Link>
            </div>
          )}
        </div>
      )}
    </main>
  );
}
