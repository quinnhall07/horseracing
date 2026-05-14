"use client";

/**
 * components/ProvenanceBanner/ProvenanceBanner.tsx
 * ────────────────────────────────────────────────
 * Visual surface for `ModelProvenance` (see lib/types.ts). Renders:
 *
 *   • Amber/critical banner when models are synthetic — predictions
 *     should not be relied on.
 *   • Slim info banner when running with stub sub-models but the trained
 *     sub-models are real (e.g. Pace stays a stub per ADR-026 while the
 *     rest of the ensemble is fit on real data).
 *   • Nothing when fully real and no stubs.
 *
 * The component is a no-op when `provenance` is null/undefined — convenient
 * when an upstream response (e.g. legacy /upload payload) doesn't carry
 * the field yet.
 */

import { AlertTriangle, Info } from "lucide-react";

import type { ModelProvenance } from "@/lib/types";

interface ProvenanceBannerProps {
  provenance?: ModelProvenance | null;
}

export default function ProvenanceBanner({ provenance }: ProvenanceBannerProps) {
  if (!provenance) return null;

  if (provenance.is_synthetic) {
    return (
      <div
        role="alert"
        className="mb-3 flex items-start gap-3 rounded border border-amber-500/40 bg-amber-900/40 px-4 py-3 text-amber-100"
      >
        <AlertTriangle size={18} className="mt-0.5 shrink-0 text-amber-300" />
        <div className="min-w-0 text-sm">
          <div className="font-semibold">Synthetic-trained predictions</div>
          <p className="mt-1 text-xs leading-relaxed text-amber-100/85">
            These predictions are based on{" "}
            <span className="font-semibold">synthetic training data</span> and
            are not meaningful. Re-bootstrap against real data (see README) before
            relying on any output.
          </p>
          {provenance.warning ? (
            <p className="mono mt-2 break-words text-xxs text-amber-200/70">
              {provenance.warning}
            </p>
          ) : null}
        </div>
      </div>
    );
  }

  const stubs = provenance.stub_sub_models ?? [];
  if (stubs.length > 0) {
    const trained = provenance.sub_models?.length ?? 0;
    const total = trained + stubs.length;
    return (
      <div
        role="status"
        className="mb-3 flex items-start gap-3 rounded border border-sky-500/30 bg-sky-900/30 px-4 py-2 text-sky-100"
      >
        <Info size={16} className="mt-0.5 shrink-0 text-sky-300" />
        <div className="min-w-0 text-xs">
          <div className="font-semibold">
            Ensemble running with {trained} of {total} sub-models
          </div>
          <p className="mt-0.5 text-xxs leading-relaxed text-sky-100/80">
            Stub sub-models:{" "}
            <span className="mono">{stubs.join(", ")}</span> — see ADR-026.
          </p>
        </div>
      </div>
    );
  }

  return null;
}
