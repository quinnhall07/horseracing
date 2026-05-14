"use client";

import { formatFraction } from "@/lib/format";

interface Props {
  riskLevels: number[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

/**
 * Discrete stepped slider — six labelled stops mapped to risk levels. Click a
 * stop or drag the handle to select. Kept stateless so the parent owns the
 * selection (synced with ParetoCurve).
 */
export default function RiskSlider({
  riskLevels,
  selectedIndex,
  onSelect,
}: Props) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Risk budget</h3>
        <span className="text-xxs text-slate-500">
          Max CVaR<sub>95</sub> drawdown as fraction of bankroll
        </span>
      </div>
      <div className="flex items-center gap-2">
        {riskLevels.map((level, i) => {
          const active = i === selectedIndex;
          return (
            <button
              key={level}
              type="button"
              onClick={() => onSelect(i)}
              className={[
                "flex-1 rounded border px-2 py-2 text-xxs font-semibold transition",
                active
                  ? "border-indigo-400 bg-indigo-500/20 text-indigo-100"
                  : "border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-600 hover:text-slate-200",
              ].join(" ")}
              aria-pressed={active}
            >
              <div className="mono">{formatFraction(level)}</div>
              <div className="mt-0.5 text-[10px] uppercase text-slate-500">
                {labelForIndex(i, riskLevels.length)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function labelForIndex(i: number, n: number): string {
  // Spread descriptive labels across the discrete stops.
  if (n <= 1) return "default";
  const frac = i / (n - 1);
  if (frac < 0.2) return "conservative";
  if (frac < 0.4) return "moderate";
  if (frac < 0.6) return "balanced";
  if (frac < 0.8) return "aggressive";
  return "max risk";
}
