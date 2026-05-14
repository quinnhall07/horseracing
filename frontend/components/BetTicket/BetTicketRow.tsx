import { AlertTriangle } from "lucide-react";

import type { BetRecommendation } from "@/lib/types";
import {
  formatBetType,
  formatEdge,
  formatFraction,
  formatMoney,
  formatOdds,
  formatProb,
  formatSelection,
} from "@/lib/format";

interface Props {
  rec: BetRecommendation;
  /** The 3% per-bet cap from ADR-002. */
  capFraction?: number;
}

export default function BetTicketRow({ rec, capFraction = 0.03 }: Props) {
  const c = rec.candidate;
  const atCap = rec.stake_fraction >= capFraction - 1e-4;
  const edgeClass = c.edge >= 0 ? "text-edge-pos" : "text-edge-neg";

  return (
    <tr className="border-b border-slate-900 hover:bg-slate-900/40">
      <td className="mono px-2 py-2 text-slate-300">{c.race_id}</td>
      <td className="px-2 py-2">
        <span className="rounded border border-indigo-500/40 bg-indigo-500/10 px-1.5 py-0.5 text-xxs font-semibold uppercase tracking-wide text-indigo-300">
          {formatBetType(c.bet_type)}
        </span>
      </td>
      <td className="mono px-2 py-2 text-slate-100">
        {formatSelection(c.selection)}
      </td>
      <td className="mono px-2 py-2 text-right text-slate-300">
        {formatOdds(c.decimal_odds)}
      </td>
      <td className="mono px-2 py-2 text-right text-slate-400">
        {formatProb(c.model_prob)}
      </td>
      <td className={`mono px-2 py-2 text-right font-semibold ${edgeClass}`}>
        {formatEdge(c.edge)}
      </td>
      <td className="mono px-2 py-2 text-right text-slate-200">
        {formatMoney(c.expected_value * rec.stake)}
      </td>
      <td className="mono px-2 py-2 text-right text-slate-100">
        {formatMoney(rec.stake)}
      </td>
      <td className="px-2 py-2 text-right">
        <div className="inline-flex items-center gap-1">
          <span className="mono text-slate-300">
            {formatFraction(rec.stake_fraction)}
          </span>
          {atCap && (
            <span
              title="At per-bet Kelly cap (ADR-002 3%)"
              className="text-amber-400"
            >
              <AlertTriangle size={11} />
            </span>
          )}
        </div>
      </td>
    </tr>
  );
}
