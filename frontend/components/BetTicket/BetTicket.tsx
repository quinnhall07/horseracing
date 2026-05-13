"use client";

import type { Portfolio } from "@/lib/types";
import BetTicketRow from "./BetTicketRow";

interface Props {
  portfolio: Portfolio;
}

export default function BetTicket({ portfolio }: Props) {
  if (portfolio.recommendations.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-8 text-center text-sm text-slate-400">
        No +EV bets met the CVaR-constrained threshold for this card.
        <br />
        <span className="text-xs text-slate-500">
          Pass on this card — preserve bankroll for higher-edge opportunities.
        </span>
      </div>
    );
  }

  // Sort by race then by stake fraction (largest first within race).
  const sorted = [...portfolio.recommendations].sort((a, b) => {
    const ra = a.candidate.race_id;
    const rb = b.candidate.race_id;
    if (ra !== rb) return ra.localeCompare(rb, undefined, { numeric: true });
    return b.stake_fraction - a.stake_fraction;
  });

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/30">
      <table className="w-full min-w-[860px] text-xs">
        <thead className="border-b border-slate-800 bg-slate-950 text-xxs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-2 py-2 text-left">Race</th>
            <th className="px-2 py-2 text-left">Type</th>
            <th className="px-2 py-2 text-left">Selection</th>
            <th className="px-2 py-2 text-right">Odds</th>
            <th className="px-2 py-2 text-right">Model p</th>
            <th className="px-2 py-2 text-right">Edge</th>
            <th className="px-2 py-2 text-right">Expected $</th>
            <th className="px-2 py-2 text-right">Stake</th>
            <th className="px-2 py-2 text-right">Frac</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((rec, i) => (
            <BetTicketRow
              key={`${rec.candidate.race_id}-${rec.candidate.bet_type}-${i}`}
              rec={rec}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}
