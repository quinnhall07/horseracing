"use client";

import type { HorseEntry } from "@/lib/types";
import HorseRow from "./HorseRow";

interface Props {
  entries: HorseEntry[];
}

export default function HorseTable({ entries }: Props) {
  // entries are pre-sorted by the page; just compute the scale.
  const scale = Math.max(0.01, ...entries.map((e) => e.model_prob ?? 0));

  if (entries.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-6 text-center text-sm text-slate-400">
        No horses in this race.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-800 bg-slate-900/30">
      <table className="w-full min-w-[820px] table-fixed text-xs">
        <colgroup>
          <col style={{ width: 28 }} />
          <col style={{ width: 36 }} />
          <col />
          <col style={{ width: 110 }} />
          <col style={{ width: 160 }} />
          <col style={{ width: 70 }} />
          <col style={{ width: 70 }} />
          <col style={{ width: 70 }} />
          <col style={{ width: 90 }} />
        </colgroup>
        <thead className="border-b border-slate-800 bg-slate-950 text-xxs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="px-2 py-2 text-left" />
            <th className="px-2 py-2 text-left">#</th>
            <th className="px-2 py-2 text-left">Horse / Connections</th>
            <th className="px-2 py-2 text-right">ML odds</th>
            <th className="px-2 py-2 text-left">Model vs market</th>
            <th className="px-2 py-2 text-right">Market</th>
            <th className="px-2 py-2 text-right">Edge</th>
            <th className="px-2 py-2 text-right">EWM spd</th>
            <th className="px-2 py-2 text-right">Pace</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((h) => (
            <HorseRow key={h.horse_id ?? `${h.post_position}-${h.horse_name}`} horse={h} scale={scale} />
          ))}
        </tbody>
      </table>
    </div>
  );
}
