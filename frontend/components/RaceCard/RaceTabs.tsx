"use client";

import type { ParsedRace } from "@/lib/types";

interface Props {
  races: ParsedRace[];
  selected: number;
  onSelect: (raceNumber: number) => void;
}

export default function RaceTabs({ races, selected, onSelect }: Props) {
  return (
    <nav
      aria-label="Races on this card"
      className="flex gap-1 overflow-x-auto border-b border-slate-800 pb-1"
    >
      {races.map((r) => {
        const active = r.header.race_number === selected;
        return (
          <button
            key={r.header.race_number}
            type="button"
            onClick={() => onSelect(r.header.race_number)}
            className={[
              "shrink-0 rounded-t px-3 py-1.5 text-xs font-medium transition-colors",
              active
                ? "bg-indigo-500/15 text-indigo-300 ring-1 ring-inset ring-indigo-500/40"
                : "text-slate-400 hover:bg-slate-900 hover:text-slate-200",
            ].join(" ")}
          >
            <span className="mono">R{r.header.race_number}</span>
            <span className="ml-1.5 text-slate-500">
              {r.entries.length}H
            </span>
          </button>
        );
      })}
    </nav>
  );
}
