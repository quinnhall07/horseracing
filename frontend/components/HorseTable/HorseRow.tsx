"use client";

import { ChevronDown, ChevronRight, Droplet, Eye } from "lucide-react";
import { useState } from "react";

import type { HorseEntry } from "@/lib/types";
import {
  formatDistance,
  formatEdge,
  formatOdds,
  formatProb,
  formatTime,
} from "@/lib/format";
import ProbabilityBar from "../ProbabilityBar/ProbabilityBar";

interface Props {
  horse: HorseEntry;
  /** Highest model_prob in the field — used to scale probability bars. */
  scale: number;
}

export default function HorseRow({ horse, scale }: Props) {
  const [open, setOpen] = useState(false);
  const edge = horse.edge ?? 0;
  const edgeClass =
    edge > 0.005
      ? "text-edge-pos"
      : edge < -0.005
        ? "text-edge-neg"
        : "text-slate-400";

  return (
    <>
      <tr
        className="border-b border-slate-900 hover:bg-slate-900/40 cursor-pointer"
        onClick={() => setOpen((v) => !v)}
      >
        <td className="px-2 py-2 text-slate-500">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </td>
        <td className="mono px-2 py-2 text-slate-300">
          {horse.program_number ?? horse.post_position}
        </td>
        <td className="px-2 py-2 text-slate-100">
          <div className="flex items-center gap-1.5">
            <span className="font-medium">{horse.horse_name}</span>
            {horse.medication_flags.includes("L") && (
              <span title="Lasix" className="text-sky-400">
                <Droplet size={11} />
              </span>
            )}
            {horse.equipment_changes.includes("blinkers_on") && (
              <span title="Blinkers on" className="text-amber-400">
                <Eye size={11} />
              </span>
            )}
          </div>
          <div className="text-xxs text-slate-500">
            {horse.jockey ?? "—"} / {horse.trainer ?? "—"}
          </div>
        </td>
        <td className="mono px-2 py-2 text-right text-slate-300">
          {formatOdds(horse.morning_line_odds)}
        </td>
        <td className="px-2 py-2">
          <ProbabilityBar
            modelProb={horse.model_prob}
            marketProb={horse.market_prob}
            scale={scale}
          />
        </td>
        <td className="mono px-2 py-2 text-right text-slate-400">
          {formatProb(horse.market_prob)}
        </td>
        <td className={`mono px-2 py-2 text-right font-semibold ${edgeClass}`}>
          {formatEdge(horse.edge)}
        </td>
        <td className="mono px-2 py-2 text-right text-slate-300">
          {horse.ewm_speed_figure?.toFixed(1) ?? "—"}
        </td>
        <td className="px-2 py-2 text-right text-xxs uppercase text-slate-500">
          {horse.pace_style.replace(/_/g, " ")}
        </td>
      </tr>

      {open && (
        <tr className="border-b border-slate-900 bg-slate-950/60">
          <td colSpan={9} className="px-4 py-3">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <h4 className="mb-1.5 text-xxs uppercase tracking-wide text-slate-500">
                  Top past performances
                </h4>
                {horse.pp_lines.length === 0 ? (
                  <p className="text-xs text-slate-500">First-time starter</p>
                ) : (
                  <table className="w-full text-xxs">
                    <thead className="text-slate-500">
                      <tr>
                        <th className="px-1 text-left">Date</th>
                        <th className="px-1 text-left">Track</th>
                        <th className="px-1 text-left">Dist</th>
                        <th className="px-1 text-right">Fin</th>
                        <th className="px-1 text-right">Speed</th>
                        <th className="px-1 text-right">Time</th>
                      </tr>
                    </thead>
                    <tbody className="mono">
                      {horse.pp_lines.slice(0, 3).map((pp, i) => (
                        <tr key={i} className="text-slate-300">
                          <td className="px-1">{pp.race_date}</td>
                          <td className="px-1">{pp.track_code}</td>
                          <td className="px-1">{formatDistance(pp.distance_furlongs)}</td>
                          <td className="px-1 text-right">
                            {pp.finish_position ?? "—"}
                          </td>
                          <td className="px-1 text-right">
                            {pp.speed_figure ?? "—"}
                          </td>
                          <td className="px-1 text-right">
                            {formatTime(pp.fraction_finish)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
              <div>
                <h4 className="mb-1.5 text-xxs uppercase tracking-wide text-slate-500">
                  Computed features
                </h4>
                <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xxs">
                  <dt className="text-slate-500">EWM speed</dt>
                  <dd className="mono text-slate-200">
                    {horse.ewm_speed_figure?.toFixed(2) ?? "—"}
                  </dd>
                  <dt className="text-slate-500">Days since last</dt>
                  <dd className="mono text-slate-200">
                    {horse.days_since_last ?? "—"}
                  </dd>
                  <dt className="text-slate-500">Class trajectory</dt>
                  <dd className="mono text-slate-200">
                    {horse.class_trajectory?.toFixed(3) ?? "—"}
                  </dd>
                  <dt className="text-slate-500">Pace style</dt>
                  <dd className="mono text-slate-200">
                    {horse.pace_style.replace(/_/g, " ")}
                  </dd>
                  <dt className="text-slate-500">Owner</dt>
                  <dd className="text-slate-200">{horse.owner ?? "—"}</dd>
                  <dt className="text-slate-500">Weight</dt>
                  <dd className="mono text-slate-200">
                    {horse.weight_lbs ? `${horse.weight_lbs} lbs` : "—"}
                  </dd>
                </dl>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
