"use client";

/**
 * EVGauge
 * ───────
 * Per-race horizontal bar chart of edge (in %) for each horse in the field.
 * Built on Recharts so it composes cleanly with the rest of the stack.
 * Positive bars emerald, negative bars rose.
 */

import type { HorseEntry } from "@/lib/types";
import { formatEdge } from "@/lib/format";
import {
  Bar,
  BarChart,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface Props {
  horses: HorseEntry[];
  height?: number;
}

interface Datum {
  name: string;
  edgePct: number;
  modelProb: number | null;
  marketProb: number | null;
}

export default function EVGauge({ horses, height = 200 }: Props) {
  const data: Datum[] = horses.map((h) => ({
    name: `${h.program_number ?? h.post_position} ${h.horse_name}`,
    edgePct: (h.edge ?? 0) * 100,
    modelProb: h.model_prob ?? null,
    marketProb: h.market_prob ?? null,
  }));

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">
          Edge by horse
        </h3>
        <span className="text-xxs text-slate-500">model − market (%)</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 24, bottom: 4, left: 0 }}
        >
          <XAxis
            type="number"
            tick={{ fill: "#94a3b8", fontSize: 10 }}
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            stroke="#1e293b"
          />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fill: "#cbd5e1", fontSize: 10 }}
            stroke="#1e293b"
          />
          <ReferenceLine x={0} stroke="#475569" />
          <Tooltip
            cursor={{ fill: "rgba(148,163,184,0.06)" }}
            contentStyle={{
              background: "#0f172a",
              border: "1px solid #1e293b",
              borderRadius: 6,
              fontSize: 11,
              color: "#e2e8f0",
            }}
            formatter={(value: number) => formatEdge(value / 100)}
          />
          <Bar dataKey="edgePct" radius={[2, 2, 2, 2]}>
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={d.edgePct >= 0 ? "#10b981" : "#f43f5e"}
                opacity={0.85}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
