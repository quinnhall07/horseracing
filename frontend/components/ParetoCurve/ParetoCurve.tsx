"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatFraction, formatMoney } from "@/lib/format";
import type { ParetoPoint } from "@/lib/types";

interface Props {
  frontier: ParetoPoint[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  height?: number;
}

interface ChartDatum {
  index: number;
  /** abs(CVaR) — display X-axis as a positive "downside" value. */
  risk: number;
  return: number;
  drawdown: number;
  nRecs: number;
  isSelected: boolean;
}

interface CustomDotProps {
  cx?: number;
  cy?: number;
  payload?: ChartDatum;
}

function CustomDot(props: CustomDotProps) {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null || !payload) return null;
  const r = payload.isSelected ? 8 : 5;
  const fill = payload.isSelected ? "#6366f1" : "#94a3b8";
  const stroke = payload.isSelected ? "#fff" : "#1e293b";
  return (
    <circle
      cx={cx}
      cy={cy}
      r={r}
      fill={fill}
      stroke={stroke}
      strokeWidth={payload.isSelected ? 2 : 1}
      style={{ cursor: "pointer", transition: "r 150ms ease-out" }}
    />
  );
}

export default function ParetoCurve({
  frontier,
  selectedIndex,
  onSelect,
  height = 280,
}: Props) {
  const data: ChartDatum[] = useMemo(
    () =>
      frontier.map((pt, i) => ({
        index: i,
        risk: Math.abs(pt.portfolio.cvar_95),
        return: pt.portfolio.expected_return,
        drawdown: pt.max_drawdown_pct,
        nRecs: pt.portfolio.recommendations.length,
        isSelected: i === selectedIndex,
      })),
    [frontier, selectedIndex],
  );

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/30 p-4">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-200">
          Risk / return frontier
        </h3>
        <span className="text-xxs text-slate-500">
          Worst-5% loss (CVaR<sub>95</sub>) vs. expected portfolio return.
          Click any point — or use the slider — to switch the bet ticket.
        </span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={data}
          margin={{ top: 8, right: 24, bottom: 24, left: 8 }}
          onClick={(state) => {
            const activePayload = state?.activePayload?.[0]?.payload;
            if (activePayload && typeof activePayload.index === "number") {
              onSelect(activePayload.index);
            }
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            type="number"
            dataKey="risk"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => formatMoney(v)}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            label={{
              value: "Risk (CVaR₉₅, worst-5% loss)",
              position: "insideBottom",
              offset: -10,
              fill: "#64748b",
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="return"
            tickFormatter={(v) => formatMoney(v)}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            label={{
              value: "Expected return",
              angle: -90,
              position: "insideLeft",
              fill: "#64748b",
              fontSize: 11,
            }}
          />
          <Tooltip
            cursor={{ stroke: "#334155", strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload as ChartDatum | undefined;
              if (!d) return null;
              return (
                <div className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-xs shadow-lg">
                  <div className="mb-1 font-semibold text-slate-100">
                    Risk level {formatFraction(d.drawdown)}
                  </div>
                  <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-slate-300">
                    <span className="text-slate-500">Expected return</span>
                    <span className="mono text-right text-emerald-300">
                      {formatMoney(d.return)}
                    </span>
                    <span className="text-slate-500">CVaR₉₅</span>
                    <span className="mono text-right text-rose-300">
                      {formatMoney(d.risk)}
                    </span>
                    <span className="text-slate-500">Bets</span>
                    <span className="mono text-right">{d.nRecs}</span>
                  </div>
                </div>
              );
            }}
          />
          <Line
            type="monotone"
            dataKey="return"
            stroke="#475569"
            strokeWidth={1.5}
            dot={false}
            activeDot={false}
            isAnimationActive={false}
            legendType="none"
          />
          <Scatter
            dataKey="return"
            shape={<CustomDot />}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
