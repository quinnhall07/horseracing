"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, CheckCheck, ChevronDown, ChevronRight, Loader2 } from "lucide-react";

import BetTicket from "@/components/BetTicket/BetTicket";
import HorseTable from "@/components/HorseTable/HorseTable";
import ParetoCurve from "@/components/ParetoCurve/ParetoCurve";
import RaceHeader from "@/components/RaceCard/RaceHeader";
import RiskSlider from "@/components/ParetoCurve/RiskSlider";
import { getCard, getParetoFrontier } from "@/lib/api";
import {
  formatDate,
  formatFraction,
  formatMoney,
  formatMoneyCompact,
} from "@/lib/format";
import type { ParetoFrontier, RaceCard } from "@/lib/types";

interface PageProps {
  params: { id: string };
}

const DEFAULT_RISK_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
const DEFAULT_SELECTED = 3; // middle-ish stop (0.20)

function HeaderStat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "pos" | "neg" | "warn";
}) {
  const toneClass =
    tone === "pos"
      ? "text-emerald-300"
      : tone === "neg"
        ? "text-rose-300"
        : tone === "warn"
          ? "text-amber-300"
          : "text-slate-100";
  return (
    <div className="rounded border border-slate-800 bg-slate-900/40 px-3 py-2">
      <div className="text-xxs uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`mono mt-0.5 text-sm font-semibold ${toneClass}`}>{value}</div>
    </div>
  );
}

export default function CardPage({ params }: PageProps) {
  const [card, setCard] = useState<RaceCard | null>(null);
  const [frontier, setFrontier] = useState<ParetoFrontier | null>(null);
  const [selectedRiskIndex, setSelectedRiskIndex] = useState(DEFAULT_SELECTED);
  const [bankroll, setBankroll] = useState(10000);
  const [bankrollInput, setBankrollInput] = useState("10000");
  const [error, setError] = useState<string | null>(null);
  const [reFetching, setReFetching] = useState(false);
  const [expandedRaceNumbers, setExpandedRaceNumbers] = useState<Set<number>>(
    new Set(),
  );
  const [confirmed, setConfirmed] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Fetch the card once.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    getCard(params.id)
      .then((c) => {
        if (!cancelled) setCard(c);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  // Fetch the pareto frontier when bankroll changes (debounced).
  const fetchFrontier = useCallback(
    (br: number) => {
      let cancelled = false;
      setReFetching(true);
      getParetoFrontier(params.id, { bankroll: br, riskLevels: DEFAULT_RISK_LEVELS })
        .then((f) => {
          if (!cancelled) setFrontier(f);
        })
        .catch((e) => {
          if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        })
        .finally(() => {
          if (!cancelled) setReFetching(false);
        });
      return () => {
        cancelled = true;
      };
    },
    [params.id],
  );

  useEffect(() => {
    const cleanup = fetchFrontier(bankroll);
    return cleanup;
  }, [bankroll, fetchFrontier]);

  const onBankrollInput = (raw: string) => {
    setBankrollInput(raw);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      const parsed = parseFloat(raw);
      if (Number.isFinite(parsed) && parsed > 0) {
        setBankroll(parsed);
        setConfirmed(false);
      }
    }, 400);
  };

  const selectedPortfolio = useMemo(() => {
    if (!frontier || frontier.frontier.length === 0) return null;
    const i = Math.min(selectedRiskIndex, frontier.frontier.length - 1);
    return frontier.frontier[i]?.portfolio ?? null;
  }, [frontier, selectedRiskIndex]);

  const recommendationsByRace = useMemo(() => {
    const map = new Map<string, number>();
    if (!selectedPortfolio) return map;
    for (const rec of selectedPortfolio.recommendations) {
      map.set(rec.candidate.race_id, (map.get(rec.candidate.race_id) ?? 0) + 1);
    }
    return map;
  }, [selectedPortfolio]);

  const toggleRace = (raceNum: number) => {
    setExpandedRaceNumbers((prev) => {
      const next = new Set(prev);
      if (next.has(raceNum)) next.delete(raceNum);
      else next.add(raceNum);
      return next;
    });
  };

  if (error && !card) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-10">
        <p className="text-sm text-rose-300">Failed to load card: {error}</p>
        <Link href="/" className="mt-3 inline-block text-xs text-indigo-300 hover:underline">
          ← Back to upload
        </Link>
      </main>
    );
  }

  if (!card) {
    return (
      <main className="flex min-h-[60vh] items-center justify-center text-sm text-slate-400">
        <Loader2 size={16} className="mr-2 animate-spin" />
        Loading card …
      </main>
    );
  }

  const expReturnPct =
    selectedPortfolio && selectedPortfolio.bankroll > 0
      ? selectedPortfolio.expected_return / selectedPortfolio.bankroll
      : 0;

  return (
    <main className="mx-auto max-w-6xl px-6 py-6">
      <header className="sticky top-0 z-10 -mx-6 mb-4 border-b border-slate-800 bg-slate-950/95 px-6 py-3 backdrop-blur">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xxs text-slate-500">
              <Link href="/" className="inline-flex items-center gap-1 hover:text-slate-300">
                <ArrowLeft size={11} /> Upload another card
              </Link>
              <span>/</span>
              <span className="mono">{card.card_id ?? params.id}</span>
            </div>
            <h1 className="mt-1 truncate text-xl font-semibold text-slate-50">
              {card.track_code} · {formatDate(card.card_date)}
            </h1>
            <p className="text-xxs text-slate-500">
              {card.source_filename} · {card.source_format} · {card.races.length} races
            </p>
          </div>
          <div className="flex items-end gap-3">
            <label className="block">
              <div className="text-xxs uppercase tracking-wide text-slate-500">
                Bankroll (USD)
              </div>
              <input
                type="number"
                min={1}
                value={bankrollInput}
                onChange={(e) => onBankrollInput(e.target.value)}
                className="mono mt-0.5 w-32 rounded border border-slate-800 bg-slate-900 px-2 py-1 text-sm text-slate-100 focus:border-indigo-400 focus:outline-none"
              />
            </label>
            {reFetching ? (
              <span className="flex items-center gap-1 text-xxs text-slate-500">
                <Loader2 size={11} className="animate-spin" /> recomputing …
              </span>
            ) : null}
          </div>
        </div>
      </header>

      {!frontier ? (
        <div className="flex min-h-[40vh] items-center justify-center text-sm text-slate-400">
          <Loader2 size={16} className="mr-2 animate-spin" />
          Solving optimiser …
        </div>
      ) : (
        <>
          {/* Risk slider + pareto curve */}
          <section className="mb-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <RiskSlider
              riskLevels={DEFAULT_RISK_LEVELS}
              selectedIndex={selectedRiskIndex}
              onSelect={(i) => {
                setSelectedRiskIndex(i);
                setConfirmed(false);
              }}
            />
            <ParetoCurve
              frontier={frontier.frontier}
              selectedIndex={selectedRiskIndex}
              onSelect={(i) => {
                setSelectedRiskIndex(i);
                setConfirmed(false);
              }}
            />
          </section>

          {/* Portfolio summary strip */}
          {selectedPortfolio ? (
            <section className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              <HeaderStat
                label="Bankroll"
                value={formatMoneyCompact(selectedPortfolio.bankroll)}
              />
              <HeaderStat
                label="Expected return"
                value={`${formatMoney(selectedPortfolio.expected_return)} (${formatFraction(expReturnPct)})`}
                tone={selectedPortfolio.expected_return >= 0 ? "pos" : "neg"}
              />
              <HeaderStat
                label="Total stake"
                value={formatFraction(selectedPortfolio.total_stake_fraction)}
              />
              <HeaderStat
                label="VaR 95"
                value={formatMoney(selectedPortfolio.var_95)}
                tone="warn"
              />
              <HeaderStat
                label="CVaR 95"
                value={formatMoney(selectedPortfolio.cvar_95)}
                tone="neg"
              />
              <HeaderStat
                label="Bets"
                value={String(selectedPortfolio.recommendations.length)}
              />
            </section>
          ) : null}

          {/* Bet ticket */}
          {selectedPortfolio ? (
            <section className="mb-4">
              <h2 className="mb-2 text-sm font-semibold text-slate-200">
                Recommended bets at this risk level
              </h2>
              <BetTicket portfolio={selectedPortfolio} />
              <div className="mt-3 flex items-center justify-end gap-3">
                <span className="text-xxs text-slate-500">
                  Paper-trading only — confirmations are a no-op.
                </span>
                <button
                  type="button"
                  disabled={selectedPortfolio.recommendations.length === 0 || confirmed}
                  onClick={() => setConfirmed(true)}
                  className={[
                    "inline-flex items-center gap-1 rounded px-3 py-1.5 text-xs font-semibold",
                    confirmed
                      ? "bg-emerald-600 text-white"
                      : selectedPortfolio.recommendations.length === 0
                        ? "cursor-not-allowed bg-slate-800 text-slate-500"
                        : "bg-indigo-500 text-white hover:bg-indigo-400",
                  ].join(" ")}
                >
                  {confirmed ? (
                    <>
                      <CheckCheck size={12} /> Placements confirmed
                    </>
                  ) : (
                    "Confirm placements"
                  )}
                </button>
              </div>
            </section>
          ) : null}

          {/* Race detail accordions */}
          <section id="races" className="mt-6">
            <h2 className="mb-2 text-sm font-semibold text-slate-200">Race detail</h2>
            <div className="space-y-2">
              {card.races.map((race) => {
                const rn = race.header.race_number;
                const expanded = expandedRaceNumbers.has(rn);
                const raceId = `${formatDate(race.header.race_date)}|${card.track_code}|${rn}`;
                // race_id format used by the backend differs but the count by candidate is
                // best-effort: try both common keys.
                const nRecs =
                  recommendationsByRace.get(raceId) ??
                  recommendationsByRace.get(`r${rn}`) ??
                  recommendationsByRace.get(String(rn)) ??
                  0;
                const sorted = [...race.entries].sort(
                  (a, b) => (b.model_prob ?? 0) - (a.model_prob ?? 0),
                );
                return (
                  <div
                    key={rn}
                    className="rounded-lg border border-slate-800 bg-slate-900/30"
                  >
                    <button
                      type="button"
                      onClick={() => toggleRace(rn)}
                      className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
                    >
                      <div className="flex items-center gap-2 text-sm text-slate-200">
                        {expanded ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                        <span className="font-semibold">Race {rn}</span>
                        <span className="text-xxs text-slate-500">
                          · {race.entries.length} horses
                        </span>
                        {nRecs > 0 ? (
                          <span className="ml-1 rounded-full bg-indigo-500/20 px-2 py-0.5 text-xxs text-indigo-200">
                            {nRecs} bet{nRecs === 1 ? "" : "s"} at this risk level
                          </span>
                        ) : null}
                      </div>
                    </button>
                    {expanded ? (
                      <div className="border-t border-slate-800 p-3">
                        <RaceHeader race={race} />
                        <div className="mt-3">
                          <HorseTable entries={sorted} />
                        </div>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
