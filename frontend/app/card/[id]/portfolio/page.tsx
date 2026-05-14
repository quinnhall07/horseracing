"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft, CheckCheck, Loader2 } from "lucide-react";

import BetTicket from "@/components/BetTicket/BetTicket";
import { getPortfolio } from "@/lib/api";
import {
  formatFraction,
  formatMoney,
  formatMoneyCompact,
} from "@/lib/format";
import type { Portfolio } from "@/lib/types";

interface PageProps {
  params: { id: string };
}

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
      ? "text-edge-pos"
      : tone === "neg"
        ? "text-edge-neg"
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

export default function PortfolioPage({ params }: PageProps) {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    getPortfolio(params.id)
      .then((p) => {
        if (!cancelled) setPortfolio(p);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  if (error) {
    return (
      <main className="mx-auto max-w-5xl px-6 py-10">
        <p className="text-sm text-rose-300">
          Failed to load portfolio: {error}
        </p>
        <Link
          href={`/card/${params.id}`}
          className="mt-3 inline-block text-xs text-indigo-300 hover:underline"
        >
          ← Back to card
        </Link>
      </main>
    );
  }

  if (!portfolio) {
    return (
      <main className="flex min-h-[60vh] items-center justify-center text-sm text-slate-400">
        <Loader2 size={16} className="mr-2 animate-spin" />
        Building portfolio …
      </main>
    );
  }

  const expReturnPct = portfolio.bankroll > 0
    ? portfolio.expected_return / portfolio.bankroll
    : 0;

  return (
    <main className="mx-auto max-w-6xl px-6 py-6">
      <header className="mb-4">
        <div className="flex items-center gap-2 text-xxs text-slate-500">
          <Link
            href={`/card/${params.id}`}
            className="inline-flex items-center gap-1 hover:text-slate-300"
          >
            <ArrowLeft size={11} /> Card
          </Link>
          <span>/</span>
          <span>Bet ticket</span>
        </div>
        <h1 className="mt-1 text-xl font-semibold text-slate-50">
          Bet execution ticket
        </h1>
        <p className="text-xxs text-slate-500">
          card_id: <span className="mono">{portfolio.card_id}</span> · CVaR-constrained 1/4-Kelly allocation
        </p>
      </header>

      <section className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
        <HeaderStat label="Bankroll" value={formatMoneyCompact(portfolio.bankroll)} />
        <HeaderStat
          label="Expected return"
          value={`${formatMoney(portfolio.expected_return)} (${formatFraction(expReturnPct)})`}
          tone={portfolio.expected_return >= 0 ? "pos" : "neg"}
        />
        <HeaderStat
          label="Total stake"
          value={formatFraction(portfolio.total_stake_fraction)}
        />
        <HeaderStat
          label="VaR 95"
          value={formatMoney(portfolio.var_95)}
          tone="warn"
        />
        <HeaderStat
          label="CVaR 95"
          value={formatMoney(portfolio.cvar_95)}
          tone="neg"
        />
        <HeaderStat label="Bets" value={String(portfolio.recommendations.length)} />
      </section>

      <BetTicket portfolio={portfolio} />

      <div className="mt-6 flex items-center justify-end gap-3">
        <span className="text-xxs text-slate-500">
          Confirmations are no-ops in this scaffold — wire up to your TAB / ADW
          provider before production use.
        </span>
        <button
          type="button"
          disabled={portfolio.recommendations.length === 0 || confirmed}
          onClick={() => setConfirmed(true)}
          className={[
            "inline-flex items-center gap-1 rounded px-3 py-1.5 text-xs font-semibold",
            confirmed
              ? "bg-emerald-600 text-white"
              : portfolio.recommendations.length === 0
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
    </main>
  );
}
