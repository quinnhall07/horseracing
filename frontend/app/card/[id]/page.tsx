"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowLeft, ArrowRight, Loader2 } from "lucide-react";

import EVGauge from "@/components/EVGauge/EVGauge";
import HorseTable from "@/components/HorseTable/HorseTable";
import RaceHeader from "@/components/RaceCard/RaceHeader";
import RaceTabs from "@/components/RaceCard/RaceTabs";
import { getCard } from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { RaceCard } from "@/lib/types";

interface PageProps {
  params: { id: string };
}

export default function CardPage({ params }: PageProps) {
  const [card, setCard] = useState<RaceCard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number>(1);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    getCard(params.id)
      .then((c) => {
        if (cancelled) return;
        setCard(c);
        if (c.races.length > 0) {
          setSelected(c.races[0]!.header.race_number);
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [params.id]);

  const currentRace = useMemo(
    () => card?.races.find((r) => r.header.race_number === selected) ?? null,
    [card, selected],
  );

  const sortedEntries = useMemo(() => {
    if (!currentRace) return [];
    return [...currentRace.entries].sort(
      (a, b) => (b.model_prob ?? 0) - (a.model_prob ?? 0),
    );
  }, [currentRace]);

  if (error) {
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

  return (
    <main className="mx-auto max-w-6xl px-6 py-6">
      <header className="sticky top-0 z-10 -mx-6 mb-4 border-b border-slate-800 bg-slate-950/95 px-6 py-3 backdrop-blur">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xxs text-slate-500">
              <Link href="/" className="inline-flex items-center gap-1 hover:text-slate-300">
                <ArrowLeft size={11} /> Upload
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
          <Link
            href={`/card/${params.id}/portfolio`}
            className="inline-flex items-center gap-1 rounded bg-indigo-500 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-400"
          >
            View bet ticket <ArrowRight size={12} />
          </Link>
        </div>
        <div className="mt-3">
          <RaceTabs races={card.races} selected={selected} onSelect={setSelected} />
        </div>
      </header>

      {currentRace ? (
        <div className="space-y-4">
          <RaceHeader race={currentRace} />
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <HorseTable entries={sortedEntries} />
            </div>
            <div className="lg:col-span-1">
              <EVGauge horses={sortedEntries} height={Math.max(220, sortedEntries.length * 22)} />
            </div>
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-400">No race selected.</p>
      )}
    </main>
  );
}
