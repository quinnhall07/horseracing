import type { ParsedRace } from "@/lib/types";
import { formatDistance, formatMoneyCompact } from "@/lib/format";
import { Clock, Cloud, Gauge, MapPin, Trophy } from "lucide-react";

interface Props {
  race: ParsedRace;
}

function Chip({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5 rounded border border-slate-800 bg-slate-900/60 px-2 py-1">
      <span className="text-slate-500">{icon}</span>
      <span className="text-xxs uppercase tracking-wide text-slate-500">{label}</span>
      <span className="mono text-xs font-medium text-slate-100">{value}</span>
    </div>
  );
}

export default function RaceHeader({ race }: Props) {
  const h = race.header;
  const raceTitle = h.race_name
    ? `${h.race_name}${h.grade ? ` (G${h.grade})` : ""}`
    : `Race ${h.race_number} — ${h.race_type.replace(/_/g, " ")}`;

  return (
    <header className="rounded-lg border border-slate-800 bg-slate-900/40 p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <h2 className="truncate text-lg font-semibold text-slate-50">{raceTitle}</h2>
          <p className="mt-0.5 text-xs text-slate-400">
            {h.distance_raw} · {h.surface} · {h.condition}
            {h.age_sex_restrictions ? ` · ${h.age_sex_restrictions}` : ""}
          </p>
        </div>
        <div className="shrink-0 rounded border border-indigo-500/40 bg-indigo-500/10 px-2.5 py-1 text-xs font-semibold text-indigo-300">
          Race {h.race_number}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <Chip icon={<Clock size={12} />} label="post" value={h.post_time ?? "—"} />
        <Chip icon={<Gauge size={12} />} label="dist" value={formatDistance(h.distance_furlongs)} />
        <Chip icon={<Trophy size={12} />} label="purse" value={formatMoneyCompact(h.purse_usd)} />
        {h.claiming_price != null && (
          <Chip icon={<Trophy size={12} />} label="clm" value={formatMoneyCompact(h.claiming_price)} />
        )}
        <Chip icon={<MapPin size={12} />} label="track" value={`${h.track_code}`} />
        {h.weather && <Chip icon={<Cloud size={12} />} label="wx" value={h.weather} />}
      </div>

      {race.parse_warnings.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-xxs text-amber-400">
          {race.parse_warnings.map((w, i) => (
            <li key={i}>· {w}</li>
          ))}
        </ul>
      )}
    </header>
  );
}
