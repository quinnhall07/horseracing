/**
 * ProbabilityBar
 * ──────────────
 * Stacked SVG bar showing model_prob vs market_prob for a single horse.
 * Visually communicates edge sign and magnitude without needing a legend.
 *
 *   - emerald slice: portion of model_prob above market_prob (the edge)
 *   - indigo slice : market_prob (what the crowd thinks)
 *   - rose slice   : portion of market_prob above model_prob (negative edge)
 *
 * Pure SVG so it renders inside table cells without layout cost.
 */
import { formatProb } from "@/lib/format";

interface Props {
  modelProb: number | null | undefined;
  marketProb: number | null | undefined;
  /** Max prob in the field — used to scale all bars consistently. */
  scale?: number;
  width?: number;
  height?: number;
}

export default function ProbabilityBar({
  modelProb,
  marketProb,
  scale = 1,
  width = 120,
  height = 14,
}: Props) {
  const m = Math.max(0, modelProb ?? 0);
  const k = Math.max(0, marketProb ?? 0);
  const s = Math.max(0.01, scale);

  const px = (p: number) => Math.max(0, Math.min(width, (p / s) * width));

  const marketW = px(Math.min(k, m));
  const edgePos = m > k ? px(m) - px(k) : 0;
  const edgeNeg = k > m ? px(k) - px(m) : 0;

  return (
    <div className="flex items-center gap-2">
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Model ${formatProb(modelProb)} vs market ${formatProb(marketProb)}`}
        className="block"
      >
        <rect x={0} y={0} width={width} height={height} rx={2} fill="#0f172a" />
        {/* market portion (shared base) */}
        <rect x={0} y={0} width={marketW} height={height} fill="#4f46e5" opacity={0.85} />
        {/* positive edge (emerald, on top of market) */}
        {edgePos > 0 && (
          <rect x={marketW} y={0} width={edgePos} height={height} fill="#10b981" />
        )}
        {/* negative edge (rose, drawn from model edge onward) */}
        {edgeNeg > 0 && (
          <rect x={px(m)} y={0} width={edgeNeg} height={height} fill="#f43f5e" opacity={0.65} />
        )}
      </svg>
      <span className="mono text-xxs text-slate-400">{formatProb(modelProb)}</span>
    </div>
  );
}
