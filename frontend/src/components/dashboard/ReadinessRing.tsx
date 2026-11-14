"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { Readiness } from "@/lib/dashboard";
import { cn } from "@/lib/utils";
import { useChartColors } from "@/lib/chartTheme";
import CountUp from "./CountUp";

interface ReadinessRingProps {
  readiness: Readiness | null;
}

const R = 52;
const CIRC = 2 * Math.PI * R;
const SIZE = 140;
const DRAW_MS = 900;

/**
 * Mirrors backend/core_memory/timeline.py::_readiness_band. If those cutoffs
 * move, these ticks must move with them or the ring will mark the wrong spot.
 */
const BAND_THRESHOLDS = [40, 70, 85] as const;

/** Emerald once the score is interview-viable, quiet grey while building. */
const BAND_TONE: Record<Readiness["band"], string> = {
  Building: "bg-neutral-800/80 text-neutral-400",
  Improving: "bg-neutral-800/80 text-neutral-300",
  "Interview Ready": "bg-emerald-500/10 text-emerald-400",
  Sharp: "bg-emerald-500/15 text-emerald-300",
};

export default function ReadinessRing({ readiness }: ReadinessRingProps) {
  const reduceMotion = useReducedMotion();
  const c = useChartColors();

  const score = readiness?.score ?? 0;
  const offset = CIRC * (1 - score / 100);

  const delta = readiness?.delta_vs_prev ?? 0;
  const deltaLabel = delta === 0 ? null : `${delta > 0 ? "+" : ""}${delta} from your latest session`;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: SIZE, height: SIZE }}>
        <svg
          width={SIZE}
          height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          role="img"
          aria-label={
            readiness
              ? `Readiness score ${score} out of 100, band ${readiness.band}. Interview ready begins at 70.`
              : "Readiness score not available yet"
          }
        >
          <defs>
            <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor={c.series} />
              <stop offset="100%" stopColor={c.seriesBright} />

            </linearGradient>
          </defs>

          <g transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}>
            <circle
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={R}
              fill="none"
              stroke={c.grid}
              strokeWidth={10}
            />

            {/* Band boundaries, so the score has somewhere to be measured
                against instead of floating on a blank track. */}
            {BAND_THRESHOLDS.map((t) => (
              <line
                key={t}
                x1={SIZE / 2 + R - 6}
                y1={SIZE / 2}
                x2={SIZE / 2 + R + 6}
                y2={SIZE / 2}
                stroke={t === 70 ? c.threshold : c.cursor}

                strokeOpacity={t === 70 ? 0.9 : 0.6}
                strokeWidth={t === 70 ? 2 : 1.5}
                transform={`rotate(${t * 3.6} ${SIZE / 2} ${SIZE / 2})`}
              />
            ))}

            <motion.circle
              cx={SIZE / 2}
              cy={SIZE / 2}
              r={R}
              fill="none"
              stroke="url(#ringGrad)"
              strokeWidth={10}
              strokeLinecap="round"
              strokeDasharray={CIRC}
              initial={{ strokeDashoffset: reduceMotion ? offset : CIRC }}
              animate={{ strokeDashoffset: offset }}
              transition={{ duration: reduceMotion ? 0 : DRAW_MS / 1000, ease: "easeOut" }}
            />
          </g>
        </svg>

        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
          {readiness ? (
            <span className="text-4xl font-semibold tabular-nums text-neutral-50">
              {/* Paced to the ring sweep so the number lands as the arc stops. */}
              <CountUp value={score} durationMs={DRAW_MS} />
            </span>
          ) : (
            <span className="max-w-[7rem] text-xs leading-snug text-neutral-500">
              Run your first session
            </span>
          )}
        </div>
      </div>

      {readiness && (
        <span
          className={cn(
            "mt-3 rounded-full px-2.5 py-1 text-xs font-medium",
            BAND_TONE[readiness.band],
          )}
        >
          {readiness.band}
        </span>
      )}

      {deltaLabel && (
        <p
          className={cn(
            "mt-2 text-xs tabular-nums",
            delta > 0 ? "text-emerald-400/80" : "text-neutral-500",
          )}
        >
          {deltaLabel}
        </p>
      )}
    </div>
  );
}
