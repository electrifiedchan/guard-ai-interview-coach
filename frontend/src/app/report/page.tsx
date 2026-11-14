"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { Shield, RotateCcw, Target, Eye, MessageSquare, LayoutDashboard, ArrowRight, FileUp } from "lucide-react";
import DashboardButton from "@/components/DashboardButton";
import DriftTimeline, {
  type TimelineFrame,
  type TimelineMoment,
} from "@/components/report/DriftTimeline";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

interface ReportData {
  report: string;
  coaching?: {
    verdict: string;
    strengths: string[];
    primary_improvement: string;
    next_actions: string[];
    readiness: string;
  };
  /** Average composure from the recorded camera frames, or null if none. */
  average_focus_score: number | null;
  turns_completed: number;
}

export default function ReportPage() {
  const router = useRouter();
  const [data, setData] = useState<ReportData | null>(null);
  // The session's average composure as the history computes it. Read from the
  // same endpoint and the same backend function as the dashboard's session
  // list, so the number here and the one in history cannot differ.
  const [timelineFocus, setTimelineFocus] = useState<number | null>(null);
  const [frames, setFrames] = useState<TimelineFrame[]>([]);
  const [moments, setMoments] = useState<TimelineMoment[]>([]);
  const [timelineState, setTimelineState] =
    useState<"loading" | "ready" | "unavailable">("loading");

  useEffect(() => {
    const raw = sessionStorage.getItem("guard_report");
    if (!raw) {
      router.push("/upload");
      return;
    }
    setData(JSON.parse(raw));
  }, [router]);

  /**
   * Pull the per-frame telemetry for the run that just finished. The verdict
   * already summarises composure as one number; this is what makes the report
   * specific enough to act on — which minute drifted, for how long, and any
   * frame the sentry flagged.
   */
  const loadTimeline = useCallback(async () => {
    const visionSession = sessionStorage.getItem("guard_vision_session");
    if (!visionSession) {
      setTimelineState("unavailable");
      return;
    }

    try {
      const res = await fetch(
        `${API_BASE}/api/v1/session/${encodeURIComponent(visionSession)}/timeline`,
      );
      if (!res.ok) throw new Error(`timeline ${res.status}`);

      const payload = await res.json();
      const nextFrames: TimelineFrame[] = payload.frames ?? [];
      setFrames(nextFrames);
      setMoments(payload.moments ?? []);
      const avg = payload.stats?.avg_composure;
      setTimelineFocus(nextFrames.length > 0 && typeof avg === "number" ? avg : null);
      setTimelineState(nextFrames.length > 0 ? "ready" : "unavailable");
    } catch {
      // The coaching report is the main event; a missing timeline degrades the
      // page rather than breaking it.
      setTimelineState("unavailable");
    }
  }, []);

  useEffect(() => {
    if (!data) return;
    // Frames are written via FastAPI background tasks, so the last few can land
    // just after the verdict returns. One short retry avoids showing a timeline
    // that's missing its own ending.
    loadTimeline();
    const retry = setTimeout(loadTimeline, 1500);
    return () => clearTimeout(retry);
  }, [data, loadTimeline]);

  if (!data) {
    return (
      <main className="min-h-screen bg-[var(--color-canvas)] flex items-center justify-center">
        <div className="flex items-center gap-2 text-[var(--color-slate)]">
          <span className="w-2 h-2 rounded-full bg-[var(--color-signal)] pulse-signal" />
          <span className="text-sm">Loading report…</span>
        </div>
      </main>
    );
  }

  // Rounded the same way the history rounds it (Math.round of the one-decimal
  // average), so 52.5 reads as 53 here and there. Null means the camera
  // recorded nothing for this session; say so rather than show a made-up 0 or 100.
  const rawFocus = timelineFocus ?? data.average_focus_score;
  const focusScore = typeof rawFocus === "number" ? Math.round(rawFocus) : null;
  const focusColor =
    focusScore === null ? "text-[var(--color-slate)]" :
    focusScore >= 80 ? "text-emerald-400" :
    focusScore >= 50 ? "text-amber-400" :
    "text-rose-500";

  const overallGrade =
    focusScore === null ? "NO CAMERA DATA" :
    focusScore >= 80 ? "STRONG" :
    focusScore >= 60 ? "MODERATE" :
    "NEEDS WORK";

  const gradeColor =
    focusScore === null ? "text-[var(--color-slate)] border-[var(--color-hairline)] bg-[var(--color-surface)]" :
    focusScore >= 80 ? "text-emerald-400 border-emerald-400/30 bg-emerald-400/[0.06]" :
    focusScore >= 60 ? "text-amber-400 border-amber-400/30 bg-amber-400/[0.06]" :
    "text-rose-400 border-rose-400/30 bg-rose-400/[0.06]";

  return (
    <main className="min-h-screen bg-[var(--color-canvas)] text-[var(--color-parchment)] px-6 py-10 flex flex-col items-center">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-[720px] flex flex-col gap-8"
      >
        {/* Header */}
        <header className="flex items-center justify-between border-b border-[var(--color-hairline)] pb-6">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[var(--color-signal-soft)] flex items-center justify-center">
              <Shield size={18} className="text-[var(--color-signal)]" />
            </div>
            <div className="flex flex-col">
              <span className="eyebrow text-[var(--color-signal)]">Session Complete</span>
              <h1 className="text-lg font-semibold text-[var(--color-snow)] tracking-tight">
                Performance Report
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className={`px-3 py-1.5 rounded-lg border text-[12px] font-semibold uppercase tracking-wider ${gradeColor}`}>
              {overallGrade}
            </span>
            <DashboardButton />
          </div>
        </header>

        {/* Metrics row */}
        <div className="grid grid-cols-3 gap-4">
          <div className="lift-1 rounded-lg p-5 flex flex-col items-center gap-2">
            <Eye size={18} className="text-[var(--color-slate)]" />
            <span className="eyebrow">Focus Score</span>
            <span className={`text-[32px] font-display font-semibold tabular leading-none ${focusColor}`}>
              {focusScore ?? "—"}
              {focusScore !== null && (
                <span className="text-[var(--color-fog)] text-[18px] font-medium ml-0.5">%</span>
              )}
            </span>
          </div>
          <div className="lift-1 rounded-lg p-5 flex flex-col items-center gap-2">
            <MessageSquare size={18} className="text-[var(--color-slate)]" />
            <span className="eyebrow">Turns</span>
            <span className="text-[32px] font-display font-semibold tabular leading-none text-[var(--color-snow)]">
              {data.turns_completed}
            </span>
          </div>
          <div className="lift-1 rounded-lg p-5 flex flex-col items-center gap-2">
            <Target size={18} className="text-[var(--color-slate)]" />
            <span className="eyebrow">Processing</span>
            <span className="text-[18px] font-semibold text-[var(--color-snow)]">Edge</span>
          </div>
        </div>

        {/* Attention timeline — the "when did I drift" evidence */}
        {timelineState === "loading" && (
          <div className="lift-1 rounded-lg p-6 flex items-center gap-2 text-[var(--color-slate)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)] pulse-signal" />
            <span className="text-[13px]">Reconstructing attention timeline…</span>
          </div>
        )}
        {timelineState === "ready" && (
          <DriftTimeline frames={frames} moments={moments} />
        )}

        {/* Coaching brief */}
        <section className="border-y border-[var(--color-hairline)] py-6">
          <span className="eyebrow mb-5 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)]" />
            AI Coach Feedback
          </span>
          {data.coaching ? (
            <div className="flex flex-col gap-6">
              <div>
                <span className="eyebrow text-[10px]">Verdict · {data.coaching.readiness}</span>
                <p className="mt-2 text-[20px] font-display font-semibold leading-snug text-[var(--color-snow)]">
                  {data.coaching.verdict}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6">
                <div>
                  <h2 className="text-[13px] font-semibold text-[var(--color-snow)]">What worked</h2>
                  <ul className="mt-2.5 space-y-2">
                    {data.coaching.strengths.slice(0, 3).map((strength) => (
                      <li key={strength} className="flex gap-2 text-[13px] leading-relaxed text-[var(--color-parchment)]">
                        <span className="mt-2 h-1 w-1 rounded-full bg-[var(--color-signal)] shrink-0" />
                        <span>{strength}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h2 className="text-[13px] font-semibold text-[var(--color-snow)]">Priority fix</h2>
                  <p className="mt-2.5 text-[13px] leading-relaxed text-[var(--color-parchment)]">
                    {data.coaching.primary_improvement}
                  </p>
                </div>
              </div>

              <div>
                <h2 className="text-[13px] font-semibold text-[var(--color-snow)]">Next attempt</h2>
                <ol className="mt-2.5 grid grid-cols-1 md:grid-cols-3 gap-3">
                  {data.coaching.next_actions.slice(0, 3).map((action, index) => (
                    <li key={action} className="flex gap-3 border-t border-[var(--color-hairline)] pt-3 text-[13px] leading-relaxed text-[var(--color-parchment)]">
                      <span className="font-mono text-[11px] text-[var(--color-signal)]">0{index + 1}</span>
                      <span>{action}</span>
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          ) : (
            <p className="text-[13px] leading-relaxed text-[var(--color-parchment)] whitespace-pre-wrap">
              {data.report}
            </p>
          )}
        </section>

        {/* Progress hand-off — one session is a data point, the dashboard is
            the trend. This was the missing exit from the flow: the report was
            a dead end, so the dashboard we built was never reachable after a run. */}
        <Link
          href="/dashboard"
          className="lift-1 rounded-lg p-5 flex items-center gap-4 group hover:border-[var(--color-signal)]/40 transition-colors"
        >
          <div className="w-10 h-10 rounded-lg bg-[var(--color-signal-soft)] flex items-center justify-center shrink-0">
            <LayoutDashboard size={18} className="text-[var(--color-signal)]" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-[var(--color-snow)]">
              See how this compares to your other sessions
            </p>
            <p className="text-[12px] text-[var(--color-slate)] mt-0.5">
              Composure trend, practice streak, and your current focus area.
            </p>
          </div>
          <ArrowRight
            size={16}
            className="text-[var(--color-slate)] group-hover:text-[var(--color-signal)] group-hover:translate-x-0.5 transition-all shrink-0"
          />
        </Link>

        {/* Actions */}
        <div className="flex flex-wrap items-center gap-3 pt-2">
          {/* Practising the same resume again is the common case, and it used to
              cost a re-upload of a PDF the app had already parsed. guard_session
              still holds the parsed text and questions, so /sentry can start
              cold — only the finished report is cleared. */}
          <button
            onClick={() => {
              sessionStorage.removeItem("guard_report");
              // Deliberately NOT clearing guard_vision_session here — the next
              // run mints its own id, and wiping it early would strand the
              // timeline if the user comes back to this page.
              router.push("/sentry");
            }}
            className="h-10 px-5 rounded-md bg-[var(--color-signal)] text-[var(--color-canvas)] text-sm font-semibold hover:brightness-110 transition-all cursor-pointer flex items-center gap-2"
          >
            <RotateCcw size={14} />
            New Session
          </button>
          {/* The separate, rarer path: practise against a different resume. This
              one does have to drop the parsed session. */}
          <button
            onClick={() => {
              sessionStorage.removeItem("guard_report");
              sessionStorage.removeItem("guard_session");
              router.push("/upload");
            }}
            className="h-10 px-5 rounded-md border border-[var(--color-hairline)] bg-[var(--color-surface)] text-[var(--color-parchment)] text-sm font-medium hover:border-[var(--color-signal)] hover:text-[var(--color-signal)] transition-colors cursor-pointer flex items-center gap-2"
          >
            <FileUp size={14} />
            Upload new resume
          </button>
          {/* A third dashboard link used to sit here, alongside the header chip
              and the hand-off card above. Two routes to the same place is
              already generous; three was noise. */}
        </div>

        {/* Footer */}
        <div className="mt-4 flex items-center justify-center">
          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-[11px] font-medium border border-[var(--color-hairline)] bg-[var(--color-surface)] text-[var(--color-slate)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)] pulse-signal" />
            Local · Edge Processing · No data leaves your device
          </span>
        </div>
      </motion.div>
    </main>
  );
}
