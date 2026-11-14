"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Smartphone, User, Users, Bot, EyeOff, Maximize } from "lucide-react";
import SniperScope, { RiskPacket, ScopeInterrupt, SniperScopeHandle } from "@/components/SniperScope";
import { PERSONA_LABELS, DEFAULT_PERSONA, type PersonaId } from "@/lib/personas";
import VoiceOrb, { VoiceState } from "@/components/VoiceOrb";
import LoadingOverlay from "@/components/LoadingOverlay";
import DashboardButton from "@/components/DashboardButton";
import { useVAD } from "@/hooks/useVAD";
import { useFocusGuard, type FocusEpisode } from "@/hooks/useFocusGuard";
import { beginSpeech, endSpeech, noteWordBoundary } from "@/lib/speechLevel";


type ChatMessage = {
  role: "interviewer" | "candidate";
  content: string;
  persona?: string;
};

type TurnState = "ai-speaking" | "listening" | "processing" | "idle";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

// How long a reminder deferred by a question stays worth saying. Past this, the
// moment has passed and the report is the right place for it — see the drain
// effect for why this is a proxy rather than a real liveness check.
const NUDGE_TTL_MS = 15_000;

// How long the banner stays up. Long enough to read after looking back at the
// screen, short enough not to sit over the video for the rest of the run.
const NUDGE_BANNER_MS = 9_000;

function fmtAway(sec: number): string {
  const whole = Math.max(1, Math.round(sec));
  if (whole < 60) return `${whole} second${whole === 1 ? "" : "s"}`;
  const m = Math.floor(whole / 60);
  const s = whole % 60;
  return s ? `${m}m ${s}s` : `${m} minute${m === 1 ? "" : "s"}`;
}

export default function SentryPage() {
  const router = useRouter();
  const sniperRef = useRef<SniperScopeHandle>(null);

  // Interview state — starts after sentry is engaged
  const [interviewActive, setInterviewActive] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const sessionIdRef = useRef("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  // Two different things, deliberately two states. `currentPersona` is whatever
  // rung the engine reports *now* — it escalates on its own, so it drifts away
  // from the pick mid-session and drives the badge. `startingPersona` is the
  // user's pre-engage choice, mirrored up from the scope's picker via
  // onPersonaChange so start-session can seed the ladder with it. The scope
  // holds the source of truth; this copy is a mirror, not a second owner.
  const [currentPersona, setCurrentPersona] = useState<string>(DEFAULT_PERSONA);
  const [startingPersona, setStartingPersona] = useState<PersonaId>(DEFAULT_PERSONA);
  const [turnState, setTurnState] = useState<TurnState>("idle");
  // Mirror of turnState for the interrupt path, which is called from a ref-bound
  // callback and so would otherwise read render 1's value forever.
  const turnStateRef = useRef<TurnState>("idle");
  turnStateRef.current = turnState;

  /**
   * The coaching reminder currently on screen, if any. Rendered as a banner over
   * the scope AND spoken, because both findings mean the candidate is probably
   * not looking at the screen — a silent banner would be missed at exactly the
   * moment it matters.
   */
  const [coachNudge, setCoachNudge] = useState<ScopeInterrupt | null>(null);
  // A reminder that arrived mid-question and is waiting for a gap to speak in.
  const pendingNudgeRef = useRef<{ nudge: ScopeInterrupt; at: number } | null>(null);
  const nudgeSpeakingRef = useRef(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);

  // Transcript (before interview starts, shows system status)
  const [liveTranscript, setLiveTranscript] = useState("SYSTEM STANDBY. UPLOAD COMPLETE. ENGAGE SENTRY TO BEGIN.");

  // Vision telemetry from SniperScope
  const [riskScore, setRiskScore] = useState(0);
  const riskScoreRef = useRef(0);
  const [interventionLevel, setInterventionLevel] = useState("CLEAR");

  const chatEndRef = useRef<HTMLDivElement>(null);
  const interviewStartedRef = useRef(false);
  /**
   * Latched once the run is over — by "End session", by Disengage, or by the
   * interview completing on its own.
   *
   * Two things outlive the click that ends a run. stopListening() can't retract
   * a recording the VAD has already captured, so a final blob would still POST
   * a turn against a session the backend has closed. And a turn already in
   * flight owns a chain of awaits that keeps running to completion — it would
   * speak its reply and reopen the microphone on a session the user just left.
   * Neither can be cancelled, so both are made to check this flag and bail.
   */
  const sessionEndingRef = useRef(false);

  /**
   * True while the candidate is away from the window (tab switch, Alt-Tab,
   * fullscreen exit) and until they press Continue. Every path that would
   * speak or open the microphone checks it, the same way they check
   * sessionEndingRef — a turn already in flight cannot be cancelled, so it is
   * made to stop at its next step instead, and resumeInterview() picks up.
   */
  const interviewPausedRef = useRef(false);
  /** The interviewer's latest line, and whether the candidate heard all of it. */
  const lastLineRef = useRef("");
  const lineHeardRef = useRef(true);
  /** The engine said the interview is complete; end it once the last line lands. */
  const completeRef = useRef(false);
  /** Resolves the in-flight speakText() promise, so a pause can cut it off. */
  const settleSpeechRef = useRef<((ok: boolean) => void) | null>(null);

  // VAD hook
  const { isVoiceActive, audioLevel, audioLevelRef, startListening, stopListening } = useVAD({
    silenceThresholdMs: 2500,
    volumeThreshold: 0.01,
    onSpeechEnd: handleSpeechEnd,
  });

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  /**
   * startListening() opens the microphone asynchronously and rejects when the
   * browser refuses — permission revoked mid-run, device busy, no input at all.
   * Every call site used to fire it and walk away, so a refusal surfaced only as
   * an unhandled rejection in the console: turnState had already been set to
   * "listening" on the line above, so the orb sat on "Waiting for voice" and the
   * candidate talked into a microphone that was never open. Route every call
   * through here so a failure is something the user can actually see and fix.
   */
  const beginListening = useCallback(async () => {
    // Never open the mic while the candidate is away or the run is over;
    // resumeInterview() reopens it when they come back.
    if (interviewPausedRef.current || sessionEndingRef.current) return;
    try {
      await startListening();
    } catch (err) {
      console.error("[MIC] Failed to open microphone:", err);
      const denied = err instanceof Error && err.name === "NotAllowedError";
      setTurnState("idle");
      setLiveTranscript(
        denied
          ? "MICROPHONE BLOCKED. ALLOW ACCESS, THEN RE-ENGAGE."
          : "MICROPHONE UNAVAILABLE. CHECK YOUR INPUT DEVICE."
      );
      setError(
        denied
          ? "Microphone permission denied — allow it in the browser, then re-engage."
          : "Could not open the microphone. Another app may be holding it."
      );
    }
  }, [startListening]);

  // Redirect if no session data. Identity resolution used to live here too;
  // DashboardButton owns that now, so this effect is back to one job.
  useEffect(() => {
    if (!sessionStorage.getItem("guard_session")) {
      router.push("/upload");
    }
  }, [router]);

  // Refreshing this page throws the interview away — `guard_session` survives
  // in sessionStorage so the guard above still passes, but the session id,
  // transcript and persona progression are all React-local and gone. The
  // confirm-then-return-to-landing behaviour that covers it is global rather
  // than wired up here; see components/RefreshGuard.tsx and lib/refreshPolicy.ts.

  // Map turnState → VoiceOrb visual state
  const orbState: VoiceState =
    turnState === "ai-speaking" ? "SPEAK" :
    turnState === "listening" ? "LISTEN" :
    turnState === "processing" ? "PROCESS" :
    "IDLE";

  // TTS
  // Resolves true when the utterance played to the end, false when it was
  // refused or cut off (cancelled, errored, or the interview paused).
  const speakText = useCallback((text: string): Promise<boolean> => {
    return new Promise((resolve) => {
      // The run ended while this turn was still in flight. Cancelling the
      // synthesiser at that moment can't suppress an utterance that hasn't been
      // queued yet, so the request has to be refused here instead. Same for a
      // reply that lands while the candidate is away from the window.
      if (sessionEndingRef.current || interviewPausedRef.current) { resolve(false); return; }
      if (!("speechSynthesis" in window)) { resolve(true); return; }
      // The cancel() below cuts off whatever was playing; settle its promise
      // now rather than relying on onend/onerror, which cancel() does not
      // reliably fire.
      settleSpeechRef.current?.(false);
      window.speechSynthesis.cancel();
      let settled = false;
      const settle = (ok: boolean) => {
        if (settled) return;
        settled = true;
        if (settleSpeechRef.current === settle) settleSpeechRef.current = null;
        resolve(ok);
      };
      settleSpeechRef.current = settle;
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.92;
      utterance.pitch = 0.85;
      const voices = window.speechSynthesis.getVoices();
      const preferred =
        voices.find(
          (v) =>
            v.lang.startsWith("en") &&
            /\b(male|david|james|guy|mark|daniel|richard)\b/i.test(v.name)
        ) ||
        voices.find(
          (v) => v.lang.startsWith("en") && v.name.toLowerCase().includes("natural")
        ) ||
        voices.find((v) => v.lang.startsWith("en"));
      if (preferred) utterance.voice = preferred;

      // Drive the Audio Interrogator waveform from the synthesiser's own
      // progress. `boundary` fires as each word starts, which is the only
      // observable we get — speechSynthesis has no AudioNode to analyse.
      utterance.onstart = () => beginSpeech();
      utterance.onboundary = (e) => {
        if (e.name === "sentence") return; // word events carry the rhythm
        noteWordBoundary(e.charLength);
      };
      utterance.onend = () => { endSpeech(); settle(true); };
      utterance.onerror = () => { endSpeech(); settle(false); };
      window.speechSynthesis.speak(utterance);

    });
  }, []);

  /**
   * Speak one interviewer line and remember whether it was heard in full. If a
   * pause cuts it off, resumeInterview() repeats it, because the candidate
   * cannot answer a question they only heard half of.
   */
  const speakInterviewer = useCallback(async (text: string): Promise<boolean> => {
    lastLineRef.current = text;
    lineHeardRef.current = false;
    const ok = await speakText(text);
    if (ok && !interviewPausedRef.current) lineHeardRef.current = true;
    return lineHeardRef.current;
  }, [speakText]);

  /**
   * Say a coaching reminder out loud, mid-session.
   *
   * The whole point of this feature is that observing silently and explaining
   * afterwards tells the candidate what cost them the session only once it is too
   * late to stop doing it. But it has to be delivered without damaging the
   * interview itself, and there are two ways it could:
   *
   * 1. speakText() calls speechSynthesis.cancel() first, so speaking over a
   *    question would truncate it — and the candidate cannot answer something
   *    they never heard. So this defers unless the floor is genuinely free.
   * 2. During `listening` the microphone is LIVE. Speaking into it would have the
   *    VAD record our own reminder and post it to /voice/transcribe as the
   *    candidate's answer. So the mic is closed for the duration and reopened
   *    afterwards.
   *
   * `processing` counts as busy for the same reason as `ai-speaking`: the
   * interviewer's reply is already on its way back and will cancel() us mid-word
   * the moment it lands.
   */
  const speakNudge = useCallback(async (nudge: ScopeInterrupt) => {
    if (sessionEndingRef.current || nudgeSpeakingRef.current || interviewPausedRef.current) return;

    nudgeSpeakingRef.current = true;
    // Fresh object on purpose. A deferred reminder already put the banner up, and
    // its auto-clear timer has been running through the wait — re-setting the same
    // reference would be a no-op to React and leave the banner about to vanish
    // just as the audio starts. A new identity restarts the timer here, at the
    // moment the candidate actually hears it.
    setCoachNudge({ ...nudge });

    // Close the mic first if it is open, so the reminder cannot be transcribed as
    // an answer. Reopened below only if the run is still going.
    const micWasLive = turnStateRef.current === "listening";
    if (micWasLive) stopListening();

    try {
      await speakText(nudge.say);
    } finally {
      nudgeSpeakingRef.current = false;
      if (micWasLive && !sessionEndingRef.current) {
        setLiveTranscript("MICROPHONE LIVE. LISTENING…");
        beginListening();
      }
    }
  }, [speakText, beginListening, stopListening]);

  /**
   * A confirmed phone or second person, from either detection path.
   *
   * The backend fires once per sighting, so no de-duplication is needed here —
   * this is called on a rising edge, not every frame.
   */
  const handleInterrupt = useCallback((nudge: ScopeInterrupt) => {
    // Nobody is at the screen to hear it while the interview is paused. The
    // sighting is still in the report.
    if (sessionEndingRef.current || interviewPausedRef.current) return;

    const busy =
      nudgeSpeakingRef.current ||
      turnStateRef.current === "ai-speaking" ||
      turnStateRef.current === "processing";

    if (busy) {
      // One slot, newest wins. If a phone and a second person both land during a
      // question, only the later one is spoken: two reminders back to back is
      // ~20 seconds of talking over a live interview, which is the opposite of a
      // gentle nudge. The dropped one is still in the report, and a candidate
      // being told about one of the two will look at the screen anyway — where
      // the banner is.
      pendingNudgeRef.current = { nudge, at: Date.now() };
      // Banner goes up immediately even though the audio waits: the finding is
      // true now, and a candidate who happens to glance up should see it.
      setCoachNudge(nudge);
      return;
    }

    void speakNudge(nudge);
  }, [speakNudge]);

  // Drain the queue once the floor is free.
  //
  // NUDGE_TTL_MS is a staleness proxy, not a liveness check: the backend reports
  // the rising edge and there is no "the phone is gone" signal to wait for. A
  // reminder that has been queued through a whole answer is about a moment the
  // candidate has very likely already moved on from, and telling someone to put
  // away a phone they have already put away makes the system look like it is not
  // watching — the exact impression this feature exists to correct. Better to say
  // nothing; the moment is still in the report either way.
  useEffect(() => {
    if (turnState === "ai-speaking" || turnState === "processing") return;
    const queued = pendingNudgeRef.current;
    if (!queued) return;

    pendingNudgeRef.current = null;
    if (Date.now() - queued.at > NUDGE_TTL_MS) {
      setCoachNudge(null);
      return;
    }
    void speakNudge(queued.nudge);
  }, [turnState, speakNudge]);

  // Clear the banner a few seconds after the voice stops, so it does not sit over
  // the video for the rest of the run.
  useEffect(() => {
    if (!coachNudge) return;
    const t = setTimeout(() => setCoachNudge(null), NUDGE_BANNER_MS);
    return () => clearTimeout(t);
  }, [coachNudge]);

  /**
   * The candidate left the window (or fullscreen) for longer than the grace
   * period. Stop talking and close the mic so the interviewer neither asks a
   * question into an empty room nor records whatever is happening elsewhere.
   * Audio captured up to this point is discarded by handleSpeechEnd; the
   * candidate answers again after resuming.
   */
  const pauseInterview = useCallback(() => {
    if (sessionEndingRef.current) return;
    interviewPausedRef.current = true;
    stopListening();
    window.speechSynthesis?.cancel();
    settleSpeechRef.current?.(false);
    endSpeech();
    pendingNudgeRef.current = null;
    setCoachNudge(null);
    setLiveTranscript("INTERVIEW PAUSED. RETURN TO THE WINDOW TO CONTINUE.");
  }, [stopListening]);

  // Bound below, once endSession exists; the guard calls through it.
  const resumeInterviewRef = useRef<() => void>(() => {});

  /** Record one away period on the same timeline as the run's frames. */
  const reportFocusEpisode = useCallback((episode: FocusEpisode) => {
    let visionSession = "";
    try {
      visionSession = sessionStorage.getItem("guard_vision_session") || "";
    } catch {
      return;
    }
    if (!visionSession) return;
    // keepalive so the event still lands if the page is being torn down.
    fetch(`${API_BASE}/api/v1/session-event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      keepalive: true,
      body: JSON.stringify({
        session_id: visionSession,
        kind: episode.kind,
        duration_sec: episode.durationSec,
      }),
    }).catch((err) => console.warn("[FOCUS] Could not record event:", err));
  }, []);

  const focusGuard = useFocusGuard({
    active: interviewActive && !isGenerating,
    onPause: pauseInterview,
    onResume: () => resumeInterviewRef.current(),
    onEpisode: reportFocusEpisode,
  });
  const { stop: stopFocusGuard, enterFullscreen } = focusGuard;

  // When user clicks "Disengage" in SniperScope — stop everything
  const handleDisengage = useCallback(() => {
    // Latch before tearing anything down, so a turn mid-flight sees the run is
    // over the moment its next await resolves. cancel() alone only silences
    // what is already queued; the reply still on its way back from the backend
    // would arrive afterwards and start speaking again.
    sessionEndingRef.current = true;
    // Records any away period still open, leaves fullscreen, stops watching.
    stopFocusGuard();
    interviewPausedRef.current = false;
    settleSpeechRef.current?.(false);
    // A queued reminder must not survive the run it belonged to.
    pendingNudgeRef.current = null;
    setCoachNudge(null);
    stopListening();
    window.speechSynthesis?.cancel();
    // cancel() is not guaranteed to fire onend, which would strand the
    // Interrogator waveform at speaking level after the voice has stopped.
    endSpeech();
    setInterviewActive(false);
    setTurnState("idle");
    interviewStartedRef.current = false;
    setLiveTranscript("SENTRY DISENGAGED. PRESS ENGAGE TO RESUME.");
    setChatHistory([]);
    // Deliberately NOT resetting startingPersona here. The scope owns that state
    // and its picker is on screen in the idle panel, so the pick stays visible
    // and survives a Disengage/Engage cycle. Resetting it from this side would
    // leave the page sending friendly_hr while the picker still showed Tech Lead
    // — practice resets because its handleReset tears down to the upload screen,
    // which unmounts the picker entirely. Different situation, different rule.
  }, [stopListening, stopFocusGuard]);

  // Always points at the current render's startInterviewFlow. handleTelemetry
  // must keep a stable identity (SniperScope binds it once), but with empty
  // deps it froze render 1's startInterviewFlow — which closed over render 1's
  // startListening. After a Disengage/Engage cycle that stale mic handle was
  // dead, so re-engaging never restarted the interview.
  const startInterviewFlowRef = useRef<() => void>(() => {});

  // SniperScope telemetry — first callback means sentry is live → start interview
  const handleTelemetry = useCallback((packet: RiskPacket, verdict: string) => {
    setRiskScore(packet.risk_score);
    riskScoreRef.current = packet.risk_score;
    setInterventionLevel(packet.intervention_level);

    if (!interviewStartedRef.current) {
      interviewStartedRef.current = true;
      setLiveTranscript("SENTRY ENGAGED. INITIATING INTERVIEW SEQUENCE…");
      startInterviewFlowRef.current();
    }
  }, []);

  // Start conversational interview
  const startInterviewFlow = useCallback(async () => {
    const sessionData = sessionStorage.getItem("guard_session");
    if (!sessionData) return;

    const { questions, resume_text } = JSON.parse(sessionData);
    // Re-arm. Disengage latches the flag to kill the previous run; pressing
    // Engage starts a new one, and without this reset it would be born dead.
    sessionEndingRef.current = false;
    interviewPausedRef.current = false;
    lastLineRef.current = "";
    lineHeardRef.current = true;
    completeRef.current = false;
    setInterviewActive(true);
    setChatHistory([]);
    setTurnState("processing");
    setLiveTranscript("CONNECTING TO AI INTERVIEWER…");

    try {
      const res = await fetch(`${API_BASE}/api/v1/interview/start-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_text,
          questions,
          starting_persona: startingPersona,
        }),
      });

      if (!res.ok) throw new Error("Failed to start session");
      const data = await res.json();

      setSessionId(data.session_id);
      sessionIdRef.current = data.session_id;
      setCurrentPersona(data.persona);
      setChatHistory([{ role: "interviewer", content: data.opening_message, persona: data.persona }]);

      setTurnState("ai-speaking");
      setLiveTranscript(`AI: ${data.opening_message}`);
      await speakInterviewer(data.opening_message);

      // Disengaged during the opening line — leave the mic shut. Paused: the
      // resume path repeats the line and opens the mic.
      if (sessionEndingRef.current || interviewPausedRef.current) return;

      setTurnState("listening");
      setLiveTranscript("MICROPHONE LIVE. LISTENING…");
      beginListening();
    } catch (err) {
      console.error("Session start failed:", err);
      setError("Failed to start interview session. Check backend.");
      setLiveTranscript("ERROR: BACKEND UNREACHABLE.");
      // Let a retry actually re-run this; otherwise the guard stays latched
      // and Engage does nothing on the second attempt.
      interviewStartedRef.current = false;
      setInterviewActive(false);
    }
  }, [startingPersona, speakInterviewer, beginListening]);

  // Reassigned every render, which is what makes the persona pick reach the
  // backend: handleTelemetry is bound into SniperScope once and calls through
  // this ref, so it always invokes the newest closure rather than render 1's.
  startInterviewFlowRef.current = startInterviewFlow;

  // VAD speech end
  async function handleSpeechEnd(blob: Blob) {
    // The run is over (or never legitimately started) — discard the audio
    // instead of transcribing it and posting a turn nobody is listening for.
    if (sessionEndingRef.current || !sessionIdRef.current) return;
    // Captured as the candidate walked away (pausing stops the recorder, which
    // flushes what it had). A half answer is worse than none: drop it and let
    // them answer again after resuming.
    if (interviewPausedRef.current) return;

    setTurnState("processing");
    setIsTranscribing(true);
    setLiveTranscript("TRANSCRIBING AUDIO…");

    try {
      const formData = new FormData();
      formData.append("file", blob, "recording.webm");
      const transcribeRes = await fetch(`${API_BASE}/api/v1/voice/transcribe`, {
        method: "POST",
        body: formData,
      });

      if (!transcribeRes.ok) {
        const detail = await transcribeRes.json().catch(() => null);
        throw new Error(detail?.detail || `Transcription failed (${transcribeRes.status})`);
      }
      const { transcript } = await transcribeRes.json();
      setIsTranscribing(false);

      // Disengaged while the audio was still uploading.
      if (sessionEndingRef.current) return;

      if (!transcript || transcript.trim().length === 0) {
        // Whisper reports silence both for a candidate who said nothing and for
        // a clip it could not decode (voice_engine treats a broken WebM as
        // silence rather than raising). Reopening the mic with the same neutral
        // "LISTENING…" line made both cases invisible — the interviewer looked
        // like it had ignored the answer. Say what happened instead.
        console.warn("[STT] Empty transcript returned — clip was silent or undecodable.");
        setTurnState("listening");
        setLiveTranscript("DIDN'T CATCH THAT — SPEAK AGAIN.");
        beginListening();
        return;
      }

      setChatHistory((prev) => [...prev, { role: "candidate", content: transcript }]);
      setLiveTranscript(`CANDIDATE: "${transcript}"`);

      const turnRes = await fetch(`${API_BASE}/api/v1/interview/conversation-turn`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          transcript,
          focus_score: 100 - riskScoreRef.current,
        }),
      });

      if (!turnRes.ok) {
        const detail = await turnRes.json().catch(() => null);
        throw new Error(detail?.detail || `Conversation turn failed (${turnRes.status})`);
      }
      const turnData = await turnRes.json();

      // Disengaged while the interviewer was composing its reply. Dropping it
      // here keeps the answer out of a transcript the user has already closed.
      if (sessionEndingRef.current) return;

      setCurrentPersona(turnData.persona);
      setChatHistory((prev) => [...prev, {
        role: "interviewer",
        content: turnData.response,
        persona: turnData.persona,
      }]);

      if (turnData.is_complete) {
        completeRef.current = true;
        setTurnState("ai-speaking");
        setLiveTranscript(`AI: ${turnData.response}`);
        await speakInterviewer(turnData.response);
        // Paused during the closing line: resumeInterview() repeats it, then ends.
        if (interviewPausedRef.current) return;
        await endSession();
        return;
      }

      setTurnState("ai-speaking");
      setLiveTranscript(`AI: ${turnData.response}`);
      await speakInterviewer(turnData.response);
      if (sessionEndingRef.current || interviewPausedRef.current) return;
      setTurnState("listening");
      setLiveTranscript("MICROPHONE LIVE. LISTENING…");
      beginListening();

    } catch (err) {
      console.error("Turn processing error:", err);
      setIsTranscribing(false);
      // A run torn down mid-flight aborts its fetches, so the resulting error
      // is expected rather than a fault worth reporting — and recovering from
      // it would reopen the mic the user just closed.
      if (sessionEndingRef.current) return;
      setError(err instanceof Error ? err.message : "Something went wrong");
      setTurnState("listening");
      setLiveTranscript("ERROR. RESUMING LISTEN…");
      beginListening();
    }
  }

  // End session → navigate to report
  const endSession = useCallback(async () => {
    sessionEndingRef.current = true;
    // Before anything else: records any away period still open and leaves
    // fullscreen, so the report is not shown in a locked fullscreen window.
    stopFocusGuard();
    interviewPausedRef.current = false;
    settleSpeechRef.current?.(false);
    setIsGenerating(true);
    stopListening();
    window.speechSynthesis?.cancel();
    endSpeech();
    sniperRef.current?.stopCamera();

    setLiveTranscript("GENERATING PERFORMANCE REPORT…");

    try {
      const res = await fetch(`${API_BASE}/api/v1/interview/end-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          // The camera run behind this session's history entry. The backend
          // reads the focus score from those frames so the report and the
          // history show the same number.
          vision_session_id: (() => {
            try {
              return sessionStorage.getItem("guard_vision_session") || undefined;
            } catch {
              return undefined;
            }
          })(),
        }),
      });

      if (!res.ok) throw new Error("Verdict generation failed");
      const data = await res.json();

      sessionStorage.setItem(
        "guard_report",
        JSON.stringify({
          report: data.report,
          coaching: data.coaching,
          // Source field is `focus_score`; the sessionStorage key stays
          // `average_focus_score` because /report reads that.
          //
          // Only kept when it was measured from the camera frames, the same
          // data the history uses. A per-answer fallback has no history entry
          // to match, so it is dropped rather than shown as if it were one,
          // and the report shows "no camera data" instead of a made-up score.
          average_focus_score:
            data.focus_source === "camera" && typeof data.focus_score === "number"
              ? data.focus_score
              : null,
          turns_completed: data.turns_completed || 0,
        })
      );

      router.push("/report");
    } catch (err) {
      console.error("End session error:", err);
      setError(err instanceof Error ? err.message : "Failed to generate report");
      setIsGenerating(false);
      setLiveTranscript("ERROR: REPORT GENERATION FAILED.");
    }
  }, [stopListening, router, stopFocusGuard]);

  /**
   * The candidate is back and pressed Continue. Pick up where the pause hit:
   *
   * - a turn still in flight (`processing`) finishes on its own;
   * - an interviewer line that was cut off, or that arrived while they were
   *   away and was never spoken, is said again in full;
   * - otherwise they were mid-answer, so the mic reopens for a fresh answer.
   */
  const resumeInterview = useCallback(async () => {
    interviewPausedRef.current = false;
    if (sessionEndingRef.current || !interviewStartedRef.current) return;

    const state = turnStateRef.current;
    if (state === "processing") {
      setLiveTranscript("RESUMING INTERVIEW…");
      return;
    }
    if (state === "idle") return;

    if (!lineHeardRef.current && lastLineRef.current) {
      setTurnState("ai-speaking");
      setLiveTranscript(`AI: ${lastLineRef.current}`);
      const heard = await speakInterviewer(lastLineRef.current);
      if (!heard || sessionEndingRef.current) return;
      if (completeRef.current) {
        await endSession();
        return;
      }
    }

    if (!sessionIdRef.current) return;
    setTurnState("listening");
    setLiveTranscript("MICROPHONE LIVE. LISTENING…");
    beginListening();
  }, [speakInterviewer, endSession, beginListening]);
  resumeInterviewRef.current = () => { void resumeInterview(); };

  return (
    <main className="h-screen bg-[var(--color-canvas)] flex flex-col px-5 md:px-8 py-5 text-[var(--color-parchment)] items-center overflow-y-auto xl:overflow-hidden">
      {/* Top bar */}
      <header className="w-full max-w-[1400px] mb-4 flex justify-between items-center border-b border-[var(--color-hairline)] pb-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg lift-2 flex items-center justify-center">
            <span className="w-2 h-2 rounded-full bg-[var(--color-signal)] glow-pulse" />
          </div>
          <div className="flex flex-col">
            <h1 className="font-display text-[18px] font-semibold tracking-tight text-[var(--color-snow)] leading-none">
              GUARD
            </h1>
            <span className="eyebrow mt-1.5">Edge-AI Proctoring</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="hidden md:inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-[11px] font-medium border border-[var(--color-hairline)] bg-[var(--color-surface)] text-[var(--color-slate)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)] pulse-signal" />
            Local · 127.0.0.1
          </span>

          {interviewActive && (
            <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium uppercase border border-[var(--color-hairline)] bg-[var(--color-surface)] ${
              PERSONA_LABELS[currentPersona]?.color || "text-[var(--color-slate)]"
            }`}>
              <Bot size={10} />
              {PERSONA_LABELS[currentPersona]?.label || "Interviewer"}
            </span>
          )}

          {/* Gated on sessionId, not interviewActive. Disengage sets
              interviewActive=false, which used to disable this button and
              stranded a finished session with no way to generate its report.
              A session that has started can always be ended.

              Now the danger tone, not signal green: "End session" is a
              destructive/terminal action, so painting it the same emerald as
              every affirmative CTA read as "go" when it means "stop". */}
          <button
            onClick={() => endSession()}
            disabled={isGenerating || !sessionId}
            className="h-9 px-4 rounded-md bg-[var(--color-danger)] text-white text-[12px] font-semibold hover:opacity-90 transition-opacity disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
          >
            {isGenerating ? "Analyzing…" : "End session · generate report"}
          </button>

          {/* Profile chip is deliberately the last, rightmost element with a
              divider before it, so the coloured avatar owns the corner and
              stays visually separate from the status pills and the End action
              rather than mingling in the row. */}
          <span aria-hidden="true" className="mx-1 h-6 w-px bg-[var(--color-hairline)]" />
          <DashboardButton />
        </div>
      </header>

      {/* Dashboard grid */}
      <div className="w-full max-w-[1400px] flex flex-col xl:flex-row gap-4 flex-1 min-h-0">
        {/* Vision Sentry (left) */}
        <div className="relative flex-1 min-h-0 min-w-0">
          <SniperScope
            ref={sniperRef}
            onTelemetryUpdate={handleTelemetry}
            onDisengage={handleDisengage}
            onPersonaChange={setStartingPersona}
            onInterrupt={handleInterrupt}
            onEngageClick={() => { void enterFullscreen(); }}
          />

          {/* Coaching reminder. Amber, not red: this is a nudge about how the
              moment reads to an interviewer, not an accusation — the palette
              should not say "caught" when the copy deliberately does not. */}
          <AnimatePresence>
            {coachNudge && (
              <motion.div
                initial={{ opacity: 0, y: -12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.25 }}
                className="absolute top-4 left-4 right-4 z-30 flex items-start gap-3
                           rounded-lg border border-amber-400/40 bg-amber-950/85
                           px-4 py-3 backdrop-blur-sm"
                role="status"
                aria-live="polite"
              >
                {coachNudge.kind === "MOBILE_DEVICE" ? (
                  <Smartphone className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-300" />
                ) : (
                  <Users className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-300" />
                )}
                <p className="text-sm leading-snug text-amber-50">{coachNudge.say}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right sidebar — VoiceOrb + Transcript/Chat */}
        <div className="flex-shrink-0 flex flex-col gap-3 w-full xl:w-[420px] h-full overflow-hidden">
          {/* VoiceOrb — driven by interview turn state */}
          <VoiceOrb
            externalState={orbState}
            level={audioLevel}
            levelRef={audioLevelRef}
            voiceActive={isVoiceActive}
          />

          {/* Live transcript / Chat panel */}
          <div className="lift-1 rounded-lg p-4 flex flex-col flex-1 min-h-0 relative overflow-hidden">
            <div
              className={`absolute top-0 left-0 w-full h-px transition-colors duration-300 ${
                turnState === "listening" || isTranscribing ? "bg-[var(--color-signal)]" : "bg-[var(--color-hairline)]"
              }`}
            />
            <span className="eyebrow mb-3 flex items-center gap-2">
              <span
                className={`w-1.5 h-1.5 rounded-full transition-colors ${
                  interviewActive ? "bg-[var(--color-signal)] pulse-signal" : "bg-[var(--color-fog)]"
                }`}
              />
              {interviewActive ? "Interview transcript" : "Voice transcript"}
            </span>

            {/* Before interview: simple transcript line */}
            {!interviewActive && (
              <>
                <p className="text-[12.5px] leading-relaxed font-mono text-[var(--color-parchment)]">
                  <span className="text-[var(--color-fog)] mr-2">&gt;</span>
                  {liveTranscript}
                </p>
                {/* Said before Engage, not discovered mid-answer. */}
                <p className="mt-4 flex gap-2 text-[12px] leading-relaxed text-[var(--color-slate)]">
                  <Maximize className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                  <span>
                    The interview runs in fullscreen. If you switch tabs, change
                    windows or leave fullscreen, it pauses until you come back,
                    and the time away is noted in your report.
                  </span>
                </p>
              </>
            )}

            {/* During interview: chat messages */}
            {interviewActive && (
              <div className="flex-1 overflow-y-auto space-y-3">
                {chatHistory.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-2 ${msg.role === "candidate" ? "flex-row-reverse" : ""}`}
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
                      msg.role === "interviewer"
                        ? "bg-[var(--color-signal-soft)] text-[var(--color-signal)]"
                        : "bg-sky-500/10 text-sky-400"
                    }`}>
                      {msg.role === "interviewer" ? <Bot size={11} /> : <User size={11} />}
                    </div>
                    <div className={`max-w-[80%] rounded-lg px-3 py-2 text-[12px] leading-relaxed ${
                      msg.role === "interviewer"
                        ? "bg-neutral-900 text-[var(--color-snow)] border border-[var(--color-hairline)]"
                        : "bg-sky-500/10 text-sky-100 border border-sky-500/20"
                    }`}>
                      {msg.content}
                    </div>
                  </motion.div>
                ))}

                {(isTranscribing || turnState === "processing") && (
                  <div className="flex items-center gap-2 text-[var(--color-slate)]">
                    <div className="flex gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)] animate-bounce" style={{ animationDelay: "0ms" }} />
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)] animate-bounce" style={{ animationDelay: "150ms" }} />
                      <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-signal)] animate-bounce" style={{ animationDelay: "300ms" }} />
                    </div>
                    <span className="text-[10px] uppercase tracking-wider">
                      {isTranscribing ? "Transcribing…" : "Generating response…"}
                    </span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          {/* The separate level-bar panel used to live here. It duplicated what
              the waveform now shows (and half its bar heights were Math.random
              rather than signal), so the orb carries mic level and VAD gating. */}
        </div>
      </div>

      {/* Error toast */}
      {error && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-2 rounded-lg text-sm z-50">
          {error}
        </div>
      )}

      {/* Paused because the candidate left the window or fullscreen. Amber like
          the coach nudge: it explains how the moment would read in a real
          proctored interview, it does not accuse. */}
      <AnimatePresence>
        {focusGuard.paused && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-sm p-6"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="focus-pause-title"
            aria-describedby="focus-pause-body"
          >
            <div className="w-full max-w-[480px] rounded-lg border border-amber-400/40 bg-[var(--color-surface)] p-6 flex flex-col gap-4">
              <div className="flex items-center gap-3">
                <span className="w-9 h-9 rounded-md bg-amber-400/10 flex items-center justify-center">
                  {focusGuard.away
                    ? <EyeOff className="h-4 w-4 text-amber-300" />
                    : <Maximize className="h-4 w-4 text-amber-300" />}
                </span>
                <h2 id="focus-pause-title" className="text-[16px] font-semibold text-[var(--color-snow)]">
                  {focusGuard.away ? "Interview paused" : "Welcome back"}
                </h2>
              </div>

              <div id="focus-pause-body" className="flex flex-col gap-2 text-[13px] leading-relaxed text-[var(--color-parchment)]">
                {focusGuard.away ? (
                  <p>You left the interview window. Come back to this window to continue; the interviewer will wait.</p>
                ) : (
                  <>
                    <p>
                      {focusGuard.lastAwaySec > 0
                        ? `You were away from the interview window for ${fmtAway(focusGuard.lastAwaySec)}.`
                        : "You left fullscreen."}{" "}
                      In a real proctored interview, leaving the window can end the session, so try to stay on this screen until you finish.
                    </p>
                    <p className="text-[12px] text-[var(--color-slate)]">
                      Pause {focusGuard.reminders} this session · noted in your report.
                    </p>
                  </>
                )}
              </div>

              {!focusGuard.away && (
                <div className="flex flex-wrap items-center gap-3 pt-1">
                  <button
                    onClick={() => { void focusGuard.resume(); }}
                    autoFocus
                    className="h-9 px-4 rounded-md bg-[var(--color-signal)] text-[var(--color-canvas)] text-[13px] font-semibold hover:brightness-110 transition-all cursor-pointer flex items-center gap-2"
                  >
                    {focusGuard.fullscreenLost && <Maximize size={14} />}
                    {focusGuard.fullscreenLost ? "Return to fullscreen and continue" : "Continue interview"}
                  </button>
                  <button
                    onClick={() => endSession()}
                    disabled={isGenerating || !sessionId}
                    className="h-9 px-4 rounded-md border border-[var(--color-hairline)] text-[var(--color-parchment)] text-[13px] font-medium hover:border-[var(--color-danger)] hover:text-[var(--color-danger)] transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                  >
                    End session
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading overlay while generating report */}
      <LoadingOverlay open={isGenerating} />
    </main>
  );
}
