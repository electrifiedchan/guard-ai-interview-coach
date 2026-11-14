// Coach-tone copy for each canonical violation type emitted by the backend
// session-breakdown endpoint. Keep tone supportive and specific — never shame.

export type ViolationType =
  | "DOWN_GAZE"
  | "SIDE_GAZE"
  | "MOBILE_DEVICE"
  | "PROHIBITED_ITEM"
  | "MULTIPLE_FACES"
  | "NO_FACE"
  | "TALKING"
  | "TAB_SWITCH"
  | "FULLSCREEN_EXIT";

export type Severity = "info" | "warn" | "amber" | "danger";

export interface ViolationBucket {
  count: number;
  first_at: string | null;
  last_at: string | null;
  peak_risk: number;
  peak_intervention_level: string;
  evidence_paths: string[];
  sample_events: Array<{
    timestamp: string;
    risk_score: number;
    intervention_level: string;
    logic_trace: string;
  }>;
  approx_total_seconds: number;
}

export interface ViolationTemplate {
  title: string;
  severity: Severity;
  // Per-instance summary that quotes the real numbers from the bucket.
  whatHappened: (b: ViolationBucket) => string;
  whyItMatters: string;
  coachAdvice: string;
  /**
   * Second-person caption printed under the captured frame. The frame alone is
   * ambiguous — it is a photo of you sitting at a desk. This names what the
   * detector saw in it, which is the whole point of showing proof.
   */
  frameCaption: string;
}

export const VIOLATION_TEMPLATES: Record<ViolationType, ViolationTemplate> = {
  DOWN_GAZE: {
    title: "Eyes drifted downward",
    severity: "warn",
    whatHappened: (b) =>
      `Logged ${b.count} moment${b.count === 1 ? "" : "s"} where your gaze dropped off-screen — roughly ${b.approx_total_seconds}s cumulative.`,
    whyItMatters:
      "From a proctor's view, sustained downward gaze reads as referencing notes or a phone in your lap, even when nothing is actually there.",
    coachAdvice:
      "Park your hands and any reference material above the desk line. If you need to think, look slightly up or to the side instead of down — it reads as reflection, not avoidance.",
    frameCaption: "Your eyeline dropped below the screen here.",
  },
  SIDE_GAZE: {
    title: "Looked off-camera",
    severity: "warn",
    whatHappened: (b) =>
      `${b.count} event${b.count === 1 ? "" : "s"} of looking sideways or up — about ${b.approx_total_seconds}s total.`,
    whyItMatters:
      "Glancing off-screen suggests a second monitor, another person, or notes outside the frame. Even brief side-glances accumulate quickly.",
    coachAdvice:
      "Close everything except the interview window before you start. Keep one screen, one camera, one focal point. If your room has movement behind you, reposition so the wall is at your back.",
    frameCaption: "You looked away from the camera here.",
  },
  MOBILE_DEVICE: {
    title: "Phone visible in frame",
    severity: "danger",
    whatHappened: (b) =>
      `A mobile device was detected in ${b.count} frame${b.count === 1 ? "" : "s"} (peak risk ${b.peak_risk}%).`,
    whyItMatters:
      "Phones in frame are an immediate red flag for any proctored process — they short-circuit the rest of the conversation regardless of intent.",
    coachAdvice:
      "Phone goes face-down in another room, or in a drawer. If you need it for 2FA, bring it in only when prompted, then put it back. Never on the desk during a session.",
    frameCaption: "A phone was visible in the shot here.",
  },
  PROHIBITED_ITEM: {
    title: "Reference material visible",
    severity: "amber",
    whatHappened: (b) =>
      `Detected a book or secondary laptop in ${b.count} frame${b.count === 1 ? "" : "s"}.`,
    whyItMatters:
      "Books and second screens in frame imply you might be reading from them, even if they're closed or unrelated.",
    coachAdvice:
      "Clear the desk to the essentials: your monitor, keyboard, water, a single notepad. Move every other surface item out of camera reach before you begin.",
    frameCaption: "Reference material was in shot here.",
  },
  MULTIPLE_FACES: {
    title: "Another person in frame",
    severity: "danger",
    whatHappened: (b) =>
      `An additional face was visible in ${b.count} frame${b.count === 1 ? "" : "s"}.`,
    whyItMatters:
      "A second face — even a passerby — is one of the hardest patterns to recover from. It's read as collusion until proven otherwise.",
    coachAdvice:
      "Pick a closed room. Tell housemates the start and end time. A 'do not disturb' sign on the door costs nothing and removes this entire risk class.",
    frameCaption: "A second face appeared in the shot here.",
  },
  NO_FACE: {
    title: "Stepped out of frame",
    severity: "amber",
    whatHappened: (b) =>
      `You weren't visible to the camera for ${b.count} sample${b.count === 1 ? "" : "s"} (~${b.approx_total_seconds}s).`,
    whyItMatters:
      "An empty chair mid-session breaks continuity. Even a quick stretch can read as leaving to consult something.",
    coachAdvice:
      "Set up everything you need within arm's reach before you start: water, tissues, charger. Treat the session as a single uninterrupted block.",
    frameCaption: "You were out of the camera's view here.",
  },
  TALKING: {
    title: "Talking detected",
    severity: "info",
    whatHappened: (b) =>
      `Speech was detected in ${b.count} frame${b.count === 1 ? "" : "s"} outside expected response windows.`,
    whyItMatters:
      "Audible speech when you weren't expected to be answering can suggest someone else in the room or coaching from off-camera.",
    coachAdvice:
      "If you're a 'think out loud' type, that's fine — just keep it framed (\"let me think this through out loud…\"). Silence in the room otherwise is your friend.",
    frameCaption: "Speech was picked up here, off your answer turn.",
  },
  TAB_SWITCH: {
    title: "Left the interview window",
    severity: "amber",
    whatHappened: (b) =>
      `You switched away from the interview ${b.count} time${b.count === 1 ? "" : "s"}, about ${b.approx_total_seconds}s in total.`,
    whyItMatters:
      "Most proctored interviews watch for tab and window switches. Even a quick look at another tab can end the session or be read as looking something up.",
    coachAdvice:
      "Before you start, close other tabs and apps and silence notifications. If you need to check something, say so out loud first, as you would with a real interviewer.",
    frameCaption: "You were on another tab or window here.",
  },
  FULLSCREEN_EXIT: {
    title: "Left fullscreen",
    severity: "warn",
    whatHappened: (b) =>
      `Fullscreen was exited ${b.count} time${b.count === 1 ? "" : "s"}, about ${b.approx_total_seconds}s in total.`,
    whyItMatters:
      "Many assessment platforms require fullscreen and flag every exit, because it means other windows could be visible.",
    coachAdvice:
      "Stay in fullscreen for the whole session. Pressing Esc is the usual accident, so keep your hands away from the top-left of the keyboard while you answer.",
    frameCaption: "Fullscreen was off here.",
  },
};

// Severity → token map. These keys exist in globals.css.
export const SEVERITY_TOKENS: Record<Severity, { ring: string; dot: string; text: string }> = {
  info:   { ring: "border-[var(--color-info)]/30",   dot: "bg-[var(--color-info)]",   text: "text-[var(--color-info)]" },
  warn:   { ring: "border-[var(--color-warn)]/30",   dot: "bg-[var(--color-warn)]",   text: "text-[var(--color-warn)]" },
  amber:  { ring: "border-[var(--color-amber)]/30",  dot: "bg-[var(--color-amber)]",  text: "text-[var(--color-amber)]" },
  danger: { ring: "border-[var(--color-danger)]/30", dot: "bg-[var(--color-danger)]", text: "text-[var(--color-danger)]" },
};

export interface SessionBreakdown {
  candidate_id: string;
  total_events: number;
  violations_by_type: Partial<Record<ViolationType, ViolationBucket>>;
  peak_event: {
    timestamp: string;
    risk_score: number;
    intervention_level: string;
    logic_trace: string;
    evidence_path: string | null;
  } | null;
  session_window: {
    first_event_at: string | null;
    last_event_at: string | null;
  };
}
