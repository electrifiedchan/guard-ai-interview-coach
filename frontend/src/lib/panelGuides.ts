/**
 * Plain-language decoder for every dashboard chart.
 *
 * The panels render real telemetry, but a sparkline of "composure" or a ring
 * marked 62 means nothing without being told what it measures and what good
 * looks like. Rather than crowd each panel with permanent caption text, the
 * copy lives here and Panel reveals it on demand behind a "?".
 *
 * Numbers quoted below must track the engine:
 *   - readiness bands  → backend/core_memory/timeline.py::_readiness_band
 *   - talking band     → TALKING_BAND in @/lib/dashboard
 *   - gaze buckets     → the engine's STRAIGHT | SIDE_OR_UP | DOWN states
 * If those move and these don't, the dashboard starts lying confidently.
 */
export interface PanelGuide {
  /** What the marks on screen literally represent. */
  reading: string;
  /** The target — what a good result looks like. */
  healthy: string;
  /** How to move the number, stated as an action. */
  action: string;
}

export const PANEL_GUIDES = {
  composure: {
    reading:
      "One point per session, left to right in time order. The line is your average composure for that session; the shaded band under it reaches down to your worst moment in the same session.",
    healthy:
      "A line that climbs — or at least holds — while the band gets thinner. A thin band means you stayed steady; a deep one means you had a bad patch even if the average looked fine.",
    action:
      "Chase the low points, not the average. One 20-second collapse drags a whole session down and is easier to fix than lifting everything by a few points.",
  },
  readiness: {
    reading:
      "A single 0–100 score: the average, across all your sessions, of a blend of eye contact, speaking time, recoveries and consistency. The ticks on the track are the band boundaries at 40, 70 and 85; the brighter tick at 70 is where 'Interview Ready' starts.",
    healthy:
      "70 or above. Below 40 is Building, 40–70 Improving, 85+ Sharp. The number under the ring shows how your latest session moved that average, so a small positive step every session beats one big jump.",
    action:
      "This is a lagging score — it moves when the habits underneath it move. Fix the panel marked 'Work on this next' and this follows within a few sessions.",
  },
  streak: {
    reading:
      "One square per day for the last 12 weeks, oldest on the left. Darker squares are days you practised; the brighter the square, the more you did that day.",
    healthy:
      "Unbroken columns matter more than dense ones. Three short sessions across three days beat one long session and six blanks.",
    action:
      "If you're about to break a streak, do one session and stop. Showing up is the variable being measured here, not duration.",
  },
  gaze: {
    reading:
      "Where your eyes spent the session, as a share of total time. Center is on the camera eye-line, Away is to the side or up, Down is below the screen.",
    healthy:
      "Center above 70%. Some Away time is normal and reads as thinking — Down is the one that hurts, because it looks like reading or checking something off-screen.",
    action:
      "Move your notes to just beside the webcam so glancing at them stays inside the Center/Away range instead of dropping your eyeline.",
  },
  sessions: {
    reading:
      "Your most recent runs, newest first, with the date, how long you practised and the composure score that session earned.",
    healthy:
      "Scores that cluster tightly. Wild swings between sessions usually mean setup is changing — different room, different lighting, different time of day.",
    action:
      "Open a session to see the frames behind its score. A number tells you something went wrong; the frame tells you what.",
  },
} as const satisfies Record<string, PanelGuide>;

export type PanelGuideKey = keyof typeof PANEL_GUIDES;
