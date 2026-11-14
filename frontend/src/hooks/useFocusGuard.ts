"use client";

/**
 * useFocusGuard — notices when the candidate leaves the interview.
 *
 * Two independent signals, each measured as an episode with a start and an end:
 *
 *   TAB_SWITCH       the page is hidden (tab switch, minimise) or the window
 *                    lost focus (Alt-Tab to another app). Ends when the page is
 *                    visible AND focused again.
 *   FULLSCREEN_EXIT  fullscreen was entered for this run and then left (Esc).
 *                    Ends when fullscreen is re-entered, which the browser only
 *                    allows from a click, hence resume().
 *
 * A browser cannot prevent either of these. What it can do is detect them,
 * pause the interview, warn on return, and record how long the candidate was
 * away. That is all this hook does; the page decides what pausing means.
 *
 * Blips shorter than GRACE_MS (an OS notification stealing focus for a moment)
 * neither pause the interview nor get recorded.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

export type FocusEventKind = "TAB_SWITCH" | "FULLSCREEN_EXIT";

export interface FocusEpisode {
  kind: FocusEventKind;
  durationSec: number;
}

export interface FocusGuardState {
  /** The interview is paused and waiting for the candidate to continue. */
  paused: boolean;
  /** The candidate is still away (page hidden or another window focused). */
  away: boolean;
  /** Fullscreen was on for this run and is currently off. */
  fullscreenLost: boolean;
  /** Length of the most recent away period, for the warning copy. */
  lastAwaySec: number;
  /** How many times this run has been paused. */
  reminders: number;
}

interface Options {
  /** Track only while the interview is actually running. */
  active: boolean;
  /** A sustained away period began: pause the interview. */
  onPause: () => void;
  /** The candidate clicked Continue: pick the interview back up. */
  onResume: () => void;
  /** An away period finished (or was flushed by stop()); record it. */
  onEpisode: (episode: FocusEpisode) => void;
}

const GRACE_MS = 1000;

const INITIAL: FocusGuardState = {
  paused: false,
  away: false,
  fullscreenLost: false,
  lastAwaySec: 0,
  reminders: 0,
};

function fullscreenSupported(): boolean {
  return typeof document !== "undefined" && !!document.documentElement?.requestFullscreen;
}

function isFullscreen(): boolean {
  return typeof document !== "undefined" && !!document.fullscreenElement;
}

function pageIsAway(): boolean {
  return document.visibilityState === "hidden" || !document.hasFocus();
}

export function useFocusGuard({ active, onPause, onResume, onEpisode }: Options) {
  const [state, setState] = useState<FocusGuardState>(INITIAL);

  // Callbacks change identity every render on the page; read them through refs
  // so the DOM listeners can stay bound once.
  const onPauseRef = useRef(onPause);
  const onResumeRef = useRef(onResume);
  const onEpisodeRef = useRef(onEpisode);
  useLayoutEffect(() => {
    onPauseRef.current = onPause;
    onResumeRef.current = onResume;
    onEpisodeRef.current = onEpisode;
  });

  // Synchronous mirrors. stop() must take effect before the exitFullscreen()
  // it triggers fires its own change event, which React state cannot do.
  const trackingRef = useRef(false);
  const pausedRef = useRef(false);
  // Whether fullscreen was successfully entered for this run. Only then is
  // leaving it an event; a browser that refused fullscreen is not penalised.
  const fsArmedRef = useRef(false);
  const tabStartRef = useRef<number | null>(null);
  const fsStartRef = useRef<number | null>(null);
  const graceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearGrace = () => {
    if (graceTimerRef.current) {
      clearTimeout(graceTimerRef.current);
      graceTimerRef.current = null;
    }
  };

  /** Close an open episode and report it if it outlasted the grace period. */
  const closeEpisode = useCallback((kind: FocusEventKind, startRef: { current: number | null }) => {
    const started = startRef.current;
    startRef.current = null;
    if (started === null) return 0;
    const sec = (performance.now() - started) / 1000;
    if (sec * 1000 >= GRACE_MS) onEpisodeRef.current({ kind, durationSec: Math.round(sec * 10) / 10 });
    return sec;
  }, []);

  /** Something just went away; pause once it has lasted past the grace period. */
  const armPause = useCallback(() => {
    if (pausedRef.current || graceTimerRef.current) return;
    graceTimerRef.current = setTimeout(() => {
      graceTimerRef.current = null;
      if (!trackingRef.current || pausedRef.current) return;
      const stillAway = tabStartRef.current !== null || fsStartRef.current !== null;
      if (!stillAway) return;
      pausedRef.current = true;
      // lastAwaySec belongs to this pause; filled in when the tab episode ends.
      setState((s) => ({ ...s, paused: true, lastAwaySec: 0, reminders: s.reminders + 1 }));
      onPauseRef.current();
    }, GRACE_MS);
  }, []);

  /** Re-read the page and update both episodes from it. */
  const sync = useCallback(() => {
    if (!trackingRef.current) return;

    const away = pageIsAway();
    if (away && tabStartRef.current === null) {
      tabStartRef.current = performance.now();
      armPause();
    } else if (!away && tabStartRef.current !== null) {
      const sec = closeEpisode("TAB_SWITCH", tabStartRef);
      if (sec * 1000 >= GRACE_MS) setState((s) => ({ ...s, lastAwaySec: sec }));
    }

    const fsLost = fsArmedRef.current && !isFullscreen();
    if (fsLost && fsStartRef.current === null) {
      fsStartRef.current = performance.now();
      armPause();
    } else if (!fsLost && fsStartRef.current !== null) {
      closeEpisode("FULLSCREEN_EXIT", fsStartRef);
    }

    if (tabStartRef.current === null && fsStartRef.current === null) clearGrace();
    setState((s) =>
      s.away === away && s.fullscreenLost === fsLost ? s : { ...s, away, fullscreenLost: fsLost },
    );
  }, [armPause, closeEpisode]);

  // Bind listeners only while the interview runs.
  useEffect(() => {
    if (!active) return;
    trackingRef.current = true;
    sync(); // the candidate may already be away when the interview starts

    document.addEventListener("visibilitychange", sync);
    document.addEventListener("fullscreenchange", sync);
    window.addEventListener("blur", sync);
    window.addEventListener("focus", sync);
    return () => {
      document.removeEventListener("visibilitychange", sync);
      document.removeEventListener("fullscreenchange", sync);
      window.removeEventListener("blur", sync);
      window.removeEventListener("focus", sync);
    };
  }, [active, sync]);

  /**
   * Enter fullscreen. Must be called straight from a click handler: browsers
   * refuse requestFullscreen() without a user gesture.
   */
  const enterFullscreen = useCallback(async () => {
    if (!fullscreenSupported()) return false;
    if (isFullscreen()) {
      fsArmedRef.current = true;
      return true;
    }
    try {
      await document.documentElement.requestFullscreen();
      fsArmedRef.current = true;
      return true;
    } catch {
      // Refused (iframe, policy, unsupported). Carry on without fullscreen
      // rather than blocking the interview on a browser setting.
      return false;
    }
  }, []);

  /** The Continue button. Restores fullscreen if needed, then resumes. */
  const resume = useCallback(async () => {
    if (pageIsAway()) return;
    if (fsArmedRef.current && !isFullscreen()) {
      const ok = await enterFullscreen();
      if (!ok) {
        // Could not get fullscreen back; stop asking for it this run.
        fsArmedRef.current = false;
        closeEpisode("FULLSCREEN_EXIT", fsStartRef);
      }
    }
    sync();
    if (tabStartRef.current !== null || fsStartRef.current !== null) return;
    clearGrace();
    pausedRef.current = false;
    setState((s) => ({ ...s, paused: false }));
    onResumeRef.current();
  }, [enterFullscreen, closeEpisode, sync]);

  /**
   * End of run. Flushes any open episode, stops tracking and leaves
   * fullscreen. Synchronous so it wins the race with the change event that
   * exitFullscreen() fires.
   */
  const stop = useCallback(() => {
    trackingRef.current = false;
    clearGrace();
    closeEpisode("TAB_SWITCH", tabStartRef);
    closeEpisode("FULLSCREEN_EXIT", fsStartRef);
    pausedRef.current = false;
    fsArmedRef.current = false;
    setState(INITIAL);
    if (isFullscreen()) document.exitFullscreen().catch(() => {});
  }, [closeEpisode]);

  // Unmount (navigating away mid-run) behaves like stop(), minus state updates.
  useEffect(() => () => {
    trackingRef.current = false;
    clearGrace();
  }, []);

  return { ...state, pausedRef, enterFullscreen, resume, stop };
}
