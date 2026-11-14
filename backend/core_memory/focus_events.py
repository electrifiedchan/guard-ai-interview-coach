"""Focus events: the candidate left the interview window or fullscreen.

These come from the browser, not the camera, so there is no frame and no
evidence image. Each one is stored as a single closed moment (one row in the
`moments` table) with a measured duration, the same shape the prop sweep uses
for a phone sighting. That keeps them visible on /report, /replay and the
verdict breakdown without touching timeline_frames or the composure curve.

Deliberately NOT fed into the BEA risk engine: leaving the tab is reported
separately, so composure stays a measure of what the camera saw.
"""
from __future__ import annotations

import time
import uuid

from core_memory.timeline import insert_moment

# Browser-side event kinds. Kept in one place so the endpoint, the breakdown
# and the tests agree on spelling.
FOCUS_EVENT_KINDS: tuple[str, ...] = ("TAB_SWITCH", "FULLSCREEN_EXIT")

# The intervention level stored in moments.type. /replay colours moments by this
# column, and "WARNING" in it renders as a warning marker there.
FOCUS_EVENT_TYPE = "WARNING_LOGGED"

# Upper bound on a single away period. A tab left in the background over lunch
# is still one event; anything longer than this is a client clock problem.
MAX_DURATION_SEC = 6 * 60 * 60


def _fmt_seconds(sec: float) -> str:
    whole = int(round(sec))
    if whole < 60:
        return f"{whole}s"
    minutes, seconds = divmod(whole, 60)
    return f"{minutes}m {seconds}s" if seconds else f"{minutes}m"


def focus_event_caption(kind: str, duration_sec: float) -> str:
    """Coach-tone caption shown on the report and replay timelines."""
    span = _fmt_seconds(duration_sec)
    if kind == "TAB_SWITCH":
        return f"Left the interview window for {span}."
    if kind == "FULLSCREEN_EXIT":
        return f"Exited fullscreen for {span}."
    raise ValueError(f"Unknown focus event kind: {kind}")


def record_focus_event(
    session_id: str,
    kind: str,
    duration_sec: float,
    now: float | None = None,
) -> dict:
    """Store one completed away period as a closed moment and return it.

    The duration is measured client-side with a monotonic clock and the
    timestamps are anchored to the server clock here, so a skewed browser clock
    cannot place the event outside the session's frames.
    """
    if kind not in FOCUS_EVENT_KINDS:
        raise ValueError(f"Unknown focus event kind: {kind}")
    if not session_id:
        raise ValueError("session_id is required")
    duration = float(duration_sec)
    if not (0 <= duration <= MAX_DURATION_SEC):
        raise ValueError("duration_sec out of range")

    t_end = time.time() if now is None else now
    t_start = t_end - duration
    moment_id = uuid.uuid4().hex
    caption = focus_event_caption(kind, duration)

    insert_moment(
        moment_id,
        session_id,
        t_start,
        FOCUS_EVENT_TYPE,
        caption,
        None,          # no evidence image: nothing on camera to capture
        kind,
        t_end,
        duration,
        0,             # not derived from telemetry frames
    )
    return {
        "moment_id": moment_id,
        "session_id": session_id,
        "kind": kind,
        "t": t_start,
        "t_end": t_end,
        "duration_sec": duration,
        "caption": caption,
    }
