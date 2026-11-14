import sqlite3
import os
import time
import contextlib
from datetime import datetime, date, timedelta

DB_PATH = os.getenv(
    "GUARD_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "guard_telemetry.db")
)

# Kept for callers that import the historical constant. Duration metrics below
# use frame timestamps so sampling cadence cannot inflate reported practice time.
DASHBOARD_CADENCE_SEC = 1


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_timeline_tables() -> None:
    """Create all MR-1 tables if they don't already exist.

    Uses the Double-Context Pattern (contextlib.closing + conn as context
    manager) so the connection is always committed-or-rolled-back and closed
    without a separate conn.close() call — no resource leaks possible.
    """
    with contextlib.closing(_db()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                started_at   REAL,
                ended_at     REAL,
                frame_count  INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS timeline_frames (
                frame_id       TEXT PRIMARY KEY,
                session_id     TEXT,
                t              REAL,
                composure      REAL,
                gaze           TEXT,
                head_pose      TEXT,
                head_pose_raw  TEXT,
                gaze_class     TEXT,
                faces_detected INTEGER,
                is_talking     BOOLEAN
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS moments (
                moment_id    TEXT PRIMARY KEY,
                session_id   TEXT,
                t            REAL,
                type         TEXT,
                caption      TEXT,
                evidence_url TEXT,
                kind         TEXT,
                t_end        REAL,
                duration_sec REAL,
                frame_count  INTEGER
            )
        """)
        _migrate_columns(conn)


# Columns added after the first release. CREATE TABLE IF NOT EXISTS is a no-op on
# a database that already has the table, so a developer or participant machine
# carrying an older guard_telemetry.db would keep the old shape and every insert
# naming a new column would fail. ALTER TABLE ADD COLUMN is the only additive
# path SQLite offers; it backfills NULL, which is the honest value for a frame
# recorded before the field was captured.
_ADDED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "timeline_frames": [
        ("head_pose_raw", "TEXT"),
        ("gaze_class", "TEXT"),
    ],
    # Episode grain. `type` already holds the intervention level, so the flag
    # kind gets its own column rather than overloading it: the verdict page
    # groups by kind, and deriving kind by grepping the caption (which is what
    # _classify_moment_caption has to do for legacy rows) only ever worked for
    # the two object flags.
    "moments": [
        ("kind", "TEXT"),
        ("t_end", "REAL"),
        ("duration_sec", "REAL"),
        ("frame_count", "INTEGER"),
    ],
}


def _migrate_columns(conn: sqlite3.Connection) -> None:
    """Add any missing columns to existing tables. Safe to run on every startup."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if not existing:
            continue  # table absent entirely; CREATE above owns it
        for name, decl in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def insert_frame(
    frame_id: str,
    session_id: str,
    t: float,
    composure: float,
    gaze: str,
    head_pose: str,
    faces_detected: int,
    is_talking: bool,
    head_pose_raw: str | None = None,
    gaze_class: str | None = None,
) -> None:
    """Insert one telemetry frame into timeline_frames.

    head_pose_raw and gaze_class are the pre-fusion sensor calls; head_pose is
    the fused label the system acted on. They are keyword-defaulted so existing
    callers and tests keep working, and so a frame from a client that doesn't
    send them stores NULL rather than a fabricated reading.
    """
    with contextlib.closing(_db()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO timeline_frames
                (frame_id, session_id, t, composure, gaze, head_pose,
                 head_pose_raw, gaze_class, faces_detected, is_talking)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (frame_id, session_id, t, composure, gaze, head_pose,
             head_pose_raw, gaze_class, faces_detected, is_talking),
        )
        # Keep session row alive, bump frame counter, and keep ended_at current
        conn.execute(
            """
            INSERT INTO sessions (session_id, started_at, ended_at, frame_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(session_id) DO UPDATE SET
                frame_count = frame_count + 1,
                ended_at    = MAX(COALESCE(ended_at, 0), excluded.ended_at)
            """,
            (session_id, t, t),
        )


def insert_moment(
    moment_id: str,
    session_id: str,
    t: float,
    type_: str,
    caption: str,
    evidence_url: str | None = None,
    kind: str | None = None,
    t_end: float | None = None,
    duration_sec: float | None = None,
    frame_count: int = 1,
) -> None:
    """Insert a flagged moment into the moments table.

    A moment is one EPISODE — a contiguous stretch of one flag kind — not one
    frame. `t` is when it started, `t_end` when the behaviour last showed, and
    `duration_sec` the difference. They are NULL on insert for an episode still
    in progress and filled in later by close_moment(); a caller that already
    knows the whole span (the object sweep, which bills one prop once) may pass
    them straight in.

    `kind` is the flag identity used for grouping (MOBILE_DEVICE, NO_FACE,
    ATTENTION_DRIFT, ...). `type_` remains the intervention level, which is what
    the replay page reads.
    """
    with contextlib.closing(_db()) as conn, conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO moments
                (moment_id, session_id, t, type, caption, evidence_url,
                 kind, t_end, duration_sec, frame_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (moment_id, session_id, t, type_, caption, evidence_url,
             kind, t_end, duration_sec, frame_count),
        )


def close_moment(
    moment_id: str,
    t_end: float,
    duration_sec: float,
    frame_count: int,
) -> None:
    """Stamp an episode's end onto the row opened for it.

    Deliberately does NOT touch caption or evidence_url: the caption was written
    against the frame that caused the flag and the image is that frame. An
    episode ending changes how long it lasted, not what it was.
    """
    with contextlib.closing(_db()) as conn, conn:
        conn.execute(
            """
            UPDATE moments
               SET t_end = ?, duration_sec = ?, frame_count = ?
             WHERE moment_id = ?
            """,
            (t_end, duration_sec, frame_count, moment_id),
        )


# ===========================================================================
# MR 3 — Dashboard summary (cross-session aggregation)
# ===========================================================================
#
# Session id convention (forward-compatible):
#   Today the frontend does not yet mint a per-run session id, so session_id
#   falls back to candidate_id (see edge_main.analyze_frame). When per-run ids
#   are wired later, use "{candidate_id}__{uuid}". This function matches BOTH
#   "session_id == candidate_id" and "session_id LIKE candidate_id__%", so it
#   keeps working across that migration with zero data loss.
#
# Gaze buckets map to the three states the engine actually emits:
#   STRAIGHT -> center (on-screen) | SIDE_OR_UP -> away | DOWN -> down
# ---------------------------------------------------------------------------

def _readiness_band(score: float) -> str:
    if score >= 85:
        return "Sharp"
    if score >= 70:
        return "Interview Ready"
    if score >= 40:
        return "Improving"
    return "Building"


def _avg_composure(values) -> float:
    """The ONE definition of a session's average composure (0-100, 1 decimal).

    Every screen that shows a session's focus/composure score reads it through
    here: the dashboard session list and trend, the replay stats, and the report
    shown right after a session ends. They used to compute it separately (the
    report averaged one live reading per answer, the dashboard averaged every
    frame), so the same session showed two different numbers. A missing
    composure counts as 0, the same as the dashboard has always done.

    Callers must pass the frames oldest-first so the float sum is identical.
    """
    comps = [float(v) if v is not None else 0.0 for v in values]
    if not comps:
        return 0.0
    return round(sum(comps) / len(comps), 1)


def get_session_focus(session_id: str) -> dict | None:
    """Average composure for one run, or None if it recorded no frames.

    Exact session_id match: a run's frames are stored under its own
    "{candidate_id}__{rand}" id, and the dashboard groups on that same id.
    """
    with contextlib.closing(_db()) as conn:
        rows = conn.execute(
            "SELECT composure FROM timeline_frames WHERE session_id = ? ORDER BY t",
            (session_id,),
        ).fetchall()
    if not rows:
        return None
    return {
        "session_id": session_id,
        "avg_composure": _avg_composure(r[0] for r in rows),
        "frames": len(rows),
    }


def _score_session(eye_pct: float, avg_comp: float, streak_s: float, recoveries: int) -> float:
    """Weighted readiness blend for a single session, clamped 0-100.

    0.4*eye_contact + 0.3*avg_composure + 0.2*focus_streak_norm + 0.1*recovery_bonus
    focus_streak_norm: 120s of unbroken focus == full marks.
    recovery_bonus:    5 recoveries == full marks.
    """
    focus_streak_norm = min(streak_s / 120.0, 1.0) * 100.0
    recovery_bonus = min(recoveries / 5.0, 1.0) * 100.0
    score = (
        0.4 * eye_pct
        + 0.3 * avg_comp
        + 0.2 * focus_streak_norm
        + 0.1 * recovery_bonus
    )
    return round(max(0.0, min(100.0, score)), 1)


def get_dashboard_summary(candidate_id: str | None = None, days: int = 84) -> dict:
    """Aggregate cross-session telemetry into the MR-3 dashboard contract.

    Pure read over the timeline DB. Returns a stable shape even for a brand-new
    user with zero sessions (readiness is null in that case, never a 404).
    """
    cutoff = time.time() - days * 86400
    where = "t >= ?"
    params: list = [cutoff]
    if candidate_id:
        where += " AND (session_id = ? OR session_id LIKE ?)"
        params += [candidate_id, f"{candidate_id}__%"]

    with contextlib.closing(_db()) as conn:
        conn.row_factory = sqlite3.Row
        frames = [
            dict(r)
            for r in conn.execute(
                f"SELECT session_id, t, composure, gaze, is_talking "
                f"FROM timeline_frames WHERE {where} ORDER BY session_id, t",
                params,
            )
        ]
        moment_rows = [
            dict(r)
            for r in conn.execute(
                f"SELECT session_id, type FROM moments WHERE {where.replace('t >=', 't >=')}",
                params,
            )
        ]

    resolved_candidate = candidate_id or "candidate"

    # --- Empty user: return the full shape with zeros -----------------------
    if not frames:
        return {
            "candidate": {
                "candidate_id": resolved_candidate,
                "display_name": resolved_candidate,
                "preferred_name": None,
            },
            "totals": {
                "sessions": 0,
                "practice_seconds": 0,
                "current_streak_days": 0,
                "best_streak_days": 0,
            },
            "readiness": None,
            "metrics": [
                {"key": "eye_contact_pct", "label": "Eye contact", "value": 0, "unit": "%", "delta": 0, "spark": []},
                {"key": "talking_pct", "label": "Speaking time", "value": 0, "unit": "%", "delta": 0, "spark": []},
                {"key": "longest_focus_streak_s", "label": "Longest focus", "value": 0, "unit": "s", "delta": 0, "spark": []},
                {"key": "recovery_count", "label": "Recoveries", "value": 0, "unit": "", "delta": 0, "spark": []},
            ],
            "composure_trend": [],
            "activity": [],
            "gaze_split": None,
            "recent_sessions": [],
            "focus_area": None,
        }

    # --- Group frames per session (already ordered by session_id, t) --------
    per_session: dict[str, dict] = {}
    for f in frames:
        sid = f["session_id"]
        s = per_session.get(sid)
        if s is None:
            s = per_session[sid] = {
                "session_id": sid,
                "frames": 0,
                "started_at": f["t"],
                "ended_at": f["t"],
                "comps": [],
                "min_composure": f["composure"],
                "straight": 0,
                "side": 0,
                "down": 0,
                "talking": 0,
                # streak tracking uses timestamps, not frame count
                "_straight_started_at": None,
                "_last_straight_at": None,
                "max_streak_s": 0.0,
                # recovery tracking (dip < 50 then climb back >= 70)
                "_dipped": False,
                "recoveries": 0,
            }
        s["frames"] += 1
        s["ended_at"] = f["t"]
        comp = f["composure"] if f["composure"] is not None else 0.0
        s["comps"].append(comp)
        s["min_composure"] = min(s["min_composure"], comp)

        gaze = f["gaze"] or ""
        if gaze == "STRAIGHT":
            s["straight"] += 1
            if s["_straight_started_at"] is None:
                s["_straight_started_at"] = f["t"]
            s["_last_straight_at"] = f["t"]
            s["max_streak_s"] = max(
                s["max_streak_s"], f["t"] - s["_straight_started_at"]
            )
        else:
            s["_straight_started_at"] = None
            s["_last_straight_at"] = None
            if gaze == "SIDE_OR_UP":
                s["side"] += 1
            elif gaze == "DOWN":
                s["down"] += 1

        if f["is_talking"]:
            s["talking"] += 1

        if comp < 50:
            s["_dipped"] = True
        elif comp >= 70 and s["_dipped"]:
            s["recoveries"] += 1
            s["_dipped"] = False

    # Prefer explicit RECOVERY moments if the engine ever logs them
    for m in moment_rows:
        if (m.get("type") or "").upper() == "RECOVERY":
            sid = m["session_id"]
            if sid in per_session:
                per_session[sid]["recoveries"] += 1

    # --- Finalise per-session derived values (oldest -> newest) -------------
    sessions = sorted(per_session.values(), key=lambda s: s["started_at"])

    def _pct(part: int, whole: int) -> float:
        return round((part / whole) * 100, 1) if whole else 0.0

    computed = []
    for s in sessions:
        fc = s["frames"]
        eye = _pct(s["straight"], fc)
        talk = _pct(s["talking"], fc)
        streak_s = round(s["max_streak_s"], 1)
        avg_comp = _avg_composure(s["comps"])
        computed.append({
            "session_id": s["session_id"],
            "started_at": s["started_at"],
            "ended_at": s["ended_at"],
            "duration_s": int(max(0.0, s["ended_at"] - s["started_at"])),
            "frames": fc,
            "avg_composure": avg_comp,
            "min_composure": round(s["min_composure"], 1),
            "eye_contact_pct": eye,
            "talking_pct": talk,
            "longest_focus_streak_s": streak_s,
            "recovery_count": s["recoveries"],
            "readiness": _score_session(eye, avg_comp, streak_s, s["recoveries"]),
        })

    # --- Totals -------------------------------------------------------------
    total_sessions = len(computed)
    practice_seconds = sum(c["duration_s"] for c in computed)

    # Streak days over the calendar (based on session start dates)
    session_days = sorted({date.fromtimestamp(c["started_at"]) for c in computed})
    best_streak = cur = 1 if session_days else 0
    for i in range(1, len(session_days)):
        if (session_days[i] - session_days[i - 1]).days == 1:
            cur += 1
        else:
            cur = 1
        best_streak = max(best_streak, cur)
    # Current streak: consecutive days counting back from today (or last active day)
    current_streak = 0
    if session_days:
        today = date.today()
        anchor = today if today in session_days else session_days[-1]
        day_set = set(session_days)
        d = anchor
        while d in day_set:
            current_streak += 1
            d = d - timedelta(days=1)

    # --- Metrics (value = latest session, delta = latest - prev) ------------
    def _spark(key: str, cumulative: bool = False) -> list:
        vals = [c[key] for c in computed][-12:]
        if cumulative:
            run = 0
            out = []
            for v in vals:
                run += v
                out.append(run)
            return out
        return vals

    def _delta(key: str) -> float:
        if len(computed) < 2:
            return 0
        return round(computed[-1][key] - computed[-2][key], 1)

    latest = computed[-1]
    metrics = [
        {"key": "eye_contact_pct", "label": "Eye contact",
         "value": latest["eye_contact_pct"], "unit": "%",
         "delta": _delta("eye_contact_pct"), "spark": _spark("eye_contact_pct")},
        {"key": "talking_pct", "label": "Speaking time",
         "value": latest["talking_pct"], "unit": "%",
         "delta": _delta("talking_pct"), "spark": _spark("talking_pct")},
        {"key": "longest_focus_streak_s", "label": "Longest focus",
         "value": latest["longest_focus_streak_s"], "unit": "s",
         "delta": _delta("longest_focus_streak_s"), "spark": _spark("longest_focus_streak_s")},
        {"key": "recovery_count", "label": "Recoveries",
         "value": sum(c["recovery_count"] for c in computed), "unit": "",
         "delta": latest["recovery_count"], "spark": _spark("recovery_count", cumulative=True)},
    ]

    # --- Readiness: the AVERAGE across all sessions in the window -----------
    # One short or unlucky run must not swing the headline, so this is the mean
    # of every session's readiness. delta_vs_prev is how much the latest session
    # moved that average (average now minus average without it).
    avg_readiness = sum(c["readiness"] for c in computed) / len(computed)
    readiness = {
        "score": int(round(avg_readiness)),
        "delta_vs_prev": (
            int(round(avg_readiness - sum(c["readiness"] for c in computed[:-1]) / (len(computed) - 1)))
            if len(computed) >= 2 else 0
        ),
        "band": _readiness_band(avg_readiness),
    }

    # --- Composure trend ----------------------------------------------------
    composure_trend = [
        {
            "session_id": c["session_id"],
            "date": date.fromtimestamp(c["started_at"]).isoformat(),
            "avg_composure": c["avg_composure"],
            "min_composure": c["min_composure"],
        }
        for c in computed
    ]

    # --- Activity heatmap (sessions per calendar day, gaps omitted) ---------
    activity_map: dict[str, int] = {}
    for c in computed:
        d = date.fromtimestamp(c["started_at"]).isoformat()
        activity_map[d] = activity_map.get(d, 0) + 1
    activity = [{"date": d, "count": n} for d, n in sorted(activity_map.items())]

    # --- Recent sessions (newest first) -------------------------------------
    def _headline(c: dict) -> str:
        if c["avg_composure"] >= 80:
            return "Composed throughout — strong eye-line."
        if c["eye_contact_pct"] < 60:
            return "Eye contact drifted off-camera."
        if c["min_composure"] < 40:
            return "Recovered well after a shaky moment."
        return "Steady session with room to sharpen focus."

    recent_sessions = [
        {
            "session_id": c["session_id"],
            "started_at": datetime.fromtimestamp(c["started_at"]).isoformat(),
            "duration_s": c["duration_s"],
            "avg_composure": c["avg_composure"],
            "headline": _headline(c),
            "recovery_count": c["recovery_count"],
        }
        for c in reversed(computed)
    ]

    # --- Focus area (weakest signal -> actionable nudge) --------------------
    worst = min(computed, key=lambda c: c["avg_composure"])
    total_frames = sum(c["frames"] for c in computed)
    down_total = sum(s["down"] for s in per_session.values())
    side_total = sum(s["side"] for s in per_session.values())
    avg_eye = _pct(sum(s["straight"] for s in per_session.values()), total_frames)

    if avg_eye < 70:
        focus_area = {
            "key": "eye_contact",
            "title": "Hold your camera eye-line",
            "detail": f"Your gaze is centered {avg_eye:.0f}% of the time. Aim for 70%+ — "
                      f"look at the lens, not your own preview.",
            "cta_session_id": worst["session_id"],
        }
    elif down_total >= side_total and down_total > 0:
        focus_area = {
            "key": "gaze_drift",
            "title": "Gaze drift under pressure",
            "detail": "You tend to look down after a question lands. Keep notes at eye level "
                      "so a glance doesn't read as disengagement.",
            "cta_session_id": worst["session_id"],
        }
    else:
        focus_area = {
            "key": "composure",
            "title": "Steady your composure curve",
            "detail": f"Your lowest point dipped to {worst['min_composure']:.0f}. Practice a "
                      f"two-second reset breath before answering to flatten the dips.",
            "cta_session_id": worst["session_id"],
        }

    return {
        "candidate": {
            "candidate_id": resolved_candidate,
            "display_name": resolved_candidate,
            "preferred_name": None,
        },
        "totals": {
            "sessions": total_sessions,
            "practice_seconds": practice_seconds,
            "current_streak_days": current_streak,
            "best_streak_days": best_streak,
        },
        "readiness": readiness,
        "metrics": metrics,
        "composure_trend": composure_trend,
        "activity": activity,
        # Engine emits three gaze states, so three buckets (not the four in the UI spec).
        "gaze_split": {
            "center_pct": avg_eye,
            "away_pct": _pct(side_total, total_frames),
            "down_pct": _pct(down_total, total_frames),
        },
        "recent_sessions": recent_sessions,
        "focus_area": focus_area,
    }


def get_timeline(session_id: str) -> dict:
    """Fetch a session's replay payload: stats, headline, frames and moments.

    The replay page reads `stats`, `headline`, `frames[].composure` and
    `moments[]` directly. This used to return only the raw table rows, so the
    page threw on `data.stats.eye_contact_pct`. Percentages are derived the same
    way get_dashboard_summary derives them (STRAIGHT gaze / is_talking over the
    frame count) so replay and the dashboard can never disagree about a session.
    """
    with contextlib.closing(_db()) as conn:
        conn.row_factory = sqlite3.Row
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM timeline_frames WHERE session_id = ? ORDER BY t",
                (session_id,),
            )
        ]
        moment_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM moments WHERE session_id = ? ORDER BY t",
                (session_id,),
            )
        ]
        session = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

    # Timestamps are stored as absolute epoch seconds; the chart's x-axis is
    # elapsed time, so rebase on the first frame.
    t0 = rows[0]["t"] if rows else 0.0
    frames = [
        {"t": round(r["t"] - t0, 1), "composure": round(r["composure"] or 0, 1)}
        for r in rows
    ]
    moments = [
        {
            "t": round(m["t"] - t0, 1),
            "type": m["type"] or "EVENT",
            "caption": m["caption"] or "",
            "evidence_url": m.get("evidence_url"),
            # Flag identity (MOBILE_DEVICE, TAB_SWITCH, ...). NULL on rows
            # written before the column existed.
            "kind": m.get("kind"),
            "duration_sec": m.get("duration_sec"),
        }
        for m in moment_rows
    ]

    fc = len(rows)
    straight = sum(1 for r in rows if (r["gaze"] or "").upper() == "STRAIGHT")
    talking = sum(1 for r in rows if r["is_talking"])

    # Use actual timestamps so faster sampling does not inflate the streak.
    best_run_s = run_started_at = None
    for r in rows:
        if (r["gaze"] or "").upper() == "STRAIGHT":
            if run_started_at is None:
                run_started_at = r["t"]
            best_run_s = max(best_run_s or 0.0, r["t"] - run_started_at)
        else:
            run_started_at = None

    def _pct(part: int, whole: int) -> float:
        return round((part / whole) * 100, 1) if whole else 0.0

    eye_contact_pct = _pct(straight, fc)
    talking_pct = _pct(talking, fc)
    # Same function the dashboard and the post-session report use, so the three
    # can never show different numbers for one session.
    avg_composure = _avg_composure(r["composure"] for r in rows)

    if fc == 0:
        headline = "No telemetry recorded for this session."
    elif eye_contact_pct >= 80 and not moments:
        headline = "Composed throughout — strong eye-line."
    elif eye_contact_pct < 60:
        headline = "Eye contact drifted off-camera."
    elif moments:
        headline = f"{len(moments)} moment(s) flagged for review."
    else:
        headline = "Steady session with minor drift."

    return {
        "session_id": session_id,
        "session": dict(session) if session else None,
        "headline": headline,
        "stats": {
            "eye_contact_pct": eye_contact_pct,
            "talking_pct": talking_pct,
            "longest_focus_streak_s": round(best_run_s or 0.0, 1),
            "avg_composure": avg_composure,
            "frames": fc,
        },
        "frames": frames,
        "moments": moments,
    }
