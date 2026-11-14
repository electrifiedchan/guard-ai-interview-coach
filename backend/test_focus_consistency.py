"""A session's score must read the same everywhere it is shown.

The bug: the report shown right after a session averaged one live reading per
answer, while the history/dashboard averaged every recorded frame. The same run
showed two different numbers. These tests pin the fix: one definition of a
session's average composure, read by the report, the history list, the trend
chart and the replay stats alike. They also pin that the dashboard's headline
readiness is the average across sessions.

Runs against a throwaway temp SQLite DB via the GUARD_DB_PATH env override.

    pytest backend/test_focus_consistency.py
    python backend/test_focus_consistency.py     # plain-run fallback
"""
import asyncio
import importlib
import os
import tempfile
import time

_TMP_DB = os.path.join(tempfile.gettempdir(), f"guard_consistency_{os.getpid()}.db")
os.environ["GUARD_DB_PATH"] = _TMP_DB

from core_memory import timeline as tl
importlib.reload(tl)  # pick up the env override

from core_memory.conversation_engine import ConversationEngine, InterviewSession


def _fresh_db():
    if os.path.exists(_TMP_DB):
        os.remove(_TMP_DB)
    tl.init_timeline_tables()


def _seed_run(session_id: str, start: float, composures, gazes=None):
    gazes = gazes or ["STRAIGHT"] * len(composures)
    for i, (c, g) in enumerate(zip(composures, gazes)):
        tl.insert_frame(
            frame_id=f"{session_id}_{i}",
            session_id=session_id,
            t=start + i * 2,
            composure=c,
            gaze=g,
            head_pose="HEAD_CENTER",
            faces_detected=1,
            is_talking=False,
        )


def _interview(engine: ConversationEngine, turn_scores):
    session = InterviewSession(
        session_id="abc123",
        resume_text="Built a thing.",
        resume_questions=[{"question": "Tell me about it.", "focus": "general"}],
    )
    session.focus_scores.extend(turn_scores)
    engine.sessions["abc123"] = session
    return session


# The phone-at-the-end shape from a real session: a long steady stretch, then a
# tail pinned at 0. Per-answer readings taken during the steady stretch say ~80;
# the frames say 53.3. The fractions are chosen to sit on rounding boundaries.
REAL_SHAPE = [100, 92, 84, 69] + [80] * 21 + [0] * 13
BOUNDARY_SHAPE = [50, 55, 45, 52, 53, 51, 50, 49.5]   # mean 50.6875


def test_every_screen_reads_the_same_session_number():
    _fresh_db()
    now = time.time() - 3600
    for sid, shape in (("c_01__real", REAL_SHAPE), ("c_01__edge", BOUNDARY_SHAPE)):
        _seed_run(sid, now, shape)
        now += 600

        report = tl.get_session_focus(sid)["avg_composure"]            # report + coaching
        replay = tl.get_timeline(sid)["stats"]["avg_composure"]        # /replay and /report timeline
        summary = tl.get_dashboard_summary(candidate_id=sid, days=84)  # history + trend
        history = summary["recent_sessions"][0]["avg_composure"]
        trend = summary["composure_trend"][0]["avg_composure"]

        assert report == replay == history == trend, (sid, report, replay, history, trend)
        # What the user actually reads: both screens round the same value.
        assert round(report) == round(history)
    print("PASS every_screen_reads_the_same_session_number")


def test_phone_tail_is_counted_not_hidden():
    """The exact failure: a late 0-tail must pull the report down like history."""
    _fresh_db()
    _seed_run("c_01__real", time.time() - 600, REAL_SHAPE)
    value = tl.get_session_focus("c_01__real")["avg_composure"]
    assert value == 53.3, value
    # The old per-answer reading (taken in the steady stretch) was about 80.
    assert abs(value - 80.0) > 20
    print("PASS phone_tail_is_counted_not_hidden")


def test_run_with_no_frames_has_no_score():
    _fresh_db()
    assert tl.get_session_focus("c_01__none") is None
    print("PASS run_with_no_frames_has_no_score")


def test_other_runs_do_not_leak_in():
    _fresh_db()
    now = time.time() - 3600
    _seed_run("c_01__a", now, [100] * 10)
    _seed_run("c_01__b", now + 600, [20] * 10)
    assert tl.get_session_focus("c_01__a")["avg_composure"] == 100.0
    assert tl.get_session_focus("c_01__b")["avg_composure"] == 20.0
    print("PASS other_runs_do_not_leak_in")


def test_verdict_uses_the_camera_score_when_given():
    engine = ConversationEngine()
    _interview(engine, [90.0, 80.0])           # per-answer readings: mean 85
    result = asyncio.run(engine.generate_final_verdict("abc123", focus_override=53.3))
    assert result["focus_score"] == 53.3, result["focus_score"]
    assert result["focus_source"] == "camera"
    print("PASS verdict_uses_the_camera_score_when_given")


def test_verdict_without_camera_data_is_labelled_turns():
    engine = ConversationEngine()
    _interview(engine, [90.0, 80.0])
    result = asyncio.run(engine.generate_final_verdict("abc123"))
    assert result["focus_score"] == 85.0
    assert result["focus_source"] == "turns"
    print("PASS verdict_without_camera_data_is_labelled_turns")


def test_dashboard_readiness_is_the_average_of_sessions():
    _fresh_db()
    base = time.time() - 5 * 86400
    _seed_run("c_01__a", base, [95] * 10)
    _seed_run("c_01__b", base + 86400, [30] * 10, gazes=["DOWN"] * 10)

    only_a = tl.get_dashboard_summary(candidate_id="c_01__a", days=84)["readiness"]["score"]
    only_b = tl.get_dashboard_summary(candidate_id="c_01__b", days=84)["readiness"]["score"]
    both = tl.get_dashboard_summary(candidate_id="c_01", days=84)["readiness"]

    assert only_a > only_b + 20, (only_a, only_b)             # the two runs really differ
    assert abs(both["score"] - (only_a + only_b) / 2) <= 1, (both, only_a, only_b)
    assert only_b < both["score"] < only_a                    # not just "the latest run"
    # The latest (bad) run dragged the average down, and the delta says so.
    assert both["delta_vs_prev"] < 0, both
    print("PASS dashboard_readiness_is_the_average_of_sessions", both)


if __name__ == "__main__":
    test_every_screen_reads_the_same_session_number()
    test_phone_tail_is_counted_not_hidden()
    test_run_with_no_frames_has_no_score()
    test_other_runs_do_not_leak_in()
    test_verdict_uses_the_camera_score_when_given()
    test_verdict_without_camera_data_is_labelled_turns()
    test_dashboard_readiness_is_the_average_of_sessions()
    if os.path.exists(_TMP_DB):
        os.remove(_TMP_DB)
    print("\nAll focus consistency tests passed.")
