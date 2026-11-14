"""Focus event (tab switch / fullscreen exit) storage tests.

Runs against a throwaway temp SQLite DB via the GUARD_DB_PATH env override, so
it never touches the real telemetry DB. Can be run two ways:

    pytest backend/test_focus_events.py
    python backend/test_focus_events.py     # plain-run fallback, no pytest needed
"""
import os
import tempfile

_TMP_DB = os.path.join(tempfile.gettempdir(), f"guard_focus_test_{os.getpid()}.db")
os.environ["GUARD_DB_PATH"] = _TMP_DB

import importlib
from core_memory import timeline as tl
importlib.reload(tl)  # ensure DB_PATH picks up the env override
from core_memory import focus_events as fe
importlib.reload(fe)  # bind insert_moment from the reloaded timeline module

SESSION = "c_focus__abc123"


def _fresh_db():
    if os.path.exists(_TMP_DB):
        os.remove(_TMP_DB)
    tl.init_timeline_tables()


def _seed_frames(start: float, count: int):
    for i in range(count):
        tl.insert_frame(
            frame_id=f"{SESSION}_{i}",
            session_id=SESSION,
            t=start + i * 2,
            composure=90,
            gaze="STRAIGHT",
            head_pose="HEAD_CENTER",
            faces_detected=1,
            is_talking=False,
        )


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_captions_are_coach_tone():
    assert fe.focus_event_caption("TAB_SWITCH", 8.4) == "Left the interview window for 8s."
    assert fe.focus_event_caption("FULLSCREEN_EXIT", 75) == "Exited fullscreen for 1m 15s."
    assert fe.focus_event_caption("TAB_SWITCH", 120) == "Left the interview window for 2m."
    for kind in fe.FOCUS_EVENT_KINDS:
        text = fe.focus_event_caption(kind, 5).lower()
        for word in ("violation", "cheat", "breach", "caught"):
            assert word not in text, (kind, text)
    print("PASS captions_are_coach_tone")


def test_record_writes_closed_moment_on_timeline():
    _fresh_db()
    start = 1_800_000_000.0
    _seed_frames(start, 10)  # frames span start .. start+18

    event = fe.record_focus_event(SESSION, "TAB_SWITCH", 6.0, now=start + 14)
    assert event["t"] == start + 8
    assert event["t_end"] == start + 14
    assert event["duration_sec"] == 6.0

    timeline = tl.get_timeline(SESSION)
    assert len(timeline["moments"]) == 1, timeline["moments"]
    m = timeline["moments"][0]
    assert m["kind"] == "TAB_SWITCH"
    assert m["type"] == fe.FOCUS_EVENT_TYPE
    assert m["t"] == 8.0                      # rebased on the first frame
    assert m["duration_sec"] == 6.0
    assert m["evidence_url"] is None          # nothing on camera to capture
    assert m["caption"] == "Left the interview window for 6s."
    print("PASS record_writes_closed_moment_on_timeline")


def test_does_not_touch_frames_or_composure():
    _fresh_db()
    start = 1_800_000_000.0
    _seed_frames(start, 5)
    before = tl.get_timeline(SESSION)["frames"]
    fe.record_focus_event(SESSION, "FULLSCREEN_EXIT", 3.0, now=start + 6)
    after = tl.get_timeline(SESSION)["frames"]
    assert before == after
    print("PASS does_not_touch_frames_or_composure")


def test_rejects_bad_input():
    _fresh_db()
    assert _raises(fe.record_focus_event, SESSION, "PHONE", 3.0)
    assert _raises(fe.record_focus_event, "", "TAB_SWITCH", 3.0)
    assert _raises(fe.record_focus_event, SESSION, "TAB_SWITCH", -1.0)
    assert _raises(fe.record_focus_event, SESSION, "TAB_SWITCH", fe.MAX_DURATION_SEC + 1)
    assert tl.get_timeline(SESSION)["moments"] == []
    print("PASS rejects_bad_input")


if __name__ == "__main__":
    test_captions_are_coach_tone()
    test_record_writes_closed_moment_on_timeline()
    test_does_not_touch_frames_or_composure()
    test_rejects_bad_input()
    if os.path.exists(_TMP_DB):
        os.remove(_TMP_DB)
    print("\nAll focus event tests passed.")
