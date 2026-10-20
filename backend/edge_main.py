from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import base64
import uuid
import time
import cv2
import numpy as np
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# The launcher intentionally starts this process from the repository root, so
# dotenv's default current-directory lookup misses the backend configuration.
# Resolve it from this file instead, otherwise the LLM keys and mode are absent
# in the normal Windows startup path.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from ultralytics import YOLO
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from core_memory.bea import bea_engine
import shutil
from core_memory import llm_config
import ollama_model_guide
from core_memory.interviewer import ai_interviewer
from core_memory.resume_gate import resume_gate
from core_memory.conversation_engine import conversation_engine
from core_memory.timeline import (
    init_timeline_tables,
    insert_frame,
    insert_moment,
    close_moment,
    get_timeline,
    get_dashboard_summary,
)
from core_memory.episodes import EpisodeTracker
from core_memory.interrupts import InterruptDirector
from core_memory.voice_engine import transcribe_audio as whisper_transcribe
import voice_engine
import tempfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
EVIDENCE_DIR = os.path.join(BACKEND_DIR, "evidence")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🛡️  G.U.A.R.D. Engine warming up...")
    init_timeline_tables()   # ← MR 1: replaces old init_db()
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    if os.getenv("DISABLE_VOICE_ENGINE", "false").lower() != "true":
        try:
            voice_engine.start_voice_loop()
        except Exception as e:
            logger.warning(f"⚠️  Voice engine failed to start: {e}")
    else:
        logger.info("🔇 Voice engine disabled via DISABLE_VOICE_ENGINE")

    app.state.executor = ThreadPoolExecutor(max_workers=2)
    # Keep object inference separate from transcription so speech processing
    # cannot queue or time out a prop scan. One worker also guarantees that the
    # YOLO predictor is never re-entered concurrently.
    app.state.prop_executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="prop-sweep"
    )
    yield
    app.state.executor.shutdown(wait=False)
    app.state.prop_executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="G.U.A.R.D. Edge Vision Sentry", lifespan=lifespan)

DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    # Any localhost port, because Next picks the next free one when 3000 is
    # taken — a second project on the machine silently moves the dev server to
    # 3007 and every fetch fails CORS with no error the user can see. A regex
    # anchored to loopback keeps that from being a debugging session; it grants
    # nothing to a remote origin, since the server binds to localhost anyway.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"🌐 CORS allowed origins: {ALLOWED_ORIGINS}")

# Serve evidence frames by URL so the frontend can <img src> them
os.makedirs(EVIDENCE_DIR, exist_ok=True)
app.mount("/api/v1/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model_prop = YOLO("yolov8s.pt")
yolo_model_prop.to(device)

# COCO class ids we actually care about.
#   0  person     — a second person in shot
#   67 cell phone — the headline cheat signal
#   73 book       — notes on the desk
# The detector was previously restricted to [0, 67], which quietly made the
# "prohibited item" branch in determine_verdict dead code: it tested for book
# and laptop, neither of which YOLO was ever asked to find.
#
# 63 (laptop) is deliberately NOT on by default. Plenty of candidates sit at a
# desk with a second machine visible, and flagging that as cheating is a false
# accusation. Opt in with BEA_WATCH_LAPTOP=true where the setup warrants it.
YOLO_WATCH_CLASSES = [0, 67, 73]
if os.getenv("BEA_WATCH_LAPTOP", "false").lower() == "true":
    YOLO_WATCH_CLASSES.append(63)

# Verdict floor: a cell-phone box at or above this accuses the candidate.
#
# History, because the number has moved three times and each move looked
# obviously right at the time:
#
#   0.65 -> 0.35   0.65 was a recall problem wearing a latency costume. An
#                  angled or partly-occluded phone scores 0.35-0.55, so those
#                  frames were discarded silently and the sweep only fired once
#                  the phone came closer or steadier — a miss that reads as a
#                  3-5 s delay.
#   0.35 -> 0.30   Chosen to claw back a desk phone believed to sit at 0.310,
#                  just under the gate. That belief came from scoring against
#                  the `moments` table, which is the detector's own output; the
#                  frame in question turned out to be a headset, not a phone.
#   0.30 -> 0.40   Measured against labels.json, where a human has looked at all
#                  25 frames. The scores separate with an empty band in between:
#
#                    real phones   0.453 0.584 0.607 0.627 0.651 0.670 0.685
#                                  0.688 0.730 0.804 0.815 0.834 0.938
#                    no phone      0.155 0.215 0.372 | 0.532 0.606
#
#                  0.30 sat below that gap and 0.40 sits inside it, so the move
#                  costs nothing (13/14 phones either way) and drops one false
#                  accusation: the shadowed edge of a face at 0.372. Precision
#                  81% -> 87%. Full table in bench_phone_recall.json.
#
# 0.40 is NOT a ceiling worth pushing. 0.50 starts losing real phones, and the
# two false positives that survive — a logo at 0.606 and a laptop screen at
# 0.532 — sit above any floor that keeps recall, so they are a class problem,
# not a threshold one. COCO "cell phone" means roughly "lit rectangle in a dark
# bezel", and a monitor genuinely is one. Threshold tuning is finished here.
#
# Object criticals still skip the 3-of-5 debounce (see bea.record_critical_signal).
YOLO_CONF = float(os.getenv("BEA_YOLO_CONF", "0.40"))
# Floor for boxes that are DRAWN but never counted.
#
# The overlay is the only way to check whether a phone accusation is looking at a
# phone, and a box that is missing tells you nothing. Detections between this and
# YOLO_CONF are returned to the frontend marked `fired: false` and drawn dashed:
# "the detector sees something here and is not acting on it". YOLO drops anything
# below the conf it is CALLED with, so this — not YOLO_CONF — is the value passed
# to the model, with YOLO_CONF applied afterwards as the verdict floor.
PHONE_DRAW_CONF = float(os.getenv("BEA_PHONE_DRAW_CONF", "0.15"))
# One frame at the floor is still not proof, so the same phone must appear on two
# consecutive sweeps before escalating. A real phone stays in shot and clears that
# in ~2 sweeps; a one-frame ghost never gets a second.
#
# Note what this canNOT do, learned the hard way: it is a filter against FLICKER,
# not against a persistent wrong answer. An over-ear headset is the most static
# object in the frame, so it satisfies "two consecutive sweeps" indefinitely. That
# is why the fix below is geometric rather than temporal.
PHONE_FULL_CONFIRMATIONS = int(os.getenv("BEA_PHONE_FULL_CONFIRMATIONS", "2"))

logger.info(f"🔫 YOLOv8s loaded on {device.upper()} | classes={YOLO_WATCH_CLASSES} conf={YOLO_CONF}")



# ---------------------------------------------------------------------------
# Evidence helper
# ---------------------------------------------------------------------------
def write_evidence_frame(session_id: str, image_base64: str) -> str | None:
    """Decode a base64 JPEG, write to EVIDENCE_DIR, return the URL path."""
    try:
        raw = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        img_bytes = base64.b64decode(raw)
        filename = f"{session_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(EVIDENCE_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(img_bytes)
        return f"/api/v1/evidence/{filename}"
    except Exception as e:
        logger.error(f"❌ Evidence write error: {e}")
        return None


# ---------------------------------------------------------------------------
# Verdict logic (unchanged)
# ---------------------------------------------------------------------------
def classify_pose(head_pose: str) -> tuple[str, str | None]:
    """Map one fused pose label to (gaze_bucket, narration).

    Split out of determine_verdict so the pose reading survives independently of
    whether some graver condition also fired in the same frame. The previous
    if/elif chain evaluated pose only when nothing else matched, so a frame with
    two faces recorded gaze="STRAIGHT" — the head could be turned 40 degrees and
    the dashboard counted it as focused time (timeline.py:522 keys focus on the
    literal string "STRAIGHT").

    Returns ("STRAIGHT", None) for a centred pose, and for anything unrecognised:
    an unknown label is the absence of a reading, not a deflection.
    """
    if head_pose in ("HEAD_LEFT", "HEAD_RIGHT"):
        direction = head_pose.replace("HEAD_", "").lower()
        return "SIDE_OR_UP", f"head tilted {direction}"
    if head_pose == "HEAD_UP":
        return "SIDE_OR_UP", "head tilted up"
    if head_pose == "HEAD_DOWN":
        return "DOWN", "head tilted down"
    if head_pose in ("GAZE_LEFT", "GAZE_RIGHT"):
        direction = head_pose.replace("GAZE_", "").lower()
        return "SIDE_OR_UP", f"eyes drifted {direction}"
    if head_pose == "GAZE_UP":
        return "SIDE_OR_UP", "eyes drifted up"
    if head_pose == "GAZE_DOWN":
        return "DOWN", "eyes drifted down"
    return "STRAIGHT", None


def determine_verdict(
    detected_objects: list,
    faces: int,
    talking: bool,
    head_pose: str,
    eyes_open: bool = True,
):
    """Returns (gaze, is_critical, critical_kind, verdict, logic_trace, flags).

    Every condition is now tested INDEPENDENTLY and all of them are reported.
    The old chain was `if object / elif faces / elif pose`, which meant the
    frame could only ever name its single gravest finding: a candidate holding a
    phone while turned left was logged as "Mobile device detected" and the turn
    was discarded — not ranked below the phone, but never recorded at all. The
    log box could therefore only ever show one flag, because only one existed.

    `flags` is the ordered list of everything true about this frame, gravest
    first. `verdict` is those flags joined into one sentence, so the string
    contract the rest of the system greps still holds.

    `critical_kind` splits criticals by how much a single frame can be trusted:
      "object"    — a phone/book/laptop is physically in shot. YOLO is restricted
                    to a specific class list, so the CLASS is the evidence, not
                    the score, and one frame is proof. Flag now.

                    The score used to carry that argument at conf>=0.65, but a
                    high floor discarded angled and partly-occluded phones
                    outright — a recall failure that presented as latency. The
                    floor is now YOLO_CONF, set from measured evidence rather
                    than feel (see its definition); a weaker box is a phone seen
                    poorly, not the absence of one. What keeps a single frame
                    trustworthy is the class restriction plus the rising-edge
                    latch, which bills one prop once however many sweeps see it.
      "no_face" / "multiple_faces" — confirmed independently with consecutive
                    samples so one condition cannot count toward the other.

    Only ONE critical_kind is returned even when several criticals are live,
    because it selects a confirmation counter in the BEA and those counters must
    stay separate (one condition must not count toward another's threshold).
    Severity order — object, then no_face, then multiple_faces — is preserved
    from the old chain: a phone in shot outranks the face count because a
    candidate raising a phone usually occludes or turns their face in the same
    moment, and a face-first ordering reported "face not visible" while the
    phone never surfaced.
    """
    flags: list[dict] = []

    if any("cell phone" in obj.lower() for obj in detected_objects):
        flags.append({
            "kind": "MOBILE_DEVICE",
            "critical": True,
            "critical_kind": "object",
            "text": "CRITICAL: Mobile device detected in frame.",
        })
    if any("book" in obj.lower() or "laptop" in obj.lower() for obj in detected_objects):
        flags.append({
            "kind": "PROHIBITED_ITEM",
            "critical": True,
            "critical_kind": "object",
            "text": "CRITICAL: Prohibited item detected on desk.",
        })
    if faces == 0:
        flags.append({
            "kind": "NO_FACE",
            "critical": True,
            "critical_kind": "no_face",
            "text": "CRITICAL: Candidate face not visible or obscured.",
        })
    elif faces > 1:
        flags.append({
            "kind": "MULTIPLE_FACES",
            "critical": True,
            "critical_kind": "multiple_faces",
            "text": "CRITICAL: Multiple persons detected in frame.",
        })

    # Pose is read on every frame, but only trusted when exactly one face is
    # present: with no face there are no landmarks to classify, and with several
    # the fused label describes whichever face MediaPipe happened to rank first.
    gaze = "STRAIGHT"
    if faces == 1:
        if not eyes_open:
            # Eyes shut, and sustained — the client only reports closed after a
            # 1.5s majority, so this is not a blink. A candidate with their eyes
            # closed is not visually engaged with the screen, which is the exact
            # frame that used to read as "fully engaged and attentive": one face,
            # head centred, nothing in shot. It is an attention lapse, not a
            # cheating critical, so it rides the same non-critical DOWN track as a
            # gaze drift — risk accrues only if the closure is held across the
            # BEA's soft/warn/hard thresholds, never on a single sample.
            gaze = "DOWN"
            flags.append({
                "kind": "EYES_CLOSED",
                "critical": False,
                "critical_kind": None,
                "text": "Candidate's eyes are closed — not visually engaged.",
            })
        else:
            gaze, pose_narration = classify_pose(head_pose)
            if pose_narration:
                flags.append({
                    "kind": "ATTENTION_DRIFT",
                    "critical": False,
                    "critical_kind": None,
                    "text": f"Attention drift detected: {pose_narration}.",
                })

    # Talking used to raise a SPEECH flag reading "possible earpiece coaching."
    # It fired on the mouth moving and nothing else — and this path has nothing
    # else to go on: the transcript that could tell coaching from a normal answer
    # lives on the conversation turn (process_candidate_turn), not here. So the
    # flag was a guess dressed as a finding, and in every recorded session it was
    # the candidate answering the question out loud. It never once coincided with
    # a phone or a second person, the two things it was meant to corroborate.
    #
    # `talking` stays in the telemetry (timeline_frames.is_talking) and in the
    # logic trace below as neutral fact; what is gone is the alarm. The honest
    # earpiece signal — an answer that is OFF-TOPIC for the question asked — can
    # only be judged where the transcript is, so it belongs on the turn path.

    criticals = [f for f in flags if f["critical"]]
    is_critical = bool(criticals)
    critical_kind = criticals[0]["critical_kind"] if criticals else None

    if flags:
        verdict = " ".join(f["text"] for f in flags)
    else:
        verdict = "Candidate is fully engaged and attentive."

    logic_trace = (
        f"Objects: {detected_objects or 'None'} | Faces: {faces} "
        f"| Pose: {head_pose} | Talking: {talking} "
        f"| Flags: {', '.join(f['kind'] for f in flags) or 'None'}"
    )
    return gaze, is_critical, critical_kind, verdict, logic_trace, flags


def build_moment_caption(
    verdict: str, is_critical: bool, gaze: str, risk_packet: dict
) -> str:
    """Caption for a flagged moment, written against the reason it was flagged.

    `verdict` narrates ONE frame; `autopsy_flag` fires off accumulated risk
    (>=75%, see bea._level_for). Those are different time horizons, so a frame
    that is clean right now can be flagged for history it did not cause — and
    reusing the frame string there printed "Candidate is fully engaged and
    attentive." next to a HARD_WARNING badge and an evidence photo. Both halves
    were true in isolation; together they read as a contradiction, and the photo
    read as proof of nothing.

    So: when the frame itself is the reason (critical, or eyes/head off-centre),
    the frame string is the honest caption and is kept verbatim — object
    criticals in particular MUST pass through unchanged, because
    _classify_moment_caption recovers MOBILE_DEVICE / PROHIBITED_ITEM by reading
    the object phrasing back out of this text.

    Otherwise the caption names the standing tier and says outright that this
    frame was clean, which is what the attached image actually shows.
    """
    if is_critical or gaze != "STRAIGHT":
        return verdict[:200]

    score = risk_packet.get("risk_score", 0)
    level = risk_packet.get("intervention_level", "WARNING")
    count = risk_packet.get("violation_count", 0)

    earlier = f" from {count} earlier flag{'' if count == 1 else 's'}" if count else ""
    return (
        f"Accumulated risk held at {score}% ({level}){earlier}. "
        "This frame itself was clean — captured for context."
    )[:200]


# ---------------------------------------------------------------------------
# Frame payload model
# ---------------------------------------------------------------------------
class FramePayload(BaseModel):
    candidate_id: str
    timestamp: int
    image_base64: str
    faces_detected: int = 1
    is_talking: bool = False
    head_pose: str = "HEAD_CENTER"
    # The two PRE-FUSION sensor calls behind head_pose. head_pose is the fused
    # label the system acts on; these two say what each sensor believed before
    # the iris veto was applied. Both are recorded but neither is acted on — the
    # decision path is unchanged.
    #
    # They exist because the veto could not be evaluated on a recorded session
    # without them: given only "HEAD_CENTER" you cannot tell a candidate who
    # never moved from one whose head turn was vetoed by centred eyes. Defaults
    # keep older clients that don't send them working.
    head_pose_raw: str = "HEAD_CENTER"
    gaze_class: str = "unknown"
    gaze_vector: list[float] | None = None
    # Optional: frontend may pass the active session_id once sessions are wired
    session_id: str | None = None
    # Whether the candidate's eyes are open, smoothed client-side over a 1.5s
    # window so an ordinary blink never reads as closed. Defaults True so older
    # clients that never send it are treated as eyes-open — no false attention
    # flags on a payload that simply predates this signal.
    eyes_open: bool = True


# Collapses the per-frame flag stream into one record per contiguous stretch.
# Process-wide and in-memory, like _prop_seen below: episode state is only
# meaningful while a session is actually streaming, and a restart mid-session
# loses the open episode's start time either way — nothing here is worth a
# database round trip on every frame.
episode_tracker = EpisodeTracker()


# Decides when to speak up mid-session, and picks the words. Same in-memory
# rationale as episode_tracker: rotation and cooldown state only mean anything
# while a session is streaming.
#
# The audio is NOT played here. voice_engine.speak() is pyttsx3 calling
# engine.runAndWait(), which blocks — inside these handlers it would stall an
# endpoint that runs every 1.2 s — and it plays on whatever machine hosts the
# backend, which on any non-local deploy is not the candidate. So this emits an
# "interrupt" object in the response and the browser speaks it through the
# speechSynthesis path /sentry already owns.
interrupt_director = InterruptDirector()


# ---------------------------------------------------------------------------
# /api/v1/analyze-frame  — MR 1 ingestion rewired to timeline_frames
# ---------------------------------------------------------------------------
@app.post("/api/v1/analyze-frame")
async def analyze_frame(payload: FramePayload, background_tasks: BackgroundTasks):
    print(
        f"\n📥 [BACKEND] INCOMING PAYLOAD → "
        f"Faces: {payload.faces_detected}, Pose: '{payload.head_pose}', Talking: {payload.is_talking}"
    )

    # Two different grains, deliberately kept separate:
    #   candidate_id -> BEA identity (lockout//status/reset all key on this)
    #   session_id   -> analytics grain, one row per practice run (dashboard trends)
    # Falls back to candidate_id so older clients that don't send one still work.
    session_id = payload.session_id or payload.candidate_id
    bea_key = payload.candidate_id

    # NOTE: deliberately no early-return on a prior SEVERE_VIOLATION_LOGGED.
    # This used to bail out here, which meant the first confirmed flag stopped
    # all further analysis: no more frames written, no gaze tracking, and the
    # final report showed nothing after that instant. A flag is recorded and
    # surfaced, but the session keeps running so the candidate still gets a
    # complete timeline and verdict.


    # --- PHASE 1: Decode image ---
    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Image decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    # Object inference has its own rising-edge endpoint and cadence. Running the
    # same CPU-heavy YOLO model here delayed gaze/face telemetry by several
    # seconds and duplicated every object scan.
    detected_objects = []

    # --- PHASE 3: Deterministic verdict ---
    gaze, is_critical, critical_kind, verdict, logic_trace, flags = determine_verdict(
        detected_objects=detected_objects,
        faces=payload.faces_detected,
        talking=payload.is_talking,
        head_pose=payload.head_pose,
        eyes_open=payload.eyes_open,
    )

    # --- PHASE 4: BEA temporal graph ---
    # Objects are owned by the prop sweep and flag on the rising edge. This path
    # handles face, speech, and pose signals without waiting on object inference.
    interrupt: dict | None = None
    if is_critical and critical_kind == "object" and _prop_seen.get(bea_key, False):
        # Prop already flagged by the fast sweep — skip escalation so we don't
        # bill it twice, but still write the frame so the verdict page has the
        # visual. Record as neutral telemetry, not as a new violation.
        risk_packet = await bea_engine.get_state(bea_key)
    elif is_critical:
        # A visible object is proof on sight; ambiguous signals keep the debounce.
        instant = critical_kind == "object"
        if instant:
            # Latch it so the sweep doesn't immediately re-flag the same prop,
            # and clear any part-built streak so a stale count can't unlatch it
            # on the very next clean sweep.
            _prop_seen[bea_key] = True
            _prop_clear_streak[bea_key] = 0
        decision = await bea_engine.record_critical_signal(
            bea_key, True, verdict, instant=instant, signal_type=critical_kind
        )

        if decision["confirmed"]:
            consolidated = "; ".join(decision["pending_reasons"]) or verdict
            # record_violation, not trigger_fatal_lockout: the flag must be logged
            # without latching the session shut for every subsequent frame.
            risk_packet = await bea_engine.record_violation(bea_key, reason=consolidated)
            verdict = (
                f"CONFIRMED: {consolidated}"
                if decision.get("instant")
                else f"CONFIRMED ({decision['count']}/{decision['threshold']}): {consolidated}"
            )
            # Speak up, but only here — this branch is the rising edge of a
            # CONFIRMED incident, which is the first moment the system is
            # entitled to assert the finding out loud. The `active_confirmed`
            # branch below is the same incident continuing, so interrupting from
            # there would repeat the reminder every 2 s for as long as the second
            # person stayed in shot.
            #
            # Objects are excluded deliberately: they are confirmed on the prop
            # sweep, which owns their interruption. Reaching this branch with an
            # object means the sweep had not yet latched it, and having both
            # paths speak would say it twice for one phone.
            if critical_kind == "multiple_faces":
                interrupt = interrupt_director.consider(session_id, "MULTIPLE_FACES")
        elif decision.get("active_confirmed"):
            # One uninterrupted face incident gets one violation and one proof
            # frame. Keep the standing risk visible without writing duplicates
            # on every subsequent one-second sample.
            risk_packet = await bea_engine.get_state(bea_key)
            risk_packet["autopsy_flag"] = False
            verdict = f"CONFIRMED (ongoing): {verdict}"
        else:
            # Pass the pose that was actually measured. This line used to hand the
            # accumulator a hardcoded "SIDE_OR_UP", so for as long as a critical sat
            # pending confirmation the candidate accrued side-look risk no matter
            # where they were looking — and the SIDE timer kept running, so a
            # flickering no-face or multiple-faces read could walk a motionless,
            # forward-facing candidate through the 5/8/12 s tiers on its own.
            #
            # Only meaningful because classify_pose now runs on every frame: while
            # pose was evaluated in the tail of an if/elif chain, a critical frame
            # always reported gaze="STRAIGHT", so there was no real reading here to
            # pass and the hardcoded value was hiding that rather than causing it.
            risk_packet = await bea_engine.record_telemetry(bea_key, gaze)
            risk_packet["critical_pending"] = decision["count"]
            risk_packet["critical_threshold"] = decision["threshold"]
            verdict = f"⚠️ ANOMALY {decision['count']}/{decision['threshold']}: {verdict}"
    else:
        await bea_engine.record_critical_signal(bea_key, False, "")
        risk_packet = await bea_engine.record_telemetry(bea_key, gaze)

    # --- PHASE 5: MR 1 — persist to timeline_frames ---
    risk_score: int = risk_packet.get("risk_score", 0)
    composure: float = 100.0 - risk_score          # ← composure = 100 - risk
    autopsy_flag: bool = risk_packet.get("autopsy_flag", False)

    frame_id = uuid.uuid4().hex
    t_now = time.time()

    background_tasks.add_task(
        insert_frame,
        frame_id,
        session_id,
        t_now,
        composure,
        gaze,
        payload.head_pose,
        payload.faces_detected,
        payload.is_talking,
        payload.head_pose_raw,
        payload.gaze_class,
    )

    # Write evidence frame and log as a moment when BEA flags it.
    #
    # Episode grain, not frame grain. autopsy_flag is true on EVERY frame while
    # risk sits at or above 75 and telemetry samples every 2 s, so this block used
    # to write a row and a JPEG roughly ten times for one continuous look-away —
    # ten photographs of the same person in the same position, each shown to them
    # as a separate finding. The tracker collapses those into one row per flag
    # kind with a real start, end and duration, and takes exactly one photo: the
    # frame that opened the episode, which is the frame that caused the flag.
    #
    # observe() is called on EVERY frame, not only flagged ones, because an
    # episode needs a frame without the behaviour in order to close.
    active_kinds = [f["kind"] for f in flags] if autopsy_flag else []
    opened, closed = episode_tracker.observe(session_id, t_now, active_kinds)

    if opened:
        # One image for the whole episode, captured now while we still hold the
        # frame. Several kinds opening on the same frame share it — it is one
        # photograph of one instant, and duplicating the file per kind would put
        # more copies of a participant's face on disk to say the same thing.
        evidence_url = write_evidence_frame(session_id, payload.image_base64)
        moment_caption = build_moment_caption(verdict, is_critical, gaze, risk_packet)
        by_kind = {f["kind"]: f for f in flags}
        for episode in opened:
            flag = by_kind.get(episode["kind"])
            # Caption per kind so a row says why IT fired, rather than repeating
            # the whole multi-flag sentence on every row.
            caption = flag["text"] if flag else moment_caption
            background_tasks.add_task(
                insert_moment,
                episode["moment_id"],
                session_id,
                episode["t_start"],
                risk_packet.get("intervention_level", "WARNING"),
                caption[:200],
                evidence_url,
                episode["kind"],
            )

    for episode in closed:
        background_tasks.add_task(
            close_moment,
            episode["moment_id"],
            episode["t_end"],
            episode["duration_sec"],
            episode["frame_count"],
        )

    background_tasks.add_task(bea_engine.cleanup_stale_sessions)

    logger.info(f"👁️  VERDICT: {verdict}")
    logger.info(f"🧠  TRACE:   {logic_trace}")
    logger.info(f"🎯  GAZE: {gaze} | COMPOSURE: {composure:.0f}% | [{risk_packet['intervention_level']}]")
    logger.info("-" * 50)

    return {
        "candidate_id": payload.candidate_id,
        "session_id": session_id,
        "timestamp": payload.timestamp,
        "verdict": verdict,
        "gaze": gaze,
        # Every condition true in this frame, gravest first. The log box renders
        # this list; `verdict` remains the single-string form for callers that
        # only want one line (and for the caption greps).
        "flags": [
            {"kind": f["kind"], "text": f["text"], "critical": f["critical"]}
            for f in flags
        ],
        # Present only on the frame a confirmed second person first appears; null
        # otherwise. The frontend speaks it, so a non-null value here is a
        # one-shot instruction rather than state to render every frame.
        "interrupt": interrupt,
        "risk_packet": risk_packet,
    }


# ---------------------------------------------------------------------------
# /api/v1/scan-objects — fast prop sweep, decoupled from the telemetry loop
# ---------------------------------------------------------------------------
class ObjectScanPayload(BaseModel):
    candidate_id: str
    image_base64: str
    session_id: str | None = None


# Person (0) is deliberately excluded: head-count comes from MediaPipe in the
# main loop, and this sweep only answers "is a prop visible". Derived from
# YOLO_WATCH_CLASSES so the BEA_WATCH_LAPTOP opt-in is picked up for free.
YOLO_PROP_CLASSES = [c for c in YOLO_WATCH_CLASSES if c != 0]

# Rising-edge memory per candidate. A phone left on the desk for 20s is ONE
# incident, not 16 — without this the fast cadence would multiply a single
# event by the scan rate and drive risk to 100% in seconds.
_prop_seen: dict[str, bool] = {}

# Consecutive clean sweeps per candidate, used to clear the latch above.
#
# The latch previously cleared on a SINGLE clean sweep, which turned one phone
# into several incidents: detected (latch set) -> one sweep where a hand, a turn,
# or a soft confidence score hides it (latch cleared) -> visible again (billed a
# second time). Requiring a RUN of clean sweeps makes "the prop is gone" need
# more evidence than "the prop is here", which is the correct asymmetry for
# physical evidence and the same reasoning that lets objects skip the 3-of-5.
#
# This matters more at the lowered YOLO_CONF: a weaker box is now kept rather
# than discarded, so scores sit closer to the floor and cross it more often.
#
# At PROP_SCAN_MS = 1200 (SniperScope.tsx), 3 sweeps is ~3.6 s of a genuinely
# empty frame before the same phone can open a second violation.
PROP_CLEAR_SWEEPS = int(os.getenv("BEA_PROP_CLEAR_SWEEPS", "3"))
_prop_clear_streak: dict[str, int] = {}
_phone_full_streak: dict[str, int] = {}

# A SECOND tracker, deliberately not the telemetry one.
#
# _close_expired closes every kind absent from the frame it is given, so a single
# shared tracker would have the 1.2 s object sweep — which never reports poses or
# face counts — expiring the 2 s telemetry loop's ATTENTION_DRIFT episodes, and
# vice versa. Each loop only knows its own vocabulary, so each gets its own book.
#
# The gap matches the risk latch above (PROP_CLEAR_SWEEPS sweeps of PROP_SCAN_MS)
# rather than the telemetry default, so an episode boundary and a re-billable
# incident are the same event. With a shorter gap the evidence page would show a
# second sighting the risk engine had not charged for.
prop_episode_tracker = EpisodeTracker(gap_sec=PROP_CLEAR_SWEEPS * 1.2)


def _detect_objects(results, names: dict, min_conf: float = 0.0) -> list[str]:
    """Return normalized labels from one or more YOLO result batches."""
    detected = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < min_conf:
                continue
            label = names[int(box.cls[0])]
            detected.append(f"{label} ({confidence:.0%})")
    return detected


def _detect_boxes(
    results,
    names: dict,
    min_conf: float,
    *,
    source: str,
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    """Return detection boxes in normalized frame coordinates, for drawing.

    Exists alongside _detect_objects because the verdict only needs labels, but a
    human checking the verdict needs to see what was boxed. "cell phone (51%)"
    reads identically whether YOLO found a phone or an over-ear headphone cup, so
    a phone accusation with no box on screen is unfalsifiable from the UI. Drawing
    the box is what turned "it says phone and there is no phone" from an argument
    into a measurement.

    Normalized rather than pixel coords so the overlay scales to whatever the
    video element is showing, instead of assuming the 640x360 the sweep sends.

    `fired` records whether the box cleared YOLO_CONF and so counted toward the
    verdict. Boxes between PHONE_DRAW_CONF and YOLO_CONF are still returned, so a
    near miss shows up as a near miss instead of as nothing at all. The model is
    called at PHONE_DRAW_CONF, which is therefore the hard visibility limit here:
    anything below it was dropped inside YOLO and cannot be drawn.
    """
    boxes = []
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes.append(
                {
                    "label": names[int(box.cls[0])],
                    "conf": round(confidence, 3),
                    "source": source,
                    "fired": confidence >= min_conf,
                    "box": [
                        round(x1 / frame_width, 4),
                        round(y1 / frame_height, 4),
                        round(x2 / frame_width, 4),
                        round(y2 / frame_height, 4),
                    ],
                }
            )
    return boxes


@app.post("/api/v1/scan-objects")
async def scan_objects(payload: ObjectScanPayload, background_tasks: BackgroundTasks):
    """YOLO-only sweep that runs several times faster than the telemetry loop.

    This remains separate because object detection has a different cadence and
    rising-edge contract from full telemetry. It writes no timeline frames.

    So this path writes NO timeline_frames. It answers one question — is a prop
    visible right now — and escalates on the rising edge if so.
    """
    session_id = payload.session_id or payload.candidate_id
    bea_key = payload.candidate_id

    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(raw_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"❌ Prop scan decode error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    loop = asyncio.get_running_loop()
    # ONE full-frame pass. There used to be two extra magnified crops of the lower
    # field, on the theory that a phone on the desk occupies too few pixels to
    # score well at full frame. Measured over 25 evidence frames that a human has
    # since labelled one by one (labels.json — 14 real phones, 11 with no phone),
    # that theory cost far more than it bought:
    #
    #   config                          real phones    false accusations    ms
    #   full + lower tiles @0.20          14 of 14        11 of 11         383
    #   full + lower tiles @0.35          14 of 14         6 of 11         383
    #   full frame only    @0.40          13 of 14         2 of 11          76
    #
    # At the floor the tile pass actually shipped with, it accused every single
    # candidate who was not holding a phone. All eleven. Upscaling a 400x240 crop
    # 1.6x does not just magnify a phone, it magnifies an over-ear headphone cup
    # and the shadowed edge of a face into exactly the dark rounded rectangle
    # COCO's "cell phone" class was trained on, lifting them from ~0.2 (correctly
    # ignored at full frame) into the 0.41-0.49 band where real phones live.
    #
    # Thirteen further geometries were swept — narrower tiles, 2x2 grids, lower
    # thirds, imgsz 960, CLAHE, 2x pre-upscale, yolov8n. None beat one full-frame
    # pass on the recall/false-accusation trade, and the six that tie it cost 3-9x
    # the compute for a bit-identical answer (bench_phone_recall.json).
    #
    # Two of those results are worth knowing before reaching for the obvious idea:
    #   - Pre-upscaling 2x is a NO-OP. It measures identically to plain full frame
    #     because LetterBox scales whatever you send back down to imgsz, so what
    #     the model sees is the object's size AFTER that scaling, never the input
    #     resolution. Every "just upscale it first" variant is this same no-op.
    #   - imgsz=960 is actively worse (10 of 14, 4 of 11). Interpolating detail
    #     that was destroyed at capture does not restore it.
    #
    # The one phone this loses scores 0.000 here — no box at all, so no floor
    # anywhere recovers it. Magnification DOES reach it (the tile pass finds 14 of
    # 14), so this is not purely a resolution wall; it is just not worth 11 false
    # accusations and 5x the latency. The honest fix is real pixels: the sweep
    # downscales a 1280x720 capture to 640x360 before sending, and this phone does
    # not survive that. A fine-tune that knows a headset from a handset would also
    # do it. Both are worth doing; neither is a reason to keep accusing candidates
    # of holding their own headphones.
    height, width = image.shape[:2]
    full_results = await loop.run_in_executor(
        app.state.prop_executor,
        lambda: yolo_model_prop(
            image,
            verbose=False,
            classes=YOLO_PROP_CLASSES,
            imgsz=640,
            augment=False,
            conf=PHONE_DRAW_CONF,
        ),
    )
    detected_objects = _detect_objects(
        full_results, yolo_model_prop.names, min_conf=YOLO_CONF
    )
    full_phone_detected = any("cell phone" in o.lower() for o in detected_objects)

    # Every box the model returned, including those below YOLO_CONF, so the
    # overlay can show a near miss as a near miss instead of as nothing.
    detection_boxes = _detect_boxes(
        full_results,
        yolo_model_prop.names,
        YOLO_CONF,
        source="full",
        frame_width=width,
        frame_height=height,
    )
    # The same phone must appear on two consecutive sweeps before escalating. The
    # streak resets to 0 on any clean sweep, so the two sightings must be genuinely
    # consecutive rather than merely two in a run.
    full_streak = _phone_full_streak.get(bea_key, 0)
    _phone_full_streak[bea_key] = full_streak + 1 if full_phone_detected else 0

    if _phone_full_streak[bea_key] >= PHONE_FULL_CONFIRMATIONS:
        verdict = "CRITICAL: Mobile device detected in frame."
    elif any("book" in o.lower() or "laptop" in o.lower() for o in detected_objects):
        verdict = "CRITICAL: Prohibited item detected on desk."
    else:
        verdict = None

    was_seen = _prop_seen.get(bea_key, False)
    if verdict is not None:
        _prop_seen[bea_key] = True
        _prop_clear_streak[bea_key] = 0
    else:
        # Hysteresis: one clean sweep is a dropout, not a departure. Only a
        # sustained run of them re-arms the latch.
        streak = _prop_clear_streak.get(bea_key, 0) + 1
        _prop_clear_streak[bea_key] = streak
        if streak >= PROP_CLEAR_SWEEPS:
            _prop_seen[bea_key] = False

    # Episode bookkeeping runs on EVERY sweep, including clean ones — an episode
    # needs a sweep without the prop in order to end. The moment row is still only
    # written on the gated path below, so an episode the risk engine declined to
    # bill simply has no row; close_moment on an id that was never inserted
    # updates nothing, which is the behaviour we want.
    prop_kind = None
    if verdict is not None:
        prop_kind = (
            "MOBILE_DEVICE" if "mobile device" in verdict.lower() else "PROHIBITED_ITEM"
        )
    t_sweep = time.time()
    prop_opened, prop_closed = prop_episode_tracker.observe(
        session_id, t_sweep, [prop_kind] if prop_kind else []
    )
    for episode in prop_closed:
        background_tasks.add_task(
            close_moment,
            episode["moment_id"],
            episode["t_end"],
            episode["duration_sec"],
            episode["frame_count"],
        )

    # Nothing there, or the same prop we already flagged — stop here.
    #
    # Deliberately NOT calling record_critical_signal(False) on the clear path:
    # that resets confirmation state owned by the telemetry loop. Clearing from
    # here could wipe a face-event streak before it fills; this loop only
    # ever adds an instant object critical.
    if verdict is None or was_seen:
        return {
            "detected": verdict is not None,
            "escalated": False,
            "objects": detected_objects,
            # Sent on the quiet path too. This is the sweep that runs while a
            # phone banner is already up (the `was_seen` latch) and the sweep that
            # runs when nothing is found at all — the two cases where "what is it
            # boxing?" is the actual question being asked.
            "boxes": detection_boxes,
        }

    logger.info(f"🔫 PROP SWEEP: {detected_objects} → escalating")

    decision = await bea_engine.record_critical_signal(
        bea_key, True, verdict, instant=True
    )
    if not decision["confirmed"]:
        return {
            "detected": True,
            "escalated": False,
            "objects": detected_objects,
            "boxes": detection_boxes,
        }

    consolidated = "; ".join(decision["pending_reasons"]) or verdict
    risk_packet = await bea_engine.record_violation(bea_key, reason=consolidated)
    verdict_text = f"CONFIRMED: {consolidated}"

    # Same evidence trail as the main loop, so the prop still shows up on the
    # verdict page with the frame that caught it. The moment_id comes from the
    # episode rather than a fresh uuid, so the closing sweep can fill in the end
    # time and duration on THIS row.
    if risk_packet.get("autopsy_flag", False):
        evidence_url = write_evidence_frame(session_id, payload.image_base64)
        episode = prop_opened[0] if prop_opened else None
        background_tasks.add_task(
            insert_moment,
            episode["moment_id"] if episode else uuid.uuid4().hex,
            session_id,
            episode["t_start"] if episode else t_sweep,
            risk_packet.get("intervention_level", "WARNING"),
            verdict_text[:200],
            evidence_url,
            prop_kind,
        )

    return {
        "detected": True,
        "escalated": True,
        "objects": detected_objects,
        "boxes": detection_boxes,
        "verdict": verdict_text,
        # Reached only past the `was_seen` latch and the confirmation gate above,
        # so this is the first confirmed sweep of a newly-appeared prop — the
        # rising edge, once per sighting.
        #
        # prop_kind is passed straight through rather than filtered to
        # MOBILE_DEVICE here: PROHIBITED_ITEM has no entry in interrupts.LINES
        # and so returns None on its own. A book or a laptop is not worth
        # interrupting someone over — plenty of people sit at a desk with a
        # second laptop — and keeping that decision in the copy table means the
        # set of findings that may speak is stated in exactly one place.
        "interrupt": interrupt_director.consider(session_id, prop_kind),
        "risk_packet": risk_packet,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/session/{session_id}/timeline  — MR 1 read endpoint
# ---------------------------------------------------------------------------

@app.get("/api/v1/session/{session_id}/timeline")
async def get_session_timeline(session_id: str):
    """Return all timeline_frames and moments for a session."""
    try:
        return get_timeline(session_id)
    except Exception as e:
        logger.error(f"❌ Timeline read error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve timeline.")


# ---------------------------------------------------------------------------
# GET /api/v1/dashboard/summary  — MR 3: cross-session practice dashboard
# ---------------------------------------------------------------------------
@app.get("/api/v1/dashboard/summary")
async def get_dashboard_summary_endpoint(candidate_id: str | None = None, days: int = 84):
    """Cross-session aggregates for the MR-3 practice dashboard.

    Returns the full contract (candidate, totals, readiness, metrics,
    composure_trend, activity, recent_sessions, focus_area). A brand-new
    candidate with zero sessions gets the same shape with zeros/nulls — never
    a 404.
    """
    try:
        return get_dashboard_summary(candidate_id=candidate_id, days=days)
    except Exception as e:
        logger.error(f"❌ Dashboard summary error: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute dashboard summary.")


# ---------------------------------------------------------------------------
# Voice status
# ---------------------------------------------------------------------------
@app.get("/api/v1/voice-status")
async def get_voice_status():
    return voice_engine.voice_state


# ---------------------------------------------------------------------------
# LLM backend — read and switch which brain answers
# ---------------------------------------------------------------------------
# Backs the /choose-model page. Without these, changing backend meant editing
# backend/.env and restarting Python, so the setting was invisible to anyone
# who was not already reading the source.
#
# These write to disk and hold a credential, and this server has no auth yet.
# That is tolerable only because it binds to localhost — see ALLOWED_ORIGINS.
# If this ever listens on 0.0.0.0, these two routes need a guard first.
class LlmModePayload(BaseModel):
    mode: str
    provider: str | None = None


class LlmKeyPayload(BaseModel):
    provider: str
    key: str


class OllamaModelPayload(BaseModel):
    tag: str


@app.get("/api/v1/llm/models")
async def get_llm_models():
    """What this machine can run, and what it already has.

    Separate from /llm/mode because it is slow relative to it: nvidia-smi is a
    subprocess and /api/tags is a second HTTP hop, and the chooser polls mode on
    every visit. The picker is opened deliberately, so it can pay that cost.
    """
    return ollama_model_guide.report()


@app.post("/api/v1/llm/model")
async def set_llm_model(payload: OllamaModelPayload):
    try:
        llm_config.set_ollama_model(payload.tag)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return llm_config.status()


@app.get("/api/v1/llm/mode")
async def get_llm_mode():
    return llm_config.status()


@app.post("/api/v1/llm/mode")
async def set_llm_mode(payload: LlmModePayload):
    try:
        llm_config.set_mode(payload.mode, payload.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return llm_config.status()


@app.post("/api/v1/llm/key")
async def set_llm_key(payload: LlmKeyPayload):
    try:
        masked = llm_config.set_api_key(payload.provider, payload.key)
    except ValueError as exc:
        # 400, not 500: every failure here is the pasted value being wrong,
        # and the message is written to be shown to the user verbatim.
        raise HTTPException(status_code=400, detail=str(exc))
    return {"masked": masked, **llm_config.status()}


# ---------------------------------------------------------------------------
# Session breakdown (kept for verdict page compatibility)
# ---------------------------------------------------------------------------
# Predicates take a whole telemetry-frame row so we can bucket everything the
# stored schema actually carries — not just gaze. faces_detected and is_talking
# are persisted per frame, so NO_FACE / MULTIPLE_FACES / TALKING are all
# reconstructable here. MOBILE_DEVICE / PROHIBITED_ITEM stay False because YOLO
# object labels aren't persisted per frame; those are derived from the flagged
# moments' captions instead (see _classify_moment_caption below).
VIOLATION_TYPES = {
    "DOWN_GAZE":       lambda row: (row.get("gaze") or "") == "DOWN",
    "SIDE_GAZE":       lambda row: (row.get("gaze") or "") == "SIDE_OR_UP",
    "MOBILE_DEVICE":   lambda row: False,
    "PROHIBITED_ITEM": lambda row: False,
    "MULTIPLE_FACES":  lambda row: (row.get("faces_detected") or 0) >= 2,
    "NO_FACE":         lambda row: (row.get("faces_detected") or 0) == 0,
    "TALKING":         lambda row: bool(row.get("is_talking")),
}

# Types that only the moments table can supply — YOLO labels are not stored per
# frame. Every other type is derived from timeline_frames above, so counting it
# from moments as well would bill the same behaviour twice.
_MOMENT_ONLY_TYPES = ("MOBILE_DEVICE", "PROHIBITED_ITEM")


def _classify_moment_caption(caption: str) -> str | None:
    """Map a flagged moment's verdict text to an object-violation type.

    YOLO labels aren't stored per frame, so phone/book criticals can't be
    rebuilt from timeline_frames. Their verdict text is preserved in the
    moment caption, which is enough to attribute the captured JPEG.

    The phrase list must match what determine_verdict and sweep_for_props
    actually emit, not what the YOLO class is called. Both writers say
    "Prohibited item detected on desk." for a book or a second laptop — the
    words 'book' and 'laptop' never survive into the caption, so matching only
    on those meant every book/laptop sighting classified as None and
    PROHIBITED_ITEM was unreachable from a caption. The raw-label spellings are
    kept too: they still appear in rows written before this was fixed, and in
    consolidated multi-reason strings.
    """
    c = (caption or "").lower()
    if "mobile" in c or "phone" in c:
        return "MOBILE_DEVICE"
    if "prohibited item" in c or "book" in c or "laptop" in c:
        return "PROHIBITED_ITEM"
    return None


# Seconds of real time each analysed frame stands for, used only to turn a frame
# COUNT into an approximate duration on the verdict page.
#
# This must track TELEMETRY_INTERVAL_MS in frontend/src/components/SniperScope.tsx.
# It was left at 5 after that loop was sped up to 2000 ms, so every duration shown
# to a candidate was inflated 2.5x — a 12-second glance away was reported back to
# them as half a minute. Overstating someone's behaviour in a report they are
# meant to trust is worse than saying nothing, so it is an env var now and the
# two numbers are named in each other's comments.
INFERENCE_CADENCE_SEC = float(os.getenv("GUARD_INFERENCE_CADENCE_SEC", "2"))

VIOLATION_LABELS = {
    "DOWN_GAZE":  "looking down (off-screen / at lap)",
    "SIDE_GAZE":  "looking sideways or up (off-screen)",
}


def _compute_session_breakdown(session_id: str, since_iso: str | None = None) -> dict:
    """Derive a violation breakdown from timeline_frames.

    Accepts either a candidate_id or a concrete per-run session_id: frames are
    now stored under "{candidate_id}__{rand}", so an exact match alone would
    miss everything. Prefix-matching keeps the verdict page working across
    both the old (session_id == candidate_id) and new per-run schemes.
    """
    import contextlib
    from core_memory.timeline import _db
    from datetime import datetime

    with contextlib.closing(_db()) as conn:
        conn.row_factory = __import__("sqlite3").Row
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM timeline_frames "
                "WHERE (session_id = ? OR session_id LIKE ?) ORDER BY t",
                (session_id, f"{session_id}__%"),
            )
        ]
        # Flagged moments carry the captured JPEG (evidence_url) and the verdict
        # text (caption). timeline_frames has no image column, so this is the
        # only place the proof lives — the breakdown must read it to have any
        # frame to show. Keyed by rounded timestamp so we can attribute each
        # captured image back to the frame that triggered it.
        moment_rows = [
            dict(r)
            for r in conn.execute(
                "SELECT t, type, caption, evidence_url, kind, duration_sec FROM moments "
                "WHERE (session_id = ? OR session_id LIKE ?) ORDER BY t",
                (session_id, f"{session_id}__%"),
            )
        ]

    # Optional time-scope filter
    if since_iso:
        try:
            since_ts = datetime.fromisoformat(since_iso.replace("Z", "+00:00")).timestamp()
            rows = [r for r in rows if r["t"] >= since_ts]
            moment_rows = [m for m in moment_rows if m["t"] >= since_ts]
        except Exception as e:
            logger.warning(f"⚠️ Could not parse since_iso '{since_iso}': {e}")

    # Index captured frames by whole-second timestamp. insert_frame and
    # insert_moment are handed the same t_now in analyze_frame, so a moment's
    # image lines up with the frame that produced it at second granularity.
    evidence_by_sec: dict[int, list[str]] = {}
    for m in moment_rows:
        url = m.get("evidence_url")
        if url:
            evidence_by_sec.setdefault(int(m["t"]), []).append(url)

    ALL_TYPES = tuple(VIOLATION_TYPES.keys())
    buckets: dict[str, dict] = {
        t: {"count": 0, "first_at": None, "last_at": None,
            "peak_risk": 0, "peak_intervention_level": "CLEAR",
            # Measured wall-clock seconds, summed from episode durations. Stays
            # 0 for types rebuilt from frame counts, which have no measured span.
            "measured_seconds": 0.0,
            "evidence_paths": [], "sample_events": []}
        for t in ALL_TYPES
    }
    peak_event = None

    for row in rows:
        composure = row.get("composure", 100)
        risk = int(100 - composure)
        ts = row["t"]
        frame_evidence = evidence_by_sec.get(int(ts), [])

        if peak_event is None or risk > peak_event["risk_score"]:
            peak_event = {"timestamp": ts, "risk_score": risk,
                          "intervention_level": "", "logic_trace": ""}

        for vtype, predicate in VIOLATION_TYPES.items():
            if not predicate(row):
                continue
            b = buckets[vtype]
            b["count"] += 1
            if b["first_at"] is None:
                b["first_at"] = ts
            b["last_at"] = ts
            if risk > b["peak_risk"]:
                b["peak_risk"] = risk
            # Cap stored proof at 6 per pattern — enough to browse, not so many
            # the picker overflows. Only frames that actually captured evidence
            # (i.e. BEA flagged them) contribute an image.
            for url in frame_evidence:
                if url not in b["evidence_paths"] and len(b["evidence_paths"]) < 6:
                    b["evidence_paths"].append(url)
            if len(b["sample_events"]) < 5:
                b["sample_events"].append({"timestamp": ts, "risk_score": risk})

    # Object criticals (phone / book) can't be rebuilt from timeline_frames —
    # YOLO labels aren't stored per frame. Recover them straight from the
    # flagged moments' captions so their captured images still surface.
    #
    # Each row is now one EPISODE, not one frame, so count is a count of separate
    # sightings and duration_sec is the measured length of each. Multiplying this
    # count by the frame cadence would report a 30-second phone as 2 seconds, so
    # measured time is summed instead and wins below wherever it exists.
    for m in moment_rows:
        # Prefer the stored kind — it is the flag identity the detector actually
        # produced. Reading it back out of English prose was only ever a
        # workaround for not having this column, and it stays as the fallback for
        # rows written before the column existed.
        #
        # Restricted to the object types on purpose: every other kind is already
        # counted from timeline_frames by the predicates above, so admitting them
        # here would bill the same behaviour twice.
        vtype = m.get("kind") if m.get("kind") in _MOMENT_ONLY_TYPES else None
        if vtype is None:
            vtype = _classify_moment_caption(m.get("caption", ""))
        if not vtype:
            continue
        b = buckets[vtype]
        b["count"] += 1
        ts = m["t"]
        if b["first_at"] is None:
            b["first_at"] = ts
        b["last_at"] = ts
        # NULL on rows written before episodes existed, and on an episode that was
        # never closed. Both mean "length unknown" — treat the row as one cadence
        # tick rather than inventing a span for it.
        span = m.get("duration_sec")
        b["measured_seconds"] += float(span) if span else INFERENCE_CADENCE_SEC
        url = m.get("evidence_url")
        if url and url not in b["evidence_paths"] and len(b["evidence_paths"]) < 6:
            b["evidence_paths"].append(url)

    violations_by_type = {}
    for vtype, b in buckets.items():
        if b["count"] == 0:
            continue
        measured = b.pop("measured_seconds")
        # Measured episode time when we have it; frame count x cadence otherwise.
        b["approx_total_seconds"] = int(round(measured or b["count"] * INFERENCE_CADENCE_SEC))
        violations_by_type[vtype] = b


    return {
        "candidate_id": session_id,
        "total_events": len(rows),
        "violations_by_type": violations_by_type,
        "peak_event": peak_event,
        "session_window": {
            "first_event_at": rows[0]["t"] if rows else None,
            "last_event_at": rows[-1]["t"] if rows else None,
        },
    }


@app.get("/api/v1/session-breakdown/{candidate_id}")
async def get_session_breakdown(candidate_id: str, since: str | None = None):
    try:
        return _compute_session_breakdown(candidate_id, since_iso=since)
    except Exception as e:
        logger.error(f"❌ Session breakdown error: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute session breakdown.")


# ---------------------------------------------------------------------------
# Verdict report (LLM coaching)
# ---------------------------------------------------------------------------
VIOLATION_LABELS_FULL = {
    "DOWN_GAZE": "looking down (off-screen / at lap)",
    "SIDE_GAZE": "looking sideways or up (off-screen)",
}


def _format_breakdown_for_prompt(breakdown: dict) -> str:
    by_type = breakdown.get("violations_by_type") or {}
    if not by_type:
        return "No behavioural violations were logged during this session."
    lines = []
    for vtype, b in sorted(by_type.items(), key=lambda kv: -kv[1]["peak_risk"]):
        label = VIOLATION_LABELS_FULL.get(vtype, vtype)
        lines.append(
            f"- {label}: {b['count']} event(s), ~{b['approx_total_seconds']}s cumulative, "
            f"peak risk {b['peak_risk']}%, first at {b['first_at']}, last at {b['last_at']}"
        )
    return "\n".join(lines)


class FinalStats(BaseModel):
    candidate_id: str
    total_violations: int
    risk_score: int
    session_duration_sec: int = 300
    critical_flags: list[str] = []
    session_started_at: str | None = None


@app.post("/generate-verdict")
async def generate_verdict(stats: FinalStats):
    # Flush any episode still open, on both loops. The session has ended, so no
    # further frame or sweep will ever arrive to expire it — and the episode most
    # likely to be open at this moment is the one the candidate never recovered
    # from, which is exactly the one the report should be able to put a duration
    # on. Done BEFORE the breakdown is computed so those rows are complete when
    # it reads them.
    for tracker in (episode_tracker, prop_episode_tracker):
        for episode in tracker.close_candidate(stats.candidate_id):
            try:
                close_moment(
                    episode["moment_id"],
                    episode["t_end"],
                    episode["duration_sec"],
                    episode["frame_count"],
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not close episode {episode['moment_id']}: {e}")

    try:
        breakdown = _compute_session_breakdown(stats.candidate_id, since_iso=stats.session_started_at)
    except Exception as e:
        logger.error(f"⚠️ Could not compute breakdown for verdict: {e}")
        breakdown = {"violations_by_type": {}, "peak_event": None, "session_window": {}}

    breakdown_block = _format_breakdown_for_prompt(breakdown)
    critical_context = ""
    if stats.critical_flags:
        flags_text = ", ".join(stats.critical_flags)
        critical_context = (
            f"\nCRITICAL VIOLATIONS: {flags_text}. Address this directly in paragraph 2.\n"
        )

    prompt = f"""You are a supportive interview coach reviewing a candidate's proctored session. Your tone is direct, warm, and constructive. Never invent details; every claim must be grounded in the data below.

SESSION DATA
Candidate: {stats.candidate_id}
Peak risk score: {stats.risk_score}%
Total flagged events: {stats.total_violations}
Approx session duration: {stats.session_duration_sec}s{critical_context}

BEHAVIOURAL BREAKDOWN (grouped by type)
{breakdown_block}

Write a 3-paragraph coaching report in plain prose (no bullet lists, no headings).
Paragraph 1 — Acknowledge effort, name one concrete strength from the numbers.
Paragraph 2 — Name the single most impactful pattern. Quote count and approx duration.
Paragraph 3 — Two specific actionable habits. Close with genuine encouragement.

Hard rules:
- Use ONLY the numbers shown. Do not invent percentages.
- Speak as a coach, not as an AI.
- Say 'moment', 'event', or 'pattern', never 'violation' or 'breach'.
- Keep it under 220 words."""

    try:
        client = llm_config.make_client()
        completion = await client.chat.completions.create(
            model=llm_config.chat_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=520,
            extra_body=llm_config.extra_body_for(llm_config.chat_model()),
        )
        ai_report = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM error ({llm_config.describe()}): {e}")
        ai_report = (
            f"Mock Report (API unreachable):\n\n"
            f"You showed up and ran the full session — that takes focus.\n\n"
            f"The breakdown logged {stats.total_violations} flagged moments with a peak risk of {stats.risk_score}%.\n\n"
            f"Next session: keep your eyeline on the camera and treat each pause as a chance to re-center. You've got this."
        )

    return {
        "candidate_id": stats.candidate_id,
        "total_violations": stats.total_violations,
        "risk_score": stats.risk_score,
        "report": ai_report,
        "breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# GET /api/v1/logs  — Feeds the frontend AuditTrail component
# ---------------------------------------------------------------------------
@app.get("/api/v1/logs")
async def get_recent_logs():
    import sqlite3
    import contextlib
    from datetime import datetime
    from core_memory.timeline import DB_PATH
    
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        # Grab latest 50 frames, joined with moments where they exist
        rows = conn.execute(
            """
            SELECT f.frame_id as id, f.session_id as candidate_id, f.t as t, 
                   f.gaze, f.composure, f.head_pose, m.type as intervention_level, m.caption as logic_trace
            FROM timeline_frames f
            LEFT JOIN moments m ON f.t = m.t AND f.session_id = m.session_id
            ORDER BY f.t DESC LIMIT 50
            """
        ).fetchall()
        
        results = []
        for r in rows:
            risk = int(100 - r["composure"])
            # Synthesize intervention level if no explicit moment is tied to this frame
            if r["intervention_level"]:
                intervention = r["intervention_level"]
                logic = r["logic_trace"]
            else:
                intervention = "CLEAR" if risk < 20 else "SOFT_WARNING" if risk < 40 else "WARNING"
                logic = f"Gaze: {r['gaze']}, Pose: {r['head_pose']}"
                
            results.append({
                "id": r["id"],
                "candidate_id": r["candidate_id"],
                "timestamp": datetime.fromtimestamp(r["t"]).isoformat(),
                "gaze": r["gaze"],
                "is_critical": risk >= 80,
                "risk_score": risk,
                "intervention_level": intervention,
                "ai_logic_trace": logic
            })
        return results


# ---------------------------------------------------------------------------
# GET /api/v1/autopsy-logs  — Feeds the frontend /autopsy deep-dive page
#
# Reads sentry_logs.db/security_logs, which is the ARCHIVE written by earlier
# builds, not the live guard_telemetry.db that /api/v1/logs serves. The two are
# deliberately different sources: /api/v1/logs rebuilds an audit trail out of
# the current session's telemetry frames, which carry no imagery, whereas the
# autopsy page exists to show flagged moments *with* their evidence JPEG.
#
# The route itself went missing while the archive and the page that reads it
# both survived, so /autopsy rendered "backend offline" against a healthy
# server with 974 usable rows behind it.
#
# image_payload is returned verbatim: rows hold either "/api/v1/evidence/<f>"
# (served by the StaticFiles mount above) or a legacy bare-base64 blob, and
# resolveEvidenceSrc() in the page already handles both. Normalising here would
# break the older form.
# ---------------------------------------------------------------------------
@app.get("/api/v1/autopsy-logs")
async def get_autopsy_logs(limit: int = 60):
    import sqlite3
    import contextlib

    db_path = os.path.join(BACKEND_DIR, "sentry_logs.db")
    if not os.path.exists(db_path):
        return []

    # Read-only: this is an archive, and the autopsy page must never be able to
    # mutate the historical record it is auditing.
    with contextlib.closing(
        sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, candidate_id, timestamp, gaze, is_critical,
                       risk_score, intervention_level, ai_logic_trace,
                       image_payload
                FROM security_logs
                ORDER BY
                    -- Frames that actually captured a photo are the ones worth
                    -- autopsying; plain CLEAR rows would otherwise crowd them
                    -- out, since they outnumber them roughly 13 to 1.
                    CASE WHEN image_payload IS NOT NULL
                              AND image_payload != '' THEN 0 ELSE 1 END,
                    id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
        except sqlite3.OperationalError:
            # Archive predates this schema, or the table was never created.
            return []

        return [
            {
                "id": r["id"],
                "candidate_id": r["candidate_id"],
                "timestamp": r["timestamp"],
                "gaze": r["gaze"],
                "is_critical": bool(r["is_critical"]),
                "risk_score": r["risk_score"],
                "intervention_level": r["intervention_level"],
                "ai_logic_trace": r["ai_logic_trace"] or "",
                "image_payload": r["image_payload"] or "",
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Misc endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/status/{candidate_id}")
async def get_candidate_status(candidate_id: str):
    return await bea_engine.get_state(candidate_id)


# candidate_id is required. It used to default to the old hardcoded
# `major_project_candidate_01`, which outlived the move to per-resume identity
# (`resume_<hash>`, see frontend/src/lib/resumeMemory.ts): a caller that omitted
# the parameter got a successful "Memory cleared." for an identity no resume can
# produce, while the BEA state it meant to clear stayed latched. Failing with a
# 422 is the honest answer — every real caller already passes one.
@app.post("/reset-session")
async def reset_session(candidate_id: str):
    await bea_engine.reset_candidate(candidate_id)
    # Re-arm the prop sweep. Without this a phone still sitting in frame when
    # the user hits "Clear memory" stays latched as already-seen, so it would
    # never re-flag for the rest of the run.
    _prop_seen.pop(candidate_id, None)
    _prop_clear_streak.pop(candidate_id, None)
    _phone_full_streak.pop(candidate_id, None)
    # Drop open episodes rather than closing them. "Clear memory" means the run
    # is being discarded, so writing an end time and duration for a stretch the
    # user just erased would leave a finished-looking row pointing at a session
    # that no longer counts.
    episode_tracker.forget_candidate(candidate_id)
    prop_episode_tracker.forget_candidate(candidate_id)
    # Same reasoning, one step further: a latched cooldown would swallow the first
    # finding of the new run, and a half-advanced rotation would greet a first
    # sighting with "your phone is visible AGAIN" about a session the user just
    # erased.
    interrupt_director.forget_candidate(candidate_id)
    return {"status": "success", "message": "Memory cleared."}



# ---------------------------------------------------------------------------
# MR 3: AI Interviewer Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/v1/interview/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Accepts a PDF, parses it, and returns LLM-generated STAR questions."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"temp_{uuid.uuid4().hex}.pdf"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        resume_text = ai_interviewer.extract_text_from_pdf(temp_path)
        if not resume_text:
            raise HTTPException(status_code=422, detail="Could not extract text from the provided PDF.")

        # Refuse a PDF that is not a resume, and refuse it BEFORE the LLM call.
        # "Is a PDF" used to be the whole test, so an invoice or a boarding pass
        # went straight to generate_questions, which would invent four STAR
        # questions about it — a wasted call, a cached empty resume, and a
        # candidate who discovers the problem on /sentry rather than here.
        #
        # Logged with the score and the signals that fired, because the only
        # failure that matters is refusing a REAL resume, and that is impossible
        # to diagnose from the joke alone.
        refusal = resume_gate.inspect(resume_text)
        if refusal:
            logger.info(
                f"\U0001f9fe Refused {file.filename}: reads as {refusal['kind']}, "
                f"{refusal['score']} resume signals "
                f"({', '.join(refusal['signals']) or 'none'})"
            )
            raise HTTPException(status_code=422, detail=refusal["say"])

        logger.info(f"\U0001f9e0 Generating questions for resume: {file.filename}")
        questions = await ai_interviewer.generate_questions(resume_text)

        return {
            "status": "success",
            "filename": file.filename,
            "questions": questions,
            # The conversation engine needs the raw resume for context — its
            # /start-session contract takes resume_text. Omitting it here meant
            # the frontend always sent "" and the interviewer never saw the
            # resume, so it could not reference the candidate's actual
            # employers, university or projects.
            "resume_text": resume_text,
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------------------------
# MR 4: Local Voice Transcription (faster-whisper, fully offline)
# ---------------------------------------------------------------------------
ALLOWED_AUDIO_TYPES = {
    "audio/webm", "audio/ogg", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mp3", "audio/mpeg", "audio/mp4", "audio/m4a", "audio/flac",
    "video/webm",  # browser MediaRecorder often sends video/webm for audio-only
}


@app.post("/api/v1/voice/transcribe")
async def transcribe_audio_endpoint(file: UploadFile = File(...)):
    """Accept an audio file, transcribe locally via faster-whisper, return text."""

    # Validate content type (lenient — browsers vary)
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type and content_type not in ALLOWED_AUDIO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type: {content_type}. Send wav, webm, ogg, mp3, or m4a.",
        )

    # Determine file extension from the original filename or content type
    ext = ".webm"
    if file.filename:
        _, dot_ext = os.path.splitext(file.filename)
        if dot_ext:
            ext = dot_ext

    tmp_fd = None
    tmp_path = None
    try:
        # Write upload to a temp file (faster-whisper needs a file path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="guard_audio_")
        with os.fdopen(tmp_fd, "wb") as tmp_file:
            tmp_fd = None  # os.fdopen takes ownership of the fd
            shutil.copyfileobj(file.file, tmp_file)

        # Run transcription on the thread pool so we don't block the event loop
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(
            app.state.executor,
            whisper_transcribe,
            tmp_path,
        )

        logger.info(f"🎙️  Transcription complete: {len(transcript)} chars")
        return {"status": "success", "transcript": transcript}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed.")
    finally:
        # Clean up temp file
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# MR 6: Verdict Engine — AI Coaching Report
# ---------------------------------------------------------------------------
class VerdictRequest(BaseModel):
    questions: list[str]
    transcripts: list[str]
    average_focus_score: float


@app.post("/api/v1/interview/generate-verdict")
async def generate_verdict(payload: VerdictRequest):
    """Evaluate interview answers + focus score and return an AI coaching report."""

    if len(payload.questions) == 0 or len(payload.transcripts) == 0:
        raise HTTPException(status_code=400, detail="Questions and transcripts are required.")

    if len(payload.questions) != len(payload.transcripts):
        raise HTTPException(
            status_code=400,
            detail="Questions and transcripts must be the same length.",
        )

    # Build the QA block for the LLM
    qa_block = ""
    for i, (q, a) in enumerate(zip(payload.questions, payload.transcripts), 1):
        qa_block += f"\n--- Question {i} ---\nQ: {q}\nA: {a}\n"

    focus_label = (
        "Excellent" if payload.average_focus_score >= 80
        else "Moderate" if payload.average_focus_score >= 50
        else "Poor"
    )

    prompt = f"""You are a direct, evidence-based interview coach. A candidate just completed a practice interview.

INTERVIEW DATA:
{qa_block}

FOCUS/COMPOSURE SCORE: {payload.average_focus_score:.0f}/100 ({focus_label})
(This score measures eye contact, head stability, and engagement via computer vision during the interview.)

Return ONLY valid JSON with this exact shape:
{{
  "verdict": "one conclusion-first sentence, max 18 words",
  "strengths": ["specific evidence-based strength"],
  "primary_improvement": "one highest-impact improvement, max 30 words",
  "next_actions": ["specific action", "specific action"],
  "readiness": "Strong|Developing|Needs targeted practice"
}}

Keep the complete output under 130 words. Use 1-3 strengths and 2-3 actions.
Use only the answers and focus score provided. Do not invent filler words,
confidence, tone, or body-language observations. No markdown or extra prose."""

    try:
        client = llm_config.make_client()

        completion = await client.chat.completions.create(
            model=llm_config.chat_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=800,
            extra_body=llm_config.extra_body_for(llm_config.chat_model()),
        )

        raw_report = completion.choices[0].message.content.strip()
        coaching = llm_config.normalize_coaching(llm_config.extract_json(raw_report))
        report = coaching.get("verdict", "Interview feedback is ready.")
        logger.info(f"📋 Verdict generated: {len(report)} chars")

        return {
            "status": "success",
            "report": report,
            "coaching": coaching,
            "focus_score": payload.average_focus_score,
            "focus_label": focus_label,
        }

    except Exception as e:
        logger.error(f"❌ Verdict generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate coaching report.")


# ---------------------------------------------------------------------------
# MR 7: Conversational Interview Engine
# ---------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    resume_text: str
    questions: list[dict]
    # The interviewer style the user picked before engaging. Optional so older
    # clients (and the tests) keep working — omitting it gives the original
    # full warmup-to-pressure ramp.
    starting_persona: str = "friendly_hr"


class ConversationTurnRequest(BaseModel):
    session_id: str
    transcript: str
    focus_score: float = 100.0


class EndSessionRequest(BaseModel):
    session_id: str


@app.post("/api/v1/interview/start-session")
async def start_interview_session(payload: StartSessionRequest):
    """Create a conversational interview session and return the AI's opening greeting."""
    try:
        result = await conversation_engine.create_session(
            starting_persona=payload.starting_persona,
            resume_text=payload.resume_text,
            questions=payload.questions,
        )
        logger.info(f"🎬 Interview session started: {result['session_id']}")
        return result
    except Exception as e:
        logger.error(f"❌ Session creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to start interview session.")


@app.post("/api/v1/interview/conversation-turn")
async def interview_conversation_turn(payload: ConversationTurnRequest):
    """Process a candidate's spoken answer and return the AI's conversational response."""
    try:
        result = await conversation_engine.process_candidate_turn(
            session_id=payload.session_id,
            transcript=payload.transcript,
            focus_score=payload.focus_score,
        )
        logger.info(
            f"💬 Turn {result['turn_number']} | persona={result['persona']} | "
            f"complete={result['is_complete']}"
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Conversation turn failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Conversation turn error: {str(e)}")


@app.post("/api/v1/interview/end-session")
async def end_interview_session(payload: EndSessionRequest):
    """Generate the final empathetic coaching verdict for the interview session."""
    try:
        result = await conversation_engine.generate_final_verdict(payload.session_id)
        logger.info(f"📋 Session {payload.session_id} verdict generated")
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Verdict generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate verdict.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
