"use client";

import { useEffect, useRef, useState, useCallback, useImperativeHandle, type Ref } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { activeCandidateId } from "@/lib/resumeMemory";
import PersonaPicker from "@/components/PersonaPicker";
import { DEFAULT_PERSONA, type PersonaId } from "@/lib/personas";

const TELEMETRY_INTERVAL_MS = 2000;

export interface FrameFlag {
  kind: string;
  text: string;
  critical: boolean;
}

/**
 * A one-shot instruction to say something to the candidate, mid-session.
 *
 * Present on a response only for the frame a confirmed phone or second person
 * FIRST appears — the backend fires on the rising edge, so this is null on every
 * subsequent frame the same prop stays in view. It is therefore not state to
 * render; act on it once and drop it.
 */
export interface ScopeInterrupt {
  kind: "MOBILE_DEVICE" | "MULTIPLE_FACES";
  /** The exact sentence to speak. Wording rotates per sighting, backend-side. */
  say: string;
  /** 1-based count of times this session has been told about this kind. */
  occurrence: number;
}

export interface RiskPacket {
  candidate_id: string;
  risk_score: number;
  // Highest risk the session ever reached. Monotonic — the backend only ever
  // raises it. `risk_score` is a LIVE reading and correctly falls when the
  // candidate recovers; this is what the "Cumulative risk" figure and the final
  // report must quote, or a session that peaked at 100% reads as 0% seconds later.
  peak_risk?: number;
  violation_count: number;
  critical_flags?: string[];
  intervention_level: "CLEAR" | "SOFT_WARNING" | "HARD_WARNING" | "WARNING_LOGGED" | "SEVERE_VIOLATION_LOGGED";
}

// Imperative surface exposed to parent for session-level control (auto-disengage on end).
export interface SniperScopeHandle {
  stopCamera: () => void;
}

/**
 * One YOLO detection from the prop sweep, in full-frame normalized coordinates.
 *
 * `source` names the pass that produced it (currently always "full") and
 * `fired` says whether it cleared the confidence
 * floor and so counted toward the verdict. Both are drawn on screen: a phone
 * accusation you cannot see the box for is impossible to argue with, and
 * knowing WHICH pass is making the claim is what separates "there is a phone"
 * from "the magnified lower tile thinks your earpiece is a phone".
 */
interface PropBox {
  label: string;
  conf: number;
  source: string;
  fired: boolean;
  box: [number, number, number, number];
}

/**
 * How long a sweep's boxes stay painted.
 *
 * The overlay repaints every telemetry frame but boxes only arrive every
 * PROP_SCAN_MS (~1.2s), so they have to be held in a ref and redrawn from it —
 * drawing them once per sweep would show a box for one frame in twenty. The TTL
 * is roughly two sweeps: long enough to survive a slow round trip without
 * flickering, short enough that a box does not outlive the object by much.
 */
const PROP_BOX_TTL_MS = 2500;

/**
 * Paint the prop sweep's detections over the video.
 *
 * Coordinates arrive normalized against the full frame, so they scale to the
 * canvas rather than assuming the 640x360 the sweep happens to send. The video
 * is not mirrored (no scaleX(-1) on either element) and both it and this canvas
 * are 16:9 object-cover over a 16:9 frame, so no crop correction is needed —
 * normalized x maps straight onto canvas x.
 */
function drawPropBoxes(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  boxes: PropBox[],
) {
  for (const b of boxes) {
    const [nx1, ny1, nx2, ny2] = b.box;
    const x = nx1 * canvas.width;
    const y = ny1 * canvas.height;
    const w = (nx2 - nx1) * canvas.width;
    const h = (ny2 - ny1) * canvas.height;

    // Solid red is what accused the candidate; dashed amber was seen and NOT
    // counted. The difference is the whole point of drawing sub-floor boxes: a
    // dashed box on an earpiece says the detector is looking at the right pixels
    // and drawing the wrong conclusion, which is a different bug from a box
    // sitting on empty desk.
    ctx.lineWidth = b.fired ? 2 : 1;
    ctx.strokeStyle = b.fired ? "#FF3B30" : "#FFB020";
    ctx.setLineDash(b.fired ? [] : [4, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    const label = `${b.label} ${Math.round(b.conf * 100)}% ${b.source}`;
    ctx.font = "600 11px ui-monospace, monospace";
    const textWidth = ctx.measureText(label).width;
    // Flip the caption below the box when the box is near the top edge, so it
    // is never clipped off the canvas.
    const labelY = y > 14 ? y - 13 : y + h + 1;

    ctx.fillStyle = b.fired ? "#FF3B30" : "#FFB020";
    ctx.fillRect(x, labelY, textWidth + 6, 13);
    ctx.fillStyle = "#000000";
    ctx.fillText(label, x + 3, labelY + 10);
  }
}

interface SniperScopeProps {
  onTelemetryUpdate: (packet: RiskPacket, verdict: string) => void;
  onDisengage?: () => void;
  /**
   * The pre-engage persona pick, lifted to the page so it can send
   * `starting_persona` on session start. The ladder is seeded once — mid-run
   * escalation is the engine's call — so the picker locks while scanning.
   */
  onPersonaChange?: (id: PersonaId) => void;
  /**
   * A confirmed phone or second person, to be said out loud to the candidate.
   *
   * Raised to the page rather than spoken here because the page owns the only
   * `speechSynthesis` in the app: its `speakText` calls `cancel()` first, so a
   * second speaker in this component would cut the interviewer off mid-question
   * and vice versa. One owner, one queue.
   */
  onInterrupt?: (interrupt: ScopeInterrupt) => void;
  /**
   * Called synchronously inside the Engage click, before the camera opens.
   * The page uses it to enter fullscreen, which browsers only allow from a
   * user gesture; by the time the interview starts that gesture is long gone.
   */
  onEngageClick?: () => void;
  ref?: Ref<SniperScopeHandle>;
}

export default function SniperScope({ onTelemetryUpdate, onDisengage, onPersonaChange, onInterrupt, onEngageClick, ref }: SniperScopeProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  /**
   * Latest prop-sweep detections, with the time they landed.
   *
   * A ref rather than state: the overlay is drawn from the telemetry loop at
   * frame rate, and putting boxes through setState would re-render the whole
   * page (camera panel, transcript, chat) on every sweep for a value only the
   * canvas reads — the same reason audioLevelRef exists in useVAD.
   */
  const propBoxesRef = useRef<{ at: number; boxes: PropBox[] }>({ at: 0, boxes: [] });
  const loopTimerRef = useRef<NodeJS.Timeout | null>(null);

  const [isScanning, setIsScanning] = useState(false);
  const [persona, setPersona] = useState<PersonaId>(DEFAULT_PERSONA);
  const [sysStatus, setSysStatus] = useState<"IDLE" | "ACTIVE" | "COMPROMISED">("IDLE");
  
  // Calibration State
  const [isCalibrating, setIsCalibrating] = useState(false);
  const calibrationRef = useRef({
    isCalibrating: false,
    // pitchDeg/yawDeg are true degrees off the 3D face normal. They were named
    // pitchRatio/noseX back when pitch was a forehead/chin proportion and yaw was
    // a raw nose x — different units entirely, so the names are not kept as
    // aliases; a "ratio" holding degrees is how a sign bug hides in plain sight.
    samples: [] as { pitchDeg: number, yawDeg: number, gazeX: number, gazeY: number, eyeW: number, faceH: number }[],
    baselinePitch: null as number | null,
    baselineYaw: null as number | null,
    baselineGazeX: null as number | null,
    baselineGazeY: null as number | null,
    // Head-on reference geometry. These two are what make the iris offsets
    // yaw-invariant — see the YAW-INVARIANT NORMALISER note in the mesh handler.
    baselineEyeW: null as number | null,
    baselineFaceH: null as number | null,
    startTime: 0
  });
  
  // Telemetry State
  const [riskScore, setRiskScore] = useState(0);
  // Session high-water mark, shown as "Cumulative risk". Held separately from
  // riskScore because that value is a live reading that falls back to 0 the
  // moment the candidate looks straight again — so the panel showed the number
  // climb to 100 and then vanish, which reads as the system forgetting. The
  // backend is authoritative (bea._stamp_peak); Math.max here only guards
  // against an older backend that omits the field.
  const [peakRisk, setPeakRisk] = useState(0);
  const [violationCount, setViolationCount] = useState(0);
  const [interventionLevel, setInterventionLevel] = useState("CLEAR");
  const [latestVerdict, setLatestVerdict] = useState("SYSTEM STANDBY");
  // Every condition true on the last analysed frame, in severity order. Empty on
  // a clean frame and before the first response, which is when latestVerdict —
  // still the single-string fallback for the object sweep and for standby text —
  // is rendered instead.
  const [flags, setFlags] = useState<FrameFlag[]>([]);
  const [toast, setToast] = useState<{ message: string; type: "success" | "error" } | null>(null);

  // Identity for this run, scoped to the resume being practised with.
  //
  // This was a hardcoded literal, which meant every resume on the machine wrote
  // into one shared timeline — upload a different CV and the dashboard still
  // showed the old one's sessions. The dashboard reads through the same helper,
  // so the write and the read cannot drift apart.
  //
  // Resolved once per mount via a ref rather than on every render: it reads
  // localStorage (unavailable during the server render, and it must not change
  // underneath a run that is already recording).
  const candidateIdRef = useRef<string>("");
  if (!candidateIdRef.current) candidateIdRef.current = activeCandidateId();
  const CANDIDATE_ID = candidateIdRef.current;


  // One id per practice run, minted at mount. Two distinct grains:
  //   CANDIDATE_ID -> stable identity (BEA lockout, /status, /reset-session)
  //   session_id   -> one row per run, so dashboard trends/streaks/deltas move.
  // Format must stay "{candidate_id}__{rand}" — the dashboard query prefix-matches it.
  const sessionIdRef = useRef<string>("");
  if (!sessionIdRef.current) {
    const rand =
      typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID().replace(/-/g, "").slice(0, 12)
        : Math.random().toString(36).slice(2, 14);
    sessionIdRef.current = `${CANDIDATE_ID}__${rand}`;
    // The report page needs this id to fetch the run's timeline. It was only
    // ever held in this ref, so /report had no way to ask for the frames it
    // had just generated and rendered a session with no drift history at all.
    try {
      sessionStorage.setItem("guard_vision_session", sessionIdRef.current);
    } catch {
      // sessionStorage unavailable — the report falls back to the candidate id.
    }
  }

  const [isFaceMeshReady, setIsFaceMeshReady] = useState(false);

  // --- MEDIAPIPE GATEKEEPER STATE ---
  const telemetryRef = useRef({
    faces_detected: 0, // Better default so we don't spam multiple face alerts before real data
    is_talking: false,
    head_pose: "HEAD_CENTER",
    gaze_vector: [0, 0],
    // Pose agreed across consecutive rAF frames. Written alongside head_pose so
    // the two can never disagree; without it, a single blink-misfire frame from
    // the 30fps gatekeeper was stamped onto the whole inference window as the
    // frame verdict. Telemetry now samples every two seconds.
    stable_pose: "HEAD_CENTER" as string | null,
    // The two pre-fusion channels, smoothed the same way as stable_pose.
    stable_head_pose_raw: "HEAD_CENTER" as string | null,
    stable_gaze_class: "unknown" as string | null,
    // Whether the candidate's eyes are open. Defaults open: an absent reading is
    // not evidence of closure, and no-face is already its own separate critical.
    eyes_open: true,
  });
  const gatekeeperRef = useRef<{ camera: any; faceMesh: any } | null>(null);

  // Trailing window of per-frame pose calls, used to smooth the gatekeeper's
  // ~30fps output before the reporting cadence. The inference loop reads
  // telemetryRef at an arbitrary instant, so before this it was sampling ONE
  // frame out of ~150 and calling that the verdict for the window — a single
  // frame of iris jitter crossing GAZE_DELTA_Y was enough to report a drift
  // while the user sat still. A frame has to be corroborated by its neighbours.
  const poseWindowRef = useRef<{ t: number; pose: string }[]>([]);
  // Separate windows for the two PRE-FUSION channels. These exist so the
  // asymmetric iris veto can be evaluated after the fact: the fused label alone
  // cannot tell you what either sensor said before fusion, so a recorded session
  // can't distinguish "the eyes vetoed a head turn" from "the head never turned".
  //
  // Each channel gets its OWN modal window rather than sharing the fused one.
  // Comparing a smoothed fused label against raw per-frame sensor reads would
  // make the pre-fusion channels look noisier purely because they weren't
  // smoothed — the fused channel would win an unfair comparison.
  const headWindowRef = useRef<{ t: number; pose: string }[]>([]);
  const gazeWindowRef = useRef<{ t: number; pose: string }[]>([]);
  // Eye open/closed, smoothed exactly like the pose channels. A blink is a few
  // frames out of the 1.5s window and never clears the >50% majority stableOver
  // demands, so this reports CLOSED only for a sustained shut — never a blink.
  const eyesWindowRef = useRef<{ t: number; pose: string }[]>([]);
  const POSE_WINDOW_MS = 1500;
  const POSE_MIN_SAMPLES = 3;

  /**
   * Records one frame's call on a channel and returns the modal value across
   * that channel's trailing window, or null while the window is too thin or too
   * split to trust.
   */
  const stableOver = (
    ref: { current: { t: number; pose: string }[] },
    pose: string,
    now: number,
  ): string | null => {
    ref.current.push({ t: now, pose });
    // Prune by TIME, not by count: the gatekeeper's frame rate varies with CPU
    // load, so a fixed-length ring would cover a different real duration on a
    // busy machine than on an idle one.
    ref.current = ref.current.filter((s) => now - s.t <= POSE_WINDOW_MS);

    const w = ref.current;
    if (w.length < POSE_MIN_SAMPLES) return null;
    const counts = new Map<string, number>();
    for (const s of w) counts.set(s.pose, (counts.get(s.pose) ?? 0) + 1);
    let best: string | null = null;
    let bestN = 0;
    for (const [pose, n] of counts) {
      if (n > bestN) { best = pose; bestN = n; }
    }
    // A plurality is not enough — with the window split three ways the "winner"
    // can hold a third of the frames. Requiring more than half means a reported
    // drift was the dominant read, not merely the most common noise.
    return bestN * 2 > w.length ? best : null;
  };

  /**
   * Records a frame's pose call and returns the modal pose across the trailing
   * window, or null while the window is too thin or too split to trust.
   */
  const stablePose = (pose: string, now: number): string | null =>
    stableOver(poseWindowRef, pose, now);

  // Stable reference to parent callback — prevents the inference loop from being
  // re-created on every parent render, which previously spawned parallel loop chains
  // and produced ~10x the intended request rate (DB rows showed 500 ms cadence).
  const onTelemetryUpdateRef = useRef(onTelemetryUpdate);
  useEffect(() => {
    onTelemetryUpdateRef.current = onTelemetryUpdate;
  }, [onTelemetryUpdate]);

  // Same reason, same shape: read through a ref inside the loop so a new
  // callback identity on the parent cannot re-create the inference chain.
  const onInterruptRef = useRef(onInterrupt);
  useEffect(() => {
    onInterruptRef.current = onInterrupt;
  }, [onInterrupt]);

  // Single-flight guard so duplicate scheduling sources can't start a second chain.
  const loopRunningRef = useRef(false);

  useEffect(() => {
    // Inject MediaPipe Face Mesh from /public/mediapipe (locally bundled, no CDN dependency).
    // Assets are copied from node_modules by scripts/copy-mediapipe.mjs at predev/prebuild.
    if (typeof window !== "undefined" && !(window as any).FaceMesh) {
      const script = document.createElement("script");
      script.src = "/mediapipe/face_mesh.js";
      script.async = true;
      script.onload = () => {
        setIsFaceMeshReady(true);
      };
      document.body.appendChild(script);
    } else {
      setIsFaceMeshReady(true);
    }
  }, []);

  // --- 1. HARDWARE TRIPWIRE: Virtual Camera Detection ---
  // Returns true if a virtual camera was detected (caller should abort startup).
  const checkHardware = async (): Promise<{ compromised: boolean; label?: string }> => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(d => d.kind === 'videoinput');

      const blocklist = ['obs', 'virtual', 'snap', 'manycam', 'xsplit', 'droidcam', 'iriun', 'epoccam'];
      for (const device of videoDevices) {
        const label = device.label.toLowerCase();
        if (blocklist.some(token => label.includes(token))) {
          console.warn("Hardware tripwire: virtual camera detected:", device.label);
          return { compromised: true, label: device.label };
        }
      }
      return { compromised: false };
    } catch {
      console.warn("Hardware scan failed.");
      return { compromised: false };
    }
  };

  // --- 1.5 DYNAMIC POSE CLASSIFICATION ---
  //
  // Thresholds are real degrees of head rotation now, not opaque ratio deltas.
  // The reference MediaPipe/solvePnP implementations gate at ~10 deg; we sit a
  // little above that because our angles come from the mesh's z channel, which
  // is noisier than a full PnP solve.
  // Pitch is split by DIRECTION; yaw is not. Down and up are not symmetric
  // problems even though one angle measures both:
  //
  //   - DOWN has a large, well-conditioned signal and a strong prior. Dropping
  //     the chin swings the face normal through a wide arc, and 13 deg is set
  //     where it is to sit above a tilted laptop lid at rest.
  //   - UP is the same rotation run backwards into a worse-conditioned part of
  //     the estimate. The normal is built from chin(152) -> forehead(10) crossed
  //     with the eye-corner vector, and on an upward tilt the chin is the
  //     landmark that moves most, is occluded soonest, and is already the least
  //     stable point on the mesh (it also moves when the jaw opens, i.e. while
  //     talking). The elevation of that normal leans on MediaPipe's z channel,
  //     which the note above already flags as noisier than a full PnP solve.
  //     The upshot is that up reads SHORT: a genuine look up measures well
  //     under its true angle and never crosses a threshold set for down.
  //
  // Lowering the up threshold is safe here specifically because of the
  // centred-iris veto in fuseSensors. A spurious 8-13 deg up reading with the
  // eyes still on the screen is Case B, which vetoes it back to HEAD_CENTER
  // below HEAD_STRONG_DEG. So this threshold can only produce a flag when the
  // iris independently agrees the user looked up. Leaning back to think stays
  // centred and stays unflagged; that is the case the veto exists for.
  const PITCH_DOWN_DELTA_DEG = 13;
  const PITCH_UP_DELTA_DEG = 8;
  const YAW_DELTA_DEG = 16;
  // A deflection this large is not a webcam sitting slightly off-axis or mesh
  // jitter — it is a deliberate head movement. Used by fuseSensors to decide
  // when the iris is allowed to overrule the head. See notes there.
  const HEAD_STRONG_DEG = 22;

  const classifyPose = (
    pitchDeg: number,
    yawDeg: number,
    baselinePitch: number | null,
    baselineYaw: number | null,
  ) => {
    // Baseline-relative, because a laptop lid puts the lens well below eye level
    // and the resting face normal is genuinely pitched. Absolute angles would
    // read that fixed offset as a permanent look-away.
    const dPitch = pitchDeg - (baselinePitch ?? 0);
    const dYaw = yawDeg - (baselineYaw ?? 0);

    // Which pitch threshold applies depends on which way the head went, so it
    // has to be picked before the tie-break — the tie-break normalises by it,
    // and normalising an upward deflection by the DOWN threshold would
    // understate the pitch axis and hand borderline frames to yaw.
    const pitchDelta = dPitch > 0 ? PITCH_DOWN_DELTA_DEG : PITCH_UP_DELTA_DEG;

    // Yaw is still tested first, but now only to pick a winner when both axes
    // fire at once — the axes are independent in a 3D frame, so this is a
    // tie-break, not the correctness crutch it used to be.
    if (Math.abs(dYaw) > YAW_DELTA_DEG && Math.abs(dYaw) / YAW_DELTA_DEG >= Math.abs(dPitch) / pitchDelta) {
      return dYaw > 0 ? "HEAD_LEFT" : "HEAD_RIGHT";
    }
    if (Math.abs(dPitch) > pitchDelta) {
      return dPitch > 0 ? "HEAD_DOWN" : "HEAD_UP";
    }
    if (Math.abs(dYaw) > YAW_DELTA_DEG) {
      return dYaw > 0 ? "HEAD_LEFT" : "HEAD_RIGHT";
    }
    return "HEAD_CENTER";
  };

  // How far the head is off its baseline, in degrees, on whichever axis is
  // furthest out. Feeds the strong-deflection rule in fuseSensors.
  const poseMagnitudeDeg = (
    pitchDeg: number,
    yawDeg: number,
    baselinePitch: number | null,
    baselineYaw: number | null,
  ) => Math.max(
    Math.abs(pitchDeg - (baselinePitch ?? 0)),
    Math.abs(yawDeg - (baselineYaw ?? 0)),
  );

  // Separate thresholds per axis. Both offsets are normalised by eye WIDTH, and
  // the eye is far wider than it is tall, so a vertical look-away moves the iris
  // through a much smaller fraction of that width than a horizontal one. A single
  // shared threshold would either miss every vertical drift or fire constantly on
  // horizontal noise.
  const GAZE_DELTA_X = 0.10;
  const GAZE_DELTA_Y = 0.055;

  /**
   * Iris direction, relative to the calibrated baseline.
   *
   * Naming is SUBJECT-relative ("gaze_left" = the user looked to their own left),
   * matching classifyPose, which calls a nose displaced toward image-right
   * HEAD_LEFT. The feed is not mirrored, so subject-left appears at image-right —
   * hence the inversion on the x comparison below. Keeping one convention across
   * both classifiers is what lets fuseSensors compare them at all.
   *
   * `irisUsable === false` means a blink or a degenerate mesh. That is reported as
   * "unknown", NOT as "center": fuseSensors treats a centred iris as positive
   * evidence the user is on-screen and vetoes head pose with it, so returning
   * "center" here would suppress a genuine look-away every time the user blinked
   * through it. "unknown" is the no-opinion value, and it hands the call to the head.
   */
  const classifyGaze = (
    gazeX: number, gazeY: number,
    baselineGazeX: number | null, baselineGazeY: number | null,
    irisUsable: boolean,
  ): string => {
    if (!irisUsable) return "unknown";

    const bx = baselineGazeX !== null ? baselineGazeX : 0;
    const by = baselineGazeY !== null ? baselineGazeY : 0;

    const dx = gazeX - bx;
    const dy = gazeY - by;

    // Compare each axis against its own threshold before deciding which wins,
    // rather than comparing raw magnitudes: dx and dy are on different scales
    // now, so |dx| > |dy| would hand almost every frame to the x axis.
    const xFires = Math.abs(dx) > GAZE_DELTA_X;
    const yFires = Math.abs(dy) > GAZE_DELTA_Y;

    if (xFires && (!yFires || Math.abs(dx) / GAZE_DELTA_X >= Math.abs(dy) / GAZE_DELTA_Y)) {
      return dx > 0 ? "gaze_left" : "gaze_right";
    }
    if (yFires) {
      return dy > 0 ? "gaze_down" : "gaze_up";
    }
    return "center";
  };


  // === SENSOR FUSION ===
  //
  // Maps an iris call onto the head vocabulary so the two can be compared.
  const GAZE_TO_POSE: Record<string, string> = {
    gaze_left: "HEAD_LEFT",
    gaze_right: "HEAD_RIGHT",
    gaze_up: "HEAD_UP",
    gaze_down: "HEAD_DOWN",
  };

  // The label an iris-only deflection is reported under. Deliberately NOT the
  // HEAD_* vocabulary: the pose field is narrated downstream as "head tilted X",
  // so sending HEAD_DOWN for a read that came purely from the eyes asserts the
  // head moved when it did not. Attention still drifted and is still flagged —
  // the sensor that saw it just stays attached to the claim.
  const GAZE_TO_DRIFT: Record<string, string> = {
    gaze_left: "GAZE_LEFT",
    gaze_right: "GAZE_RIGHT",
    gaze_up: "GAZE_UP",
    gaze_down: "GAZE_DOWN",
  };
  const poseAxis = (p: string) =>
    p === "HEAD_LEFT" || p === "HEAD_RIGHT" ? "h" : p === "HEAD_UP" || p === "HEAD_DOWN" ? "v" : null;

  const fuseSensors = (headPose: string, gazePose: string, headMagDeg: number): string => {
    const gazeAsPose = GAZE_TO_POSE[gazePose] ?? null;

    // "unknown" is a blink or an unusable iris read, NOT a centred eye. Treating
    // the two as the same thing is why looking down used to vanish: dropping the
    // eyes narrows the lids, the iris read was discarded as "center", and Case B
    // below then vetoed a perfectly good head-down into HEAD_CENTER. With no iris
    // opinion available the head is the only sensor left, so it is believed.
    if (gazePose === "unknown") return headPose;

    // Case A: Both agree on center → clean center
    if (headPose === "HEAD_CENTER" && gazePose === "center") return "HEAD_CENTER";

    // Case B: head deflects, eyes centred → normally the eyes veto, which is what
    // suppresses a tilted laptop lid reading as a permanent look-away. But a lid
    // tilt is a few degrees; past HEAD_STRONG_DEG the head has genuinely moved and
    // eyes centred *in their sockets* just means they travelled with it. Vetoing
    // there hid the most obvious cheat there is — looking down at a lap.
    if (headPose !== "HEAD_CENTER" && gazePose === "center") {
      return headMagDeg >= HEAD_STRONG_DEG ? headPose : "HEAD_CENTER";
    }

    // Case B2: the sensors disagree on AXIS — the head reads a side turn while the
    // iris reads a vertical drift. They cannot both describe one movement, so
    // promoting the iris (Case C) turned a real right turn into HEAD_UP.
    //
    // On a side turn the vertical iris channel is the weaker of the two: the eye is
    // foreshortened and the far one is partly self-occluded, whereas the 3D face
    // normal stays well conditioned. So the head keeps its horizontal call here.
    if (poseAxis(headPose) === "h" && poseAxis(gazeAsPose ?? "") === "v") return headPose;

    // Case B3: same axis, OPPOSITE directions. This is the other half of the
    // "I looked down and it said up" bug, and it survives the geometry fix above
    // because it is anatomy rather than maths: drop your chin while still reading
    // the screen and the eyes counter-roll UP in their sockets. The iris genuinely
    // points up relative to the face; the face is pointing down. Only one of those
    // describes where the attention went, and it is the head — the counter-roll
    // exists precisely to keep the eyes where they already were.
    if (
      gazeAsPose &&
      poseAxis(headPose) !== null &&
      poseAxis(headPose) === poseAxis(gazeAsPose) &&
      headPose !== gazeAsPose
    ) {
      return headPose;
    }

    // Case B4: a strong head deflection is not overruled by the iris at all.
    if (headPose !== "HEAD_CENTER" && headMagDeg >= HEAD_STRONG_DEG) return headPose;

    // Case C: Eyes drift regardless of head pose → iris wins, this is the killer
    // feature. Reported under GAZE_* rather than HEAD_*: the head is centred in
    // this branch by construction (every headPose !== HEAD_CENTER path has
    // already returned above), so calling it a head tilt is a claim about a
    // sensor that did not fire. This is the "dead centre, says tilted down" bug.
    if (gazeAsPose) return GAZE_TO_DRIFT[gazePose] ?? gazeAsPose;

    // Fallback
    return headPose;
  };

  // --- 2. MEDIAPIPE FACE MESH (30fps CPU Gatekeeper) ---
  const initGatekeeper = (videoEl: HTMLVideoElement) => {
    // @ts-ignore
    const FaceMesh = window.FaceMesh || (window as any).FaceMesh;
    if (!FaceMesh) {
      console.warn("FaceMesh not yet loaded. Trying again in 1s.");
      setTimeout(() => {
        if (!gatekeeperRef.current && videoRef.current) {
          gatekeeperRef.current = initGatekeeper(videoRef.current) as any;
        }
      }, 1000);
      return null;
    }

    const faceMesh = new FaceMesh({
      locateFile: (file: string) => `/mediapipe/${file}`,
    });

    faceMesh.setOptions({
      maxNumFaces: 3,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults((results: any) => {
      let faces = 0;
      let talking = false;
      let pose = "HEAD_CENTER";
      let gazeVector = [0, 0];
      // Pre-fusion sensor calls, hoisted so they survive past the landmark block
      // and can be reported alongside the fused label. Defaults match the "no
      // opinion" value of each classifier: HEAD_CENTER is classifyPose's null
      // reading, "unknown" is classifyGaze's. With no face present neither ran,
      // and the backend ignores pose entirely unless exactly one face is in
      // shot, so these defaults are never analysed — they only keep the field
      // types stable.
      let headPoseRaw = "HEAD_CENTER";
      let gazeClass = "unknown";
      // Eyelids open? Hoisted like the pose channels so a sustained closure can
      // be forwarded to the backend. Defaults open: with no face there are no
      // eyelids to measure, and that frame is already the NO_FACE critical.
      let eyesOpen = true;

      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        // Size-gate the face count. MediaPipe fits a full mesh to ANY face it
        // finds — a bystander's partial face at the edge of frame, or someone
        // several feet behind the candidate — and the old count treated those
        // identically to a second person sitting shoulder-to-shoulder. A genuine
        // second candidate is about as close, and therefore about as large, as
        // the primary; a background face is smaller. Counting only faces at least
        // half the height of the largest one keeps the real second person and
        // drops the passer-by. Both heights scale together as the candidate moves
        // toward or away from the lens, so the ratio holds at any sitting
        // distance. (A same-size face at the frame edge is a real second person
        // and still counts — which is correct.)
        const SECONDARY_FACE_MIN_RATIO = 0.5;
        const faceHeights = results.multiFaceLandmarks.map((lm: Array<{ y: number }>) =>
          Math.abs((lm[152]?.y ?? 0) - (lm[10]?.y ?? 0))
        );
        const maxFaceH = Math.max(...faceHeights);
        faces =
          maxFaceH > 1e-6
            ? faceHeights.filter((h: number) => h >= maxFaceH * SECONDARY_FACE_MIN_RATIO).length
            : results.multiFaceLandmarks.length;

        if (faces > 1) {
          console.warn("🚨 CPU GATEKEEPER: Multiple Faces Detected.");
        }

        const primary = results.multiFaceLandmarks[0];
        
        // 1. Mouth Aspect Ratio (MAR)
        const upperLip = primary[13];
        const lowerLip = primary[14];
        if (Math.abs(upperLip.y - lowerLip.y) > 0.015) {
          talking = true;
        }

        // 2. Head Pose — TRUE 3D FACE NORMAL, in degrees.
        //
        // This replaces two 2D projection ratios (forehead/chin proportion for
        // pitch, nose-between-eye-corners for yaw) that were the root cause of
        // "I looked down and it said head up".
        //
        // Why the old pitch ratio could invert the sign: forehead(10) and
        // chin(152) sit on opposite faces of a curved skull, so a pitch rotation
        // moves them toward and away from the camera at the same time. The 2D
        // y-gap between them is therefore NOT monotonic in pitch — it shrinks on
        // the way down, bottoms out, then grows again as the crown comes over the
        // top. Past that turning point "further down" reads as a smaller ratio,
        // which is exactly the HEAD_UP branch. No threshold tuning can fix a
        // non-monotonic signal; the geometry has to change.
        //
        // MediaPipe already gives a z per landmark, so we can build the face's
        // own coordinate frame and read the orientation off it directly:
        //   right = outer eye corner -> outer eye corner
        //   up    = chin -> forehead
        //   fwd   = right x up       (the direction the face actually points)
        // pitch/yaw are then the elevation/azimuth of fwd, in degrees. Both are
        // monotonic across the full range and each axis is independent of the
        // other, which also removes the yaw-contaminates-pitch coupling the old
        // code had to work around by ordering its comparisons.
        const forehead = primary[10];
        const chin = primary[152];
        const leftEye = primary[33];
        const rightEye = primary[263];

        // MediaPipe normalises x by frame WIDTH and y by frame HEIGHT, while z is
        // on roughly the same scale as x. Left unconverted, y is stretched by the
        // aspect ratio relative to the other two axes and every angle below comes
        // out wrong. Scaling y by H/W puts all three axes in one unit.
        const vw = videoEl.videoWidth || 640;
        const vh = videoEl.videoHeight || 360;
        const aspect = vw > 0 ? vh / vw : 1;
        const p3 = (l: { x: number; y: number; z: number }) => ({
          x: l.x,
          y: l.y * aspect,
          z: l.z ?? 0,
        });

        const sub3 = (a: {x:number,y:number,z:number}, b: {x:number,y:number,z:number}) =>
          ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
        const cross3 = (a: {x:number,y:number,z:number}, b: {x:number,y:number,z:number}) => ({
          x: a.y * b.z - a.z * b.y,
          y: a.z * b.x - a.x * b.z,
          z: a.x * b.y - a.y * b.x,
        });

        // Image axes are x-right, y-DOWN, z-into-screen (MediaPipe: smaller z is
        // nearer the camera). That triple is right-handed, so for a face square to
        // the lens right x up = (0,0,-1) — the normal points back at the camera.
        const rightVec = sub3(p3(rightEye), p3(leftEye));
        const upVec = sub3(p3(forehead), p3(chin));
        const fwd = cross3(rightVec, upVec);
        const fwdLen = Math.sqrt(fwd.x ** 2 + fwd.y ** 2 + fwd.z ** 2);

        const RAD2DEG = 180 / Math.PI;
        const clamp1 = (v: number) => (v < -1 ? -1 : v > 1 ? 1 : v);

        // Sign conventions, chosen to match the existing HEAD_* vocabulary:
        //   pitchDeg > 0  → face normal tilted toward +y (down)  → HEAD_DOWN
        //   yawDeg   > 0  → face normal tilted toward +x (image-right).
        // The feed is not mirrored, so image-right is the subject's OWN left —
        // hence positive yaw is HEAD_LEFT, the same subject-relative naming
        // classifyGaze uses.
        let pitchDeg = 0;
        let yawDeg = 0;
        if (fwdLen > 1e-9) {
          pitchDeg = Math.asin(clamp1(fwd.y / fwdLen)) * RAD2DEG;
          yawDeg = Math.asin(clamp1(fwd.x / fwdLen)) * RAD2DEG;
        }

        // === IRIS MATH ===
        //
        // Both axes measure the iris centre against the EYE CORNERS, normalised by
        // eye width. The corners are skin-anchored and do not move when the eye
        // opens, closes or blinks, which is what makes the signal stable.
        //
        // This replaces two bugs that made the old version report the opposite of
        // what the user was doing:
        //
        // 1. Vertical was measured against the EYELID landmarks (159/386 upper,
        //    145/374 lower) and normalised by the eyelid gap. The upper lid tracks
        //    the gaze — look down and it droops onto the iris — so "distance from
        //    iris to upper lid" SHRANK when looking down and the classifier read it
        //    as looking up. That is the "I looked down, it said tilted up" report.
        //    Normalising by the eyelid gap made it worse: the denominator collapses
        //    toward zero on a blink and the ratio explodes.
        //
        // 2. Horizontal used distance-to-inner minus distance-to-outer per eye. The
        //    inner corner is on opposite sides of the two eyes, so for one physical
        //    look the two eyes produced equal and OPPOSITE values, and averaging
        //    them cancelled to ~0. Horizontal gaze was effectively dead — which is
        //    also why head pose was never vetoed sideways.
        //
        // Measuring a signed offset from the corner midpoint fixes both: the two
        // eyes now agree in sign, and neither axis depends on the eyelid.
        const dist = (a: {x:number,y:number}, b: {x:number,y:number}) =>
          Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);

        // Left Eye Nodes (image-left eye)
        const lIris = primary[468];
        const lInner = primary[133];
        const lOuter = primary[33];
        const lTop = primary[159];
        const lBottom = primary[145];

        // Right Eye Nodes (image-right eye)
        const rIris = primary[473];
        const rInner = primary[362];
        const rOuter = primary[263];
        const rTop = primary[386];
        const rBottom = primary[374];

        const lWidth = dist(lInner, lOuter);
        const rWidth = dist(rInner, rOuter);

        // Guard against a degenerate mesh (width 0 → Infinity poisons the baseline
        // average for the whole session, since calibration takes a plain mean).
        const EYE_MIN_WIDTH = 1e-6;
        const lOk = lWidth > EYE_MIN_WIDTH;
        const rOk = rWidth > EYE_MIN_WIDTH;

        // === YAW-INVARIANT NORMALISER ===
        //
        // Dividing by the CURRENT eye width was the second half of the direction
        // bug. Turn your head and the eyes foreshorten — the corner-to-corner
        // distance can halve — so the same physical iris position divides by a
        // much smaller number and the normalised offset doubles. A plain side
        // turn therefore manufactured a large fake gaze deflection, which then
        // beat head pose in fuseSensors and got reported on whichever axis
        // happened to cross first.
        //
        // Face HEIGHT is the axis a horizontal turn does not compress, so
        // baselineEyeW * (faceH / baselineFaceH) reconstructs "how wide the eye
        // would be if the head were facing forward at this distance". Moving
        // closer/further still scales correctly; turning no longer inflates.
        // Before the baseline locks we have no reference, so we use the live
        // width — during calibration the user is looking straight ahead anyway.
        const faceH = Math.abs(chin.y - forehead.y);
        const bEyeW = calibrationRef.current.baselineEyeW;
        const bFaceH = calibrationRef.current.baselineFaceH;
        const scale =
          bEyeW !== null && bFaceH !== null && bFaceH > 1e-6 && faceH > 1e-6
            ? bEyeW * (faceH / bFaceH)
            : null;
        const lNorm = scale ?? lWidth;
        const rNorm = scale ?? rWidth;

        // Signed offset from the corner midpoint, normalised by eye width.
        // x: positive = iris toward image-right = the subject's OWN left.
        // y: positive = iris below the corner line = looking DOWN.
        const offX = (iris: {x:number}, inner: {x:number}, outer: {x:number}, w: number) =>
          (iris.x - (inner.x + outer.x) / 2) / w;
        const offY = (iris: {y:number}, inner: {y:number}, outer: {y:number}, w: number) =>
          (iris.y - (inner.y + outer.y) / 2) / w;

        const xs: number[] = [];
        const ys: number[] = [];
        if (lOk) { xs.push(offX(lIris, lInner, lOuter, lNorm)); ys.push(offY(lIris, lInner, lOuter, lNorm)); }
        if (rOk) { xs.push(offX(rIris, rInner, rOuter, rNorm)); ys.push(offY(rIris, rInner, rOuter, rNorm)); }

        const currentGazeX = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
        const currentGazeY = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;

        // Eye aspect ratio — lid gap over eye width. Used ONLY to suppress the
        // gaze read during a blink; it is deliberately not part of the gaze
        // signal itself, which is the mistake the old vertical maths made.
        const EAR_OPEN = 0.18;
        const lEAR = lOk ? dist(lTop, lBottom) / lWidth : 0;
        const rEAR = rOk ? dist(rTop, rBottom) / rWidth : 0;
        const openCount = (lEAR > EAR_OPEN ? 1 : 0) + (rEAR > EAR_OPEN ? 1 : 0);
        eyesOpen = openCount > 0;


        // === CALIBRATION GAZE SAMPLES ===
        if (calibrationRef.current.isCalibrating) {
          if (faces === 1) {
            // eyeW/faceH are the head-on reference geometry the yaw-invariant
            // normaliser needs. Only meaningful while the user is facing forward,
            // which is exactly the contract of this calibration window.
            const eyeWObs = lOk && rOk ? (lWidth + rWidth) / 2 : lOk ? lWidth : rOk ? rWidth : 0;
            if (eyeWObs > EYE_MIN_WIDTH && faceH > 1e-6) {
              calibrationRef.current.samples.push({
                pitchDeg,
                yawDeg,
                gazeX: currentGazeX,
                gazeY: currentGazeY,
                eyeW: eyeWObs,
                faceH,
              });
            }
          }

          // Lock baseline after 5 seconds
          if (Date.now() - calibrationRef.current.startTime > 5000) {
            const samples = calibrationRef.current.samples;
            if (samples.length >= 30) {
              const meanPitch = samples.reduce((acc, s) => acc + s.pitchDeg, 0) / samples.length;
              const meanYaw = samples.reduce((acc, s) => acc + s.yawDeg, 0) / samples.length;
              const meanGazeX = samples.reduce((acc, s) => acc + s.gazeX, 0) / samples.length;
              const meanGazeY = samples.reduce((acc, s) => acc + s.gazeY, 0) / samples.length;
              const meanEyeW = samples.reduce((acc, s) => acc + s.eyeW, 0) / samples.length;
              const meanFaceH = samples.reduce((acc, s) => acc + s.faceH, 0) / samples.length;

              calibrationRef.current.baselinePitch = meanPitch;
              calibrationRef.current.baselineYaw = meanYaw;
              calibrationRef.current.baselineGazeX = meanGazeX;
              calibrationRef.current.baselineGazeY = meanGazeY;
              calibrationRef.current.baselineEyeW = meanEyeW;
              calibrationRef.current.baselineFaceH = meanFaceH;

              console.log(`[CALIBRATION] Baseline Locked -> Pitch: ${meanPitch.toFixed(2)}, Yaw: ${meanYaw.toFixed(2)}, GazeX: ${meanGazeX.toFixed(2)}, GazeY: ${meanGazeY.toFixed(2)}, EyeW: ${meanEyeW.toFixed(4)}, FaceH: ${meanFaceH.toFixed(4)}`);
            } else {
              console.warn(`[CALIBRATION] Failed to lock baseline. Only ${samples.length} valid samples collected. Falling back to hardcoded heuristics.`);
            }
            calibrationRef.current.isCalibrating = false;
            setIsCalibrating(false);
          }
        }

        // Determine Head Pose
        const headPose = classifyPose(pitchDeg, yawDeg, calibrationRef.current.baselinePitch, calibrationRef.current.baselineYaw);
        
        // Determine Gaze Pose
        const gazePose = classifyGaze(
          currentGazeX,
          currentGazeY,
          calibrationRef.current.baselineGazeX,
          calibrationRef.current.baselineGazeY,
          // Both halves of "usable": lids far enough apart to see the iris, and at
          // least one eye with a non-degenerate width. With neither eye measurable
          // currentGazeX/Y fall back to 0, which is not a centred gaze — it is the
          // absence of one, and against a non-zero baseline it reads as a deflection.
          eyesOpen && xs.length > 0,
        );


        // Fuse Sensors. The magnitude is what lets a strong deflection outrank a
        // centred-iris veto; without it every look-away below HEAD_STRONG_DEG —
        // and, since an absent third argument compares false against it, every
        // look-away above it too — collapsed back to HEAD_CENTER.
        const headMagDeg = poseMagnitudeDeg(
          pitchDeg,
          yawDeg,
          calibrationRef.current.baselinePitch,
          calibrationRef.current.baselineYaw,
        );
        pose = fuseSensors(headPose, gazePose, headMagDeg);
        gazeVector = [currentGazeX, currentGazeY];

        // Keep what each sensor said BEFORE fusion. fuseSensors is the paper's
        // central mechanism and its inputs were discarded here, so no recorded
        // session could show whether the veto changed the outcome on a real
        // person — only the scripted harness could, by calling it directly.
        headPoseRaw = headPose;
        gazeClass = gazePose;

        // --- 4. Draw Overlay ---
        if (overlayRef.current && videoRef.current) {
          const canvas = overlayRef.current;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            if (canvas.width !== videoRef.current.videoWidth) {
              canvas.width = videoRef.current.videoWidth;
              canvas.height = videoRef.current.videoHeight;
            }
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = "#00FF9C"; // Cyber Green Gaze
            ctx.beginPath();
            ctx.arc(lIris.x * canvas.width, lIris.y * canvas.height, 2, 0, 2 * Math.PI);
            ctx.arc(rIris.x * canvas.width, rIris.y * canvas.height, 2, 0, 2 * Math.PI);
            ctx.fill();

            if (Date.now() - propBoxesRef.current.at < PROP_BOX_TTL_MS) {
              drawPropBoxes(ctx, canvas, propBoxesRef.current.boxes);
            }
          }
        }
      } else {
        if (overlayRef.current) {
          const canvas = overlayRef.current;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Boxes survive the no-face branch on purpose. A candidate holding a
            // phone below the camera, ducking out of frame, or turned fully away
            // is exactly when MediaPipe loses the face and exactly when seeing
            // what YOLO boxed matters most. Clearing them here would blank the
            // overlay in the one situation it was built for.
            if (Date.now() - propBoxesRef.current.at < PROP_BOX_TTL_MS) {
              drawPropBoxes(ctx, canvas, propBoxesRef.current.boxes);
            }
          }
        }
      }

      const now = Date.now();
      telemetryRef.current = {
        faces_detected: faces,
        is_talking: talking,
        head_pose: pose,
        gaze_vector: gazeVector,
        stable_pose: stablePose(pose, now),
        // Same modal smoothing, one window each, so all three channels are
        // measured the same way.
        stable_head_pose_raw: stableOver(headWindowRef, headPoseRaw, now),
        stable_gaze_class: stableOver(gazeWindowRef, gazeClass, now),
        // Modal eye state across the trailing window. stableOver returns null
        // until the window agrees, so "not yet decided" and "OPEN" both read as
        // open — only a sustained, corroborated closure reports eyes shut.
        eyes_open: stableOver(eyesWindowRef, eyesOpen ? "OPEN" : "CLOSED", now) !== "CLOSED",
      };
    });

  let isRunning = true;
  let lastVideoTime = -1;

  const onFrame = async () => {
    if (!isRunning || !videoEl) return;
    if (videoEl.readyState >= 3 && videoEl.currentTime !== lastVideoTime) { // readyState 3 is 'HAVE_FUTURE_DATA'
      lastVideoTime = videoEl.currentTime;
      await faceMesh.send({ image: videoEl });
    }
    requestAnimationFrame(onFrame);
  };
  
  videoEl.addEventListener('play', () => {
    requestAnimationFrame(onFrame);
  });

  return { 
    faceMesh,
    stop: () => { isRunning = false; }
  };
};

  // --- 3. OPTICS INITIALIZATION ---
  const startCamera = async () => {
    try {
      // Hardware tripwire: abort if virtual camera is found before any feed opens
      const hw = await checkHardware();
      if (hw.compromised) {
        setSysStatus("COMPROMISED");
        setLatestVerdict(`🚨 HARDWARE TRIPWIRE: Virtual camera detected (${hw.label}). Session blocked.`);
        showToast(`BLOCKED: ${hw.label?.toUpperCase()} DETECTED`, "error");
        // Notify parent — this is a session-fatal event
        onTelemetryUpdateRef.current({
          candidate_id: CANDIDATE_ID,
          risk_score: 100,
          violation_count: 0,
          critical_flags: [`Virtual camera: ${hw.label}`],
          intervention_level: "SEVERE_VIOLATION_LOGGED"
        }, "Session blocked: virtual camera detected.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsScanning(true);
      setSysStatus("ACTIVE");
      
      // Reset Calibration State on start
      calibrationRef.current = {
        isCalibrating: true,
        samples: [],
        baselinePitch: null,
        baselineYaw: null,
        baselineGazeX: null,
        baselineGazeY: null,
        baselineEyeW: null,
        baselineFaceH: null,
        startTime: Date.now()
      };
      setIsCalibrating(true);
      
      setLatestVerdict("OPTICS ONLINE. CALIBRATING BASELINE...");

      if (videoRef.current && !gatekeeperRef.current && isFaceMeshReady) {
         // Start the 30fps Gatekeeper
         gatekeeperRef.current = initGatekeeper(videoRef.current) as any;
      }
      // The post-mount useEffect picks up isScanning=true and starts the inference
      // loop. Calling it here too would double-start parallel chains.
    } catch {
      console.warn("Camera access denied.");
      setSysStatus("COMPROMISED");
      setLatestVerdict("ERROR: OPTICS UNAVAILABLE. CHECK PERMISSIONS.");
    }
  };

  const showToast = (message: string, type: "success" | "error") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  const handleReset = async () => {
    if (riskScore === 0 && peakRisk === 0 && violationCount === 0 && interventionLevel === "CLEAR") {
      showToast("NO DATA TO CLEAR!", "error");
      return;
    }

    try {
      const response = await fetch("http://localhost:8080/reset-session?candidate_id=" + CANDIDATE_ID, {
        method: "POST"
      });
      if (response.ok) {
        setRiskScore(0);
        setPeakRisk(0);
        setViolationCount(0);
        setInterventionLevel("CLEAR");
        setLatestVerdict("SYSTEM STANDBY");
        setFlags([]);
        // Reset calibration on manual clear
        calibrationRef.current = {
          isCalibrating: true,
          samples: [],
          baselinePitch: null,
          baselineYaw: null,
          baselineGazeX: null,
          baselineGazeY: null,
          baselineEyeW: null,
          baselineFaceH: null,
          startTime: Date.now()
        };
        setIsCalibrating(true);
        showToast("MEMORY CLEARED SUCCESSFULLY.", "success");
        console.log("Memory reset successfully.");
        // Ensure parent component un-flags warning too
        onTelemetryUpdateRef.current({
          candidate_id: CANDIDATE_ID,
          risk_score: 0,
          violation_count: 0,
          intervention_level: "CLEAR"
        }, "SYSTEM STANDBY");
      }
    } catch {
      console.warn("Failed to reset session.");
    }
  };

  const stopCamera = () => {
    if (gatekeeperRef.current) {
      // @ts-ignore
      gatekeeperRef.current.stop();
      if (gatekeeperRef.current.faceMesh) {
        gatekeeperRef.current.faceMesh.close();
      }
      gatekeeperRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      // Detach the spent stream. Stopping tracks leaves them in "ended" state
      // but still bound to the element, so a re-engage would attach a fresh
      // stream on top of a dead one and the video never produced frames —
      // MediaPipe then had nothing to read and the interview never restarted.
      videoRef.current.srcObject = null;
    }
    if (loopTimerRef.current) clearTimeout(loopTimerRef.current);
    loopRunningRef.current = false;
    // Frames from the run that just ended must not corroborate frames from the
    // next one — the pose window is trailing, so without this the first reading
    // after a re-engage is a mix of two sessions' geometry.
    poseWindowRef.current = [];
    // Reset calibration so a re-engage recalibrates against a fresh baseline
    // instead of inheriting the previous run's head-pose reference.
    calibrationRef.current.isCalibrating = false;
    calibrationRef.current.samples = [];
    calibrationRef.current.baselinePitch = null;
    calibrationRef.current.baselineYaw = null;
    calibrationRef.current.baselineGazeX = null;
    calibrationRef.current.baselineGazeY = null;
    calibrationRef.current.baselineEyeW = null;
    calibrationRef.current.baselineFaceH = null;
    setIsCalibrating(false);
    setIsScanning(false);
    setSysStatus("IDLE");
    onDisengage?.();
  };

  // Expose stopCamera so the parent can auto-disengage on "End session".
  // Kept in a ref: an empty-dep useImperativeHandle froze the first render's
  // stopCamera, which closed over a stale onDisengage.
  const stopCameraRef = useRef(stopCamera);
  stopCameraRef.current = stopCamera;
  useImperativeHandle(ref, () => ({ stopCamera: () => stopCameraRef.current() }), []);

  // --- 4. THE SELF-HEALING ASYNC LOOP ---
  // useCallback with no deps + refs for changing values keeps this function reference
  // stable for the component's lifetime. This is critical: if the reference changed
  // on parent re-renders, the scheduling useEffect would re-fire and spawn parallel
  // chains, causing the cadence pile-up we observed in the DB.
  const runInferenceLoop = useCallback(async () => {
    if (!loopRunningRef.current) return; // chain was cancelled (disengage / unmount)
    if (!videoRef.current || !canvasRef.current) {
      loopTimerRef.current = setTimeout(runInferenceLoop, TELEMETRY_INTERVAL_MS);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // readyState >= 3 means we have data for the current and future frames
    if (ctx && video.readyState >= 3) {
      // Scale down the whole image to save bandwidth without losing context
      const targetWidth = 640;
      const targetHeight = 360;

      canvas.width = targetWidth;
      canvas.height = targetHeight;
      ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, targetWidth, targetHeight);

      const base64Image = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

      // The gatekeeper writes per-frame, the backend only ever hears the 5s loop.
      // Forward the modal pose across the trailing window — not the last frame —
      // so a single jittery or blinking frame cannot become the whole verdict.
      // When the window has not agreed, there is no evidence of anything; report
      // a centred pose rather than whichever noise happened to come last.
      const pose = telemetryRef.current.stable_pose ?? "HEAD_CENTER";
      // Same fallback rule for the pre-fusion channels: an unsettled window is
      // an absence of evidence, so report each classifier's no-opinion value
      // rather than whichever noise arrived last.
      const headPoseRaw = telemetryRef.current.stable_head_pose_raw ?? "HEAD_CENTER";
      const gazeClass = telemetryRef.current.stable_gaze_class ?? "unknown";

      console.log("🚀 [FRONTEND] OUTGOING PAYLOAD:", {
        faces_detected: telemetryRef.current.faces_detected,
        is_talking: telemetryRef.current.is_talking,
        head_pose: pose,
        head_pose_raw: headPoseRaw,
        gaze_class: gazeClass,
        gaze_vector: telemetryRef.current.gaze_vector,
        eyes_open: telemetryRef.current.eyes_open,
      });

      if (calibrationRef.current.isCalibrating) {
        // Skip payload POST and wait for baseline lock
        if (loopRunningRef.current) loopTimerRef.current = setTimeout(runInferenceLoop, TELEMETRY_INTERVAL_MS);
        return;
      }

      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000);

        const response = await fetch("http://localhost:8080/api/v1/analyze-frame", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            candidate_id: CANDIDATE_ID,
            session_id: sessionIdRef.current,
            timestamp: Date.now(),
            image_base64: base64Image,
            faces_detected: telemetryRef.current.faces_detected,
            is_talking: telemetryRef.current.is_talking,
            head_pose: pose,
            head_pose_raw: headPoseRaw,
            gaze_class: gazeClass,
            gaze_vector: telemetryRef.current.gaze_vector,
            eyes_open: telemetryRef.current.eyes_open
          })
        });
        clearTimeout(timeoutId);


        const data = await response.json();

        if (data.risk_packet) {
          setRiskScore(data.risk_packet.risk_score);
          setPeakRisk((p) => Math.max(p, data.risk_packet.peak_risk ?? data.risk_packet.risk_score));
          setViolationCount(data.risk_packet.violation_count);
          setInterventionLevel(data.risk_packet.intervention_level);
          setLatestVerdict(data.verdict);
          setFlags(Array.isArray(data.flags) ? data.flags : []);

          // Pass data up via stable ref — never re-binds the loop.
          onTelemetryUpdateRef.current(data.risk_packet, data.verdict);
        }

        // Outside the risk_packet guard on purpose. An interruption is the one
        // thing here that must not be lost to a malformed packet — it is the only
        // moment the candidate can still act on the finding.
        if (data.interrupt) onInterruptRef.current?.(data.interrupt);
      } catch (error) {
        console.log("Backend unreachable:", (error as Error).message);
        setLatestVerdict("⚠️ ERROR: BACKEND CONNECTION FAILED. CHECK IF EDGE_MAIN.PY IS RUNNING.");
      }
    }

    // Schedule next tick only if the chain is still alive.
    if (loopRunningRef.current) {
      loopTimerRef.current = setTimeout(runInferenceLoop, TELEMETRY_INTERVAL_MS);
    }
  }, []);

  useEffect(() => {
    // Single-flight: only one chain can ever be active at a time.
    if (isScanning && isFaceMeshReady && !loopRunningRef.current) {
      if (videoRef.current && !gatekeeperRef.current) {
        gatekeeperRef.current = initGatekeeper(videoRef.current) as any;
      }
      loopRunningRef.current = true;
      runInferenceLoop();
    }
    return () => {
      // Cancel chain on unmount or when isScanning flips false.
      loopRunningRef.current = false;
      if (loopTimerRef.current) {
        clearTimeout(loopTimerRef.current);
        loopTimerRef.current = null;
      }
    };
  }, [isScanning, isFaceMeshReady, runInferenceLoop]);


  // ── Fast prop sweep (independent of the telemetry chain) ─────────────────
  // This YOLO-only endpoint writes no timeline frames, so object detection can
  // keep its own cadence and rising-edge contract.
  const propCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const propTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const propRunningRef = useRef(false);

  const runPropScanLoop = useCallback(async () => {
    const PROP_SCAN_MS = 1200;
    // Must be > PROP_SCAN_MS. These were previously the same value, which meant
    // any YOLO inference slower than one scan interval aborted its own request
    // before the answer came back. On CPU, YOLOv8s at imgsz=640 routinely takes
    // 1.5-3s, so every sweep aborted, the catch below swallowed it silently,
    // and a phone was only caught by the 5s fallback loop — the exact "used to
    // flag instantly, now takes several seconds" regression.
    const PROP_TIMEOUT_MS = 5000;
    // Floor between sweeps. Deadline scheduling (below) can drive the computed
    // delay to zero when inference overruns the window; this keeps a slow backend
    // from being hammered back-to-back with no breathing room.
    const PROP_MIN_GAP_MS = 150;
    if (!propRunningRef.current) return;

    const tickStart = Date.now();

    const video = videoRef.current;
    // Deliberately NOT gated on calibration. The 5s baseline window exists for
    // head-pose maths; YOLO does not consume it. Waiting for it meant a phone
    // already on the desk at Engage went unseen for the first five seconds.
    if (video && video.readyState >= 3) {
      // Own canvas — sharing canvasRef would race with the telemetry loop and
      // one could ship a half-drawn frame.
      if (!propCanvasRef.current) propCanvasRef.current = document.createElement("canvas");
      const canvas = propCanvasRef.current;
      const ctx = canvas.getContext("2d");

      if (ctx) {
        canvas.width = 640;
        canvas.height = 360;
        ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 640, 360);
        // 0.8 quality, matching the main loop. 0.6 was chosen to keep 4x the
        // cadence from costing 4x the bandwidth — but it smeared a dark phone on
        // a dark shirt below YOLO's confidence floor (measured 0.185 vs a 0.20
        // gate at q0.6; 0.258 at q0.8), turning a real detection into a miss that
        // read as a 5s delay. At 640x360 the extra bytes are worth the recall.
        const b64 = canvas.toDataURL("image/jpeg", 0.8).split(",")[1];

        try {
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), PROP_TIMEOUT_MS);
          const res = await fetch("http://localhost:8080/api/v1/scan-objects", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
              candidate_id: CANDIDATE_ID,
              session_id: sessionIdRef.current,
              image_base64: b64,
            }),
          });
          clearTimeout(timeoutId);
          const data = await res.json();

          // Set on EVERY sweep, unlike the risk fields below, which are gated on
          // the rising edge. The boxes are an observation, not a verdict: they
          // must keep updating while a banner is already up (so you can watch
          // what is holding it up) and they must go empty the moment a sweep
          // finds nothing, or a stale box would read as a live detection.
          propBoxesRef.current = { at: Date.now(), boxes: data.boxes ?? [] };

          // Only paint on a real escalation. The backend returns the rising
          // edge only, so a prop sitting in frame reports escalated=false on
          // every subsequent sweep — repainting from those would stomp the
          // verdict the full telemetry loop owns.
          if (data.escalated && data.risk_packet) {
            setRiskScore(data.risk_packet.risk_score);
            setPeakRisk((p) => Math.max(p, data.risk_packet.peak_risk ?? data.risk_packet.risk_score));
            setViolationCount(data.risk_packet.violation_count);
            setInterventionLevel(data.risk_packet.intervention_level);
            setLatestVerdict(data.verdict);
            // The object sweep reports one prop and returns no flags array. Clear
            // the telemetry flags rather than leaving stale lines from an earlier
            // frame sitting under a fresh prop verdict.
            setFlags([]);
            onTelemetryUpdateRef.current(data.risk_packet, data.verdict);
          }

          // Outside the `escalated` guard for the same reason as above: the
          // backend only sets this on a confirmed rising edge, so it is already
          // gated where the gating belongs. Re-gating it here on a different
          // condition would risk swallowing the one thing the candidate needs to
          // hear while they can still do something about it.
          if (data.interrupt) onInterruptRef.current?.(data.interrupt);
        } catch {
          // Stay silent — the telemetry loop already surfaces backend-down.
          // Two error banners fighting over the same panel helps nobody.
        }
      }
    }

    if (propRunningRef.current) {
      // Cadence is measured from the START of this sweep, not from its end.
      //
      // This is the "takes ~3s to see a phone" bug. The delay used to be appended
      // AFTER the fetch resolved, so the real period was PROP_SCAN_MS *plus* the
      // full round trip. YOLOv8s on CPU runs 1.5-3s per frame, which put the true
      // sweep interval at 3-4s and worst-case detection latency at nearly double
      // the 1.2s the status chip advertises.
      //
      // Timing from tickStart makes the interval a deadline rather than a gap:
      // when inference fits inside the window the cadence is a genuine 1.2s, and
      // when it overruns the next sweep starts as soon as PROP_MIN_GAP_MS allows
      // instead of idling for another 1.2s on top.
      const elapsed = Date.now() - tickStart;
      propTimerRef.current = setTimeout(
        runPropScanLoop,
        Math.max(PROP_MIN_GAP_MS, PROP_SCAN_MS - elapsed),
      );
    }
  }, []);

  useEffect(() => {
    if (isScanning && isFaceMeshReady && !propRunningRef.current) {
      propRunningRef.current = true;
      runPropScanLoop();
    }
    return () => {
      propRunningRef.current = false;
      if (propTimerRef.current) {
        clearTimeout(propTimerRef.current);
        propTimerRef.current = null;
      }
    };
  }, [isScanning, isFaceMeshReady, runPropScanLoop]);


  // ── Severity-colour helpers (single source of truth for the panel) ──────

  const tierToken =
    interventionLevel === "SEVERE_VIOLATION_LOGGED" ? "danger" :
    interventionLevel === "HARD_WARNING"            ? "amber"  :
    interventionLevel === "WARNING_LOGGED"          ? "amber"  :
    interventionLevel === "SOFT_WARNING"            ? "warn"   :
    "clear";

  const tierStyles: Record<string, string> = {
    clear:  "border-[var(--color-hairline)] bg-[var(--color-surface-2)] text-[var(--color-slate)]",
    warn:   "border-[var(--color-warn)]/40 bg-[var(--color-warn)]/[0.06] text-[var(--color-warn)]",
    amber:  "border-[var(--color-amber)]/40 bg-[var(--color-amber)]/[0.06] text-[var(--color-amber)]",
    danger: "border-[var(--color-danger)]/45 bg-[var(--color-danger)]/[0.07] text-[var(--color-danger)]",
  };

  // Keyed to the peak, because it tints the peak figure. Tinting the session
  // high-water mark by the live reading made a 100% session render in plain white
  // the instant the candidate looked back.
  const riskColor =
    peakRisk >= 80 ? "text-[var(--color-danger)]" :
    peakRisk >= 40 ? "text-[var(--color-amber)]"  :
    peakRisk >= 20 ? "text-[var(--color-warn)]"   :
    "text-[var(--color-snow)]";

  return (
    <div className="w-full h-full min-h-0 flex flex-col gap-3 relative">

      {/* Toast — minimal, hairline-bordered, no neon glow spam */}
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            role="status"
            className={`absolute top-0 right-0 z-50 flex items-center gap-2.5 px-3.5 py-2.5 min-w-[260px] rounded-lg border text-[12px] font-medium backdrop-blur-md ${
              toast.type === "error"
                ? "bg-[var(--color-surface)]/90 border-[var(--color-danger)]/40 text-[var(--color-danger)]"
                : "bg-[var(--color-surface)]/90 border-[var(--color-signal)]/40 text-[var(--color-signal)]"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                toast.type === "error" ? "bg-[var(--color-danger)]" : "bg-[var(--color-signal)]"
              }`}
            />
            <span>{toast.message}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* LEFT — Camera frame.
          aspect-video (16:9) matches the 1280x720 camera exactly, so the
          object-cover below is a no-op and the preview shows the SAME pixels
          the backend analyses. A fixed height (was h-[380px]) rendered a ~2.49
          box, and object-cover then cropped ~103px off the top and bottom —
          hiding a band that YOLO still received, so a phone held low could be
          flagged with nothing on screen to explain it. Keep the ratio: the
          evidence frames on /replay and /report are this same canvas, so a
          preview of a different shape shows the candidate less than the report
          later holds against them.

          Height-driven, not width-driven. `w-full aspect-video` derived height
          from the column width, so a wide-but-short window (1531x730 on a 125%
          display) made this 542px tall inside ~600px of usable height and the
          telemetry strip below was clipped. flex-1 min-h-0 takes the leftover
          height instead and the width follows from the ratio. */}
      <div
        className={`relative h-full aspect-video max-w-full mx-auto rounded-lg overflow-hidden flex-1 min-h-0 lift-1 haze transition-shadow ${
          sysStatus === "ACTIVE" ? "ring-signal" : ""
        }`}
      >
        <canvas ref={canvasRef} className="hidden" />

        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />

        <canvas
          ref={overlayRef}
          className="absolute inset-0 w-full h-full pointer-events-none object-cover"
        />

        {/* Calibration Overlay */}
        {isScanning && isCalibrating && (
          <div className="absolute inset-0 dot-grid flex items-center justify-center text-center p-6 bg-[var(--color-surface)]/60 backdrop-blur-sm z-10 transition-opacity">
            <div className="terminal-slab w-full max-w-[560px] p-6 text-center haze">
              <span className="flex items-center justify-center gap-2 mb-3">
                <span className="w-2 h-2 rounded-full bg-[var(--color-signal)] pulse-signal" />
                <span className="eyebrow text-[var(--color-signal)]">Zero-Hour Calibration</span>
              </span>
              <div className="font-mono text-[15px] leading-relaxed text-[var(--color-slate)]">
                Calibrating Posture Baseline...<br/>
                Please look naturally at the screen.
              </div>
            </div>
          </div>
        )}

        {/* Idle-state placeholder grid (only when no stream) */}
        {!isScanning && (
          <div className="absolute inset-0 dot-grid flex items-center justify-center text-center p-6">
            {/* Sized for legibility in figures: this panel is the shot that
                ends up in the paper, where the whole frame gets scaled to a
                column width. Undersized type here becomes unreadable there. */}
            <div className="terminal-slab w-full max-w-[560px] p-6 text-left haze">
              <div className="flex items-center justify-between mb-4">
                <span className="eyebrow">Optics offline</span>
                <span className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-[var(--color-danger)]" />
                  <span className="w-2 h-2 rounded-full bg-[var(--color-warn)]" />
                  <span className="w-2 h-2 rounded-full bg-[var(--color-signal)]" />
                </span>
              </div>
              <div className="font-mono text-[15px] leading-relaxed">

                <div className="text-[var(--color-slate)]">&gt; load gatekeeper.face_mesh</div>
                <div className="text-[var(--color-slate)]">&gt; attach camera.stream</div>
                <div className="text-[var(--color-signal)]">&gt; ready: local inference only</div>

                {/* Pre-engage persona pick. Choosing before Engage is the whole
                    point — the ladder is seeded at session start, so a pick made
                    mid-run would be ignored. Locks while scanning. */}
                <div className="mt-4 border-t border-[var(--color-hairline)] pt-4">
                  <div className="flex items-baseline justify-between mb-2.5">
                    <span className="text-[12px] font-semibold text-[var(--color-snow)]">
                      Choose your interviewer
                    </span>
                    <span className="text-[10px] text-[var(--color-slate)]">
                      Difficulty still ramps up as you go
                    </span>
                  </div>
                  <PersonaPicker
                    value={persona}
                    onChange={(id) => {
                      setPersona(id);
                      onPersonaChange?.(id);
                    }}
                    compact
                  />
                </div>

                <div className="text-[var(--color-parchment)] mt-4">Press Engage sentry to begin behavioural analysis.</div>
              </div>
            </div>
          </div>
        )}

        {/* Status chip — top-left */}
        <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
          <span
            className={`flex items-center gap-2 px-2.5 py-1 rounded-md text-[11px] font-medium uppercase backdrop-blur-md border ${
              sysStatus === "ACTIVE"
                ? "bg-[var(--color-signal)]/10 border-[var(--color-signal)]/40 text-[var(--color-signal)]"
                : sysStatus === "COMPROMISED"
                ? "bg-[var(--color-danger)]/10 border-[var(--color-danger)]/40 text-[var(--color-danger)]"
                : "bg-[var(--color-surface)]/70 border-[var(--color-hairline)] text-[var(--color-slate)]"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                sysStatus === "ACTIVE"
                  ? "bg-[var(--color-signal)] pulse-signal"
                  : sysStatus === "COMPROMISED"
                  ? "bg-[var(--color-danger)] pulse-danger"
                  : "bg-[var(--color-fog)]"
              }`}
            />
            {sysStatus === "ACTIVE" ? "Live" : sysStatus === "COMPROMISED" ? "Blocked" : "Idle"}
          </span>

          {sysStatus === "ACTIVE" && (
            <span className="px-2.5 py-1 rounded-md text-[11px] font-medium uppercase backdrop-blur-md border bg-[var(--color-surface)]/70 border-[var(--color-hairline)] text-[var(--color-slate)]">
              2s composure · 1.2s objects
            </span>

          )}
        </div>

        {/* Controls — bottom-right */}
        <div className="absolute bottom-3 right-3 z-20 flex gap-2">
          {!isScanning ? (
            <button
              onClick={() => {
                onEngageClick?.();
                void startCamera();
              }}
              className="px-4 h-9 rounded-md bg-[var(--color-signal)]/15 text-[var(--color-signal)] text-[13px] font-medium border border-[var(--color-signal)]/40 hover:bg-[var(--color-signal)]/25 hover:border-[var(--color-signal)] transition-colors cursor-pointer backdrop-blur-md"
            >
              Engage sentry
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="px-4 h-9 rounded-md bg-[var(--color-danger)]/10 backdrop-blur-md border border-[var(--color-danger)]/30 text-[var(--color-danger)] text-[13px] font-medium hover:bg-[var(--color-danger)]/20 hover:border-[var(--color-danger)]/60 transition-colors cursor-pointer"
            >
              Disengage
            </button>
          )}
        </div>
      </div>

      {/* Bottom — Telemetry strip */}
      <div className="w-full flex-shrink-0 flex flex-col sm:flex-row gap-3">

        {/* BEA Temporal panel — sized to its content. It holds four short,
            fixed-width readouts, so giving it the leftover space just pads
            the middle; the log next to it carries a full sentence and is the
            one that actually needs the width. */}
        <div className="lift-1 rounded-lg p-3 flex-shrink-0">
          <div className="flex justify-between items-center mb-2">
            <span className="eyebrow flex items-center gap-2">
              <span className="w-1 h-1 rounded-full bg-[var(--color-fog)]" />
              BEA · Temporal
            </span>
            <button
              onClick={handleReset}
              className="h-6 px-2 rounded-md text-[11px] font-medium text-[var(--color-warn)]/70 border border-[var(--color-warn)]/20 bg-transparent hover:text-[var(--color-warn)] hover:border-[var(--color-warn)]/50 transition-colors cursor-pointer"

            >
              Clear memory
            </button>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <span className="eyebrow mb-0.5">Cumulative risk</span>
              <h2
                data-text={`${peakRisk}%`}
                data-active={peakRisk === 100 ? "true" : "false"}
                className={`font-display text-[28px] leading-none tabular font-semibold tracking-tight transition-colors duration-500 ${riskColor} ${
                  peakRisk === 100 ? "glitch-text" : ""
                }`}
              >
                {peakRisk}
                <span className="text-[var(--color-fog)] text-[18px] font-medium ml-0.5">%</span>
              </h2>
              {/* The live reading, only while it differs. "Cumulative" has to mean
                  cumulative — it used to show the live value, so it fell back to 0
                  on recovery and the label was a lie. Showing both keeps the
                  present tense visible without letting it erase the session. */}
              {riskScore !== peakRisk && (
                <span className="text-[10px] text-[var(--color-fog)] tabular mt-0.5">
                  now {riskScore}%
                </span>
              )}
            </div>

            <div className={`px-2.5 py-1.5 rounded-lg border flex items-center gap-2 transition-colors duration-500 ${tierStyles[tierToken]}`}>
              <div className="flex flex-col gap-0.5">
                <span className="eyebrow opacity-80">Tier</span>
                <span className="text-[11px] font-medium">{interventionLevel}</span>
              </div>
              {interventionLevel !== "CLEAR" && (
                <span className={`w-1.5 h-1.5 rounded-full bg-current ${
                  tierToken === "danger" ? "pulse-danger" : ""
                }`} />
              )}
            </div>

            <div className="flex flex-col gap-0.5">
              <span className="eyebrow">Violations</span>
              <span className="text-[14px] font-semibold tabular text-[var(--color-snow)]">{violationCount}</span>
            </div>
            <div className="flex flex-col gap-0.5">
              <span className="eyebrow">Window</span>
              <span className="text-[14px] font-semibold tabular text-[var(--color-parchment)]">5m</span>
            </div>
          </div>
        </div>

        {/* Behavioral Log — takes the remaining width. Verdict strings run
            long ("BACKEND CONNECTION FAILED. CHECK IF EDGE_MAIN.PY..."), and
            at 200px they wrapped to four cramped lines. */}
        <div className="lift-1 rounded-lg px-3 py-2.5 flex flex-col justify-center flex-1 min-w-0">

          <span className="eyebrow mb-1 flex items-center gap-2">
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                sysStatus === "ACTIVE" ? "bg-[var(--color-signal)] pulse-signal" : "bg-[var(--color-fog)]"
              }`}
            />
            Behavioral Log
          </span>
          {/* Colour comes from the tier, NOT from sniffing the verdict text.
              Sniffing for "CRITICAL" meant the log painted danger-red while the
              Tier badge beside it still read HARD_WARNING — two independent
              sources for one piece of state, guaranteed to disagree.

              One line per flag. The backend used to compute a single verdict with
              an if/elif chain, so a phone in shot silenced the head turn happening
              in the same instant — the box showed one finding because only one was
              ever calculated. It now returns every concurrent flag, and each gets
              its own line so nothing is hidden behind something graver. */}
          {flags.length > 0 ? (
            <div className="flex flex-col gap-1">
              {flags.map((f, i) => (
                <p
                  key={`${f.kind}-${i}`}
                  className={`font-mono text-[13px] leading-snug ${
                    f.critical
                      ? "text-[var(--color-danger)]"
                      : "text-[var(--color-amber)]"
                  }`}
                >
                  <span className="text-[var(--color-fog)] mr-2">&gt;</span>
                  {f.text}
                </p>
              ))}
            </div>
          ) : (
            <p
              className={`font-mono text-[13px] leading-snug ${
                tierToken === "danger"
                  ? "text-[var(--color-danger)]"
                  : tierToken === "amber"
                  ? "text-[var(--color-amber)]"
                  : "text-[var(--color-parchment)]"
              }`}
            >
              <span className="text-[var(--color-fog)] mr-2">&gt;</span>
              {latestVerdict}
            </p>
          )}
        </div>

      </div>
    </div>
  );
}
