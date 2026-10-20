"""One place that decides which LLM backend this process talks to.

Ollama speaks the OpenAI wire protocol on `/v1`, so "local vs cloud" is a
base_url + api_key + model swap — not a second client library and not a second
code path. Every call site imports `make_client()` and `chat_model()` from here
so the two backends cannot drift apart.

Three modes, via `LLM_MODE`:

    auto     probe for a local Ollama; use it if it answers, else cloud (default)
    ollama   force local — fail loudly rather than silently phoning home
    nvidia   force the NVIDIA NIM cloud endpoint

`auto` is the default on purpose. A first-time user who installs Ollama gets
Sovereign Mode without editing a config file, and one who has not installed it
still gets a working app. The alternative — defaulting to `nvidia` — means the
local path only ever runs for people who already knew it existed.
"""
import os
import re
import json
import socket
import logging
from pathlib import Path
from urllib.parse import urlparse

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# PRIMARY. Benchmarked 2026-10 by replaying the app's exact requests (question
# plan, opening, three turns, final report) against a real resume, with
# thinking switched off (see extra_body_for). lightning-30b was the only model
# to pass all 12 calls, and the fastest: 0.5-0.8 s per spoken reply, ~2 s for
# the question plan, ~1.5 s for the report. gpt-oss-20b — an earlier primary —
# timed out in Phase 1 and returned ~23 s TTFT in Phase 2, making it unusable
# for a live interview where the user is waiting.
NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

# Fallbacks on the SAME vendor, tried in order after the primary fails.
# Same benchmark, same thinking-off setting:
#   #1 fallback — super-120b: 10/12, 0.7-1.7 s replies, 5-8 s report; twice
#                             opened without asking a question
#   #2 fallback — ultra-550b: 11/12, 1-6 s replies, 2-3 s report
NVIDIA_FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_FALLBACK_MODEL_2 = "nvidia/nemotron-3-ultra-550b-a55b"

# ── Cloud providers ──────────────────────────────────────────────────────
# All of these speak the OpenAI wire protocol, so supporting one more is a row
# in this table rather than a code path. Both listed here are free to start and
# need no card — the difference is which console the user finds easier.
#
# `env_key` is the variable each provider's key lives under in backend/.env,
# kept distinct so a user can hold keys for both and switch by flipping
# LLM_PROVIDER without re-pasting anything.
#
# `console` must point at a *key management* page, not a model page. It used to
# link to build.nvidia.com/mistralai/mistral-nemotron, which went stale the
# moment the model changed — and sent people to a page for a model this app no
# longer calls. The account-level keys page is model-agnostic, so it survives
# every future swap of NVIDIA_MODEL and matches what Groq's link already does.
CLOUD_PROVIDERS = {
    "nvidia": {
        "label": "NVIDIA NIM",
        "base_url": NVIDIA_BASE_URL,
        "model": NVIDIA_MODEL,
        # Same-vendor hot standbys, tried in order after `model` when it fails
        # and before the cascade changes clouds. See cloud_models_for().
        # super-120b, then ultra-550b — see the constants above.
        "fallbacks": [NVIDIA_FALLBACK_MODEL, NVIDIA_FALLBACK_MODEL_2],
        "env_key": "NVIDIA_API_KEY",
        "prefix": "nvapi-",
        "console": "https://build.nvidia.com/settings/api-keys",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        # llama-3.3-70b-versatile: current live model on Groq free tier (2026-10)
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "prefix": "gsk_",
        "console": "https://console.groq.com/keys",
    },
    # Free Cloudflare Workers endpoint backed by Llama 8B — no API key needed.
    # Not OpenAI-compatible: POST /chat {messages:[...]} -> {answer:"..."}.
    # Benchmarked 2026-10: ~0.9-1.3 s full round-trip (non-streaming).
    # Used as a last-resort free fallback after NVIDIA and Groq.
    "custom_llama": {
        "label": "Llama 8B (free)",
        "base_url": "https://interview-ai-api.mmnirupam.workers.dev",
        "model": "llama-3.1-8b",
        "env_key": "",          # no key needed
        "prefix": "",
        "console": "https://interview-ai-api.mmnirupam.workers.dev",
        "non_openai": True,     # signals engine to use direct HTTP, not SDK
    },
}

DEFAULT_CLOUD_PROVIDER = "nvidia"

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# Small enough to run on a laptop without a discrete GPU, and unusually good at
# holding a format for its size — which matters because `generate_questions`
# parses the reply as JSON. A weaker 3B model returns prose with a JSON-ish
# shape and the parse throws.
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"

# How long to wait for the local port before deciding nobody is home. This runs
# on the first LLM call, so it is a latency floor for that request — kept short
# because a loopback TCP connect either succeeds in microseconds or is refused.
PROBE_TIMEOUT_SEC = 0.6

_probe_result: bool | None = None

# The cloud probe is a real internet round-trip, not loopback, so it gets a more
# forgiving ceiling than PROBE_TIMEOUT_SEC. In practice a live host answers a TCP
# connect in well under a second, and a truly offline machine fails almost
# instantly (DNS failure / no route) — this ceiling only bites when a host is
# reachable at the network layer but silent, which is rare for the cloud CDNs.
CLOUD_PROBE_TIMEOUT_SEC = 1.5

_cloud_probe_result: bool | None = None

# Runtime overrides, set by the /api/v1/llm/* endpoints.
#
# These exist because `load_dotenv()` runs once at import, so rewriting
# backend/.env does NOT change this process — the user would save a setting,
# see it persisted, and watch the app carry on with the old one until Python
# restarted. Writing the file is for the next boot; these are for right now.
# `None` means "nobody has overridden it", which is not the same as empty.
_mode_override: str | None = None
_provider_override: str | None = None

# Bumped every time a runtime override changes. Cached clients compare against
# this to know their base_url is stale: chat_model() is read per call, but a
# client's base_url is frozen at construction, so a provider switch would
# otherwise send the new provider's model name to the OLD provider's endpoint
# (Groq's `openai/gpt-oss-20b` posted to integrate.api.nvidia.com → 404).
_config_generation = 0


def config_generation() -> int:
    return _config_generation


def _bump_generation() -> None:
    global _config_generation
    _config_generation += 1


def configured_mode() -> str:
    if _mode_override is not None:
        return _mode_override
    return os.getenv("LLM_MODE", "auto").strip().lower()


def cloud_provider() -> str:
    """Which cloud vendor is in play. Falls back rather than raising, because a
    typo'd LLM_PROVIDER should not take the whole app down."""
    name = (
        _provider_override
        or os.getenv("LLM_PROVIDER", DEFAULT_CLOUD_PROVIDER)
    ).strip().lower()
    if name not in CLOUD_PROVIDERS:
        logger.warning(f"Unknown LLM_PROVIDER {name!r} — using {DEFAULT_CLOUD_PROVIDER}")
        return DEFAULT_CLOUD_PROVIDER
    return name


def provider_spec(name: str | None = None) -> dict:
    return CLOUD_PROVIDERS[name or cloud_provider()]


def api_key_for(name: str | None = None) -> str:
    env_key = provider_spec(name).get("env_key", "")
    if not env_key:
        return ""  # keyless provider — no env var to look up
    return os.getenv(env_key, "").strip()


def has_api_key(name: str | None = None) -> bool:
    spec = provider_spec(name)
    # Keyless providers (non_openai with no env_key) are always "available".
    if spec.get("non_openai") and not spec.get("env_key"):
        return True
    return bool(api_key_for(name))


def mask_key(key: str) -> str:
    """`nvapi-abc…xyz` → `nvapi-••••wxyz`. Enough to recognise which key is
    installed, not enough to use it. Short strings collapse entirely rather
    than leaking a meaningful fraction of themselves."""
    if not key:
        return ""
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:6]}{'•' * 4}{key[-4:]}"


def ollama_base_url() -> str:
    """Normalise OLLAMA_HOST into an OpenAI-compatible `/v1` base URL.

    Accepts what people actually write: `localhost:11434`, `http://host:11434`,
    or a URL that already ends in `/v1`.
    """
    host = os.getenv("OLLAMA_HOST", OLLAMA_DEFAULT_HOST).strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    if not host.endswith("/v1"):
        host = f"{host}/v1"
    return host


def ollama_reachable(force: bool = False) -> bool:
    """True if something is listening on the Ollama port.

    A TCP connect, not an HTTP GET: it needs no dependency, and the question
    here is only "is the daemon up". Cached, because in `auto` mode this is
    consulted on every call and the answer does not change mid-process.
    """
    global _probe_result
    if _probe_result is not None and not force:
        return _probe_result

    parsed = urlparse(ollama_base_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_SEC):
            _probe_result = True
    except OSError:
        _probe_result = False
    return _probe_result


def cloud_reachable(force: bool = False) -> bool:
    """True if the selected cloud provider's API host accepts a TCP connection.

    The mirror of `ollama_reachable`, aimed outward: it answers "can this machine
    reach the cloud model at all", which is the question the chooser needs before
    it lets someone commit to Online. Like the local probe it is a bare TCP
    connect — no key spent, no request billed — and cached, because status() may
    ask more than once and the answer does not move between two calls.

    Probes the chosen provider's host rather than a generic ping: the honest
    claim is "we cannot reach the API you picked", and a network that blocks the
    provider but not the wider internet should still read as unreachable here.
    """
    global _cloud_probe_result
    if _cloud_probe_result is not None and not force:
        return _cloud_probe_result

    parsed = urlparse(provider_spec()["base_url"])
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        _cloud_probe_result = False
        return _cloud_probe_result
    try:
        with socket.create_connection((host, port), timeout=CLOUD_PROBE_TIMEOUT_SEC):
            _cloud_probe_result = True
    except OSError:
        _cloud_probe_result = False
    return _cloud_probe_result


def effective_mode() -> str:
    """Resolve `auto` to a concrete backend; pass explicit choices through."""
    mode = configured_mode()
    if mode in ("ollama", "local"):
        return "ollama"
    if mode in ("nvidia", "cloud") or mode in CLOUD_PROVIDERS:
        return "cloud"

    if ollama_reachable():
        logger.info(f"🟢 Sovereign Mode — local Ollama at {ollama_base_url()}")
        return "ollama"
    logger.info(f"🟡 No local Ollama found — using {provider_spec()['label']} cloud")
    return "cloud"


def is_local() -> bool:
    return effective_mode() == "ollama"


def cloud_model_for(name: str) -> str:
    """The model a specific cloud provider should use.

    Honours the per-provider env override (NVIDIA_MODEL, GROQ_MODEL, …) before
    the table default, so someone can point Groq at a bigger model without
    touching this file.
    """
    spec = CLOUD_PROVIDERS[name]
    return os.getenv(spec["env_key"].replace("_API_KEY", "_MODEL"), spec["model"])


def cloud_models_for(name: str) -> list[str]:
    """Ordered models to try for one provider: the primary first, then any hot
    standbys.

    The primary honours the per-provider *_MODEL env override (via
    cloud_model_for); the standbys are the provider table's `fallbacks`. Deduped
    and order-preserving, so a standby equal to the primary — e.g. after someone
    points NVIDIA_MODEL at the standby — collapses to one entry instead of being
    tried twice.
    """
    spec = CLOUD_PROVIDERS[name]
    ordered = [cloud_model_for(name), *spec.get("fallbacks", ())]
    seen: set[str] = set()
    out: list[str] = []
    for model in ordered:
        if model and model not in seen:
            seen.add(model)
            out.append(model)
    return out


def chat_model() -> str:
    if is_local():
        return os.getenv("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
    return cloud_model_for(cloud_provider())


# Nemotron models "think" (a hidden reasoning trace) before answering unless told
# not to, and this app does not stream, so the candidate waits out the whole
# trace. Measured in-app on the exact interview requests with thinking on:
# super-120b took 6-43 s a turn and broke the report JSON, and lightning-30b
# came back empty every time because the trace ate its token budget. With this
# switch: 0.5-1.7 s replies and valid JSON.
NEMOTRON_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}


def extra_body_for(model: str) -> dict | None:
    """Vendor-specific request fields for `model`, passed as `extra_body=`."""
    if model.startswith("nvidia/nemotron"):
        return NEMOTRON_THINKING_OFF
    return None


def cloud_fallback_order() -> list[str]:
    """Cloud providers to try, in order: the one the user picked first, then any
    other provider that also has a key.

    A free hosted tier can be degraded for one vendor while another is healthy —
    measured, not hypothetical: NVIDIA's NIM returned a 500 "inference
    connection error" on one call shape and stalled another past 180s while the
    key and params were valid. Trying the second cloud before dropping to a
    local 3B model keeps the cloud-quality answer the user chose when only one
    vendor is down. Providers without a key are skipped — nothing to try there.
    """
    primary = cloud_provider()
    ordered = [primary] + [n for n in CLOUD_PROVIDERS if n != primary]
    return [n for n in ordered if has_api_key(n)]


def make_ollama_client() -> AsyncOpenAI:
    """A client pointed at local Ollama regardless of the configured mode.

    The conversation engine's last fallback rung needs Ollama even while the
    process mode is `cloud`, so this cannot go through `make_client()` (which
    follows the mode and would hand back a cloud client instead).
    """
    return AsyncOpenAI(base_url=ollama_base_url(), api_key="ollama", max_retries=0)


def make_client(provider: str | None = None) -> AsyncOpenAI:
    """An AsyncOpenAI pointed at a backend.

    With no argument it follows the effective mode: local Ollama, or the
    configured cloud provider. Pass a provider name to force a specific cloud
    vendor — the conversation engine uses this to try a second cloud provider
    when the first is degraded, without disturbing the process-wide mode.

    Ollama ignores the API key, but the OpenAI SDK refuses to construct without
    a non-empty one — hence the placeholder rather than "".

    max_retries=0 because this app has its own fallback: the SDK retries twice
    by default, which turns one degraded vendor (a 500, or a request that stalls
    to the timeout) into a 60-180s hang before the caller can move on. Failing
    fast is the whole point of having somewhere to fall back to.
    """
    if provider is None and is_local():
        return make_ollama_client()
    spec = provider_spec(provider)
    return AsyncOpenAI(
        base_url=spec["base_url"], api_key=api_key_for(provider), max_retries=0
    )


def describe() -> str:
    """One line for startup logs and /health."""
    if is_local():
        return f"ollama · {chat_model()} · {ollama_base_url()}"
    return f"{cloud_provider()} · {chat_model()}"


# ── Runtime configuration (the pill page writes through these) ───────────

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# A key is an opaque credential, so the allowlist is deliberately narrow rather
# than "anything without a newline". This is what stops a pasted value from
# carrying `\nADMIN=1` into .env and defining extra variables on next boot.
_KEY_RE = re.compile(r"^[A-Za-z0-9_\-.]{16,256}$")


def validate_key(key: str) -> str:
    """Return the cleaned key, or raise ValueError with a user-facing reason."""
    cleaned = (key or "").strip().strip('"').strip("'")
    if not cleaned:
        raise ValueError("Paste a key first.")
    if not _KEY_RE.match(cleaned):
        raise ValueError(
            "That does not look like an API key — expected 16+ characters of "
            "letters, digits, dashes or underscores, on a single line."
        )
    return cleaned


def write_env_var(name: str, value: str) -> None:
    """Set one variable in backend/.env, preserving every other line.

    Rewrites in place rather than appending, so saving twice does not leave two
    conflicting definitions (python-dotenv would silently honour the first).
    Comments, blank lines and unrelated keys survive byte-for-byte.
    """
    line = f"{name}={value}"
    try:
        existing = ENV_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        existing = []

    out, replaced = [], False
    for row in existing:
        if row.strip().startswith(f"{name}=") and not row.strip().startswith("#"):
            out.append(line)
            replaced = True
        else:
            out.append(row)
    if not replaced:
        out.append(line)

    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def set_api_key(provider: str, key: str) -> str:
    """Validate, persist, and apply a cloud key. Returns the masked form."""
    if provider not in CLOUD_PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}")
    cleaned = validate_key(key)
    env_name = CLOUD_PROVIDERS[provider]["env_key"]

    write_env_var(env_name, cleaned)   # for the next boot
    os.environ[env_name] = cleaned     # for this one
    # A cached client holds the OLD key in its auth header, so a pasted key
    # would not take effect until restart without this.
    _bump_generation()
    # Never log `cleaned` — masked only, since these lines reach terminals,
    # CI output and screen shares.
    logger.info(f"🔑 {provider} key set ({mask_key(cleaned)})")
    return mask_key(cleaned)


def set_mode(mode: str, provider: str | None = None) -> None:
    """Switch backend at runtime and persist the choice for the next boot."""
    mode = (mode or "").strip().lower()
    if mode not in ("auto", "ollama", "cloud"):
        raise ValueError(f"Unknown mode {mode!r} — expected auto, ollama or cloud")

    global _mode_override, _provider_override
    _mode_override = mode
    write_env_var("LLM_MODE", mode)

    if provider:
        if provider not in CLOUD_PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}")
        _provider_override = provider
        write_env_var("LLM_PROVIDER", provider)

    if mode != "ollama":
        # Re-probe on the next `auto` resolution: the user may have started
        # Ollama since this process booted, and the cached answer would be a lie.
        global _probe_result
        _probe_result = None

    # The cloud probe is cached per host, and a provider switch changes the host;
    # drop it here so status()'s forced probe on the reply this call returns is
    # measured against the provider the user just chose.
    global _cloud_probe_result
    _cloud_probe_result = None

    _bump_generation()
    logger.info(f"🔀 LLM mode → {mode} ({describe()})")


# An Ollama tag, e.g. `qwen2.5:3b` or `hf.co/user/repo:Q4_K_M`. Narrow for the
# same reason as _KEY_RE: this value is written into .env, so anything that can
# carry a newline can define extra variables on the next boot. Slash and colon
# are in because registry paths and quantisation suffixes need them.
_MODEL_RE = re.compile(r"^[A-Za-z0-9_\-./:]{1,128}$")


def set_ollama_model(tag: str) -> str:
    """Point local mode at a different Ollama model. Returns the cleaned tag.

    Deliberately does NOT verify the model is pulled. The picker already knows
    what is installed — it reads /api/tags — and refusing an absent tag here
    would block the legitimate "pull it, then select it" order. A missing model
    surfaces on the first question with Ollama's own 404, which names the tag.
    """
    cleaned = (tag or "").strip()
    if not cleaned:
        raise ValueError("Pick a model first.")
    if not _MODEL_RE.match(cleaned):
        raise ValueError(
            "That does not look like an Ollama tag — expected something like "
            "qwen2.5:3b, on a single line."
        )

    write_env_var("OLLAMA_MODEL", cleaned)  # for the next boot
    os.environ["OLLAMA_MODEL"] = cleaned  # for this one
    logger.info(f"🧠 Ollama model → {cleaned}")
    return cleaned


def status() -> dict:
    """Everything the pill page needs to render truthfully. No raw keys."""
    return {
        "mode": configured_mode(),
        "effective": effective_mode(),
        "provider": cloud_provider(),
        "model": chat_model(),
        "ollama_reachable": ollama_reachable(force=True),
        "cloud_reachable": cloud_reachable(force=True),
        "ollama_model": os.getenv("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL),
        "providers": {
            name: {
                "label": spec["label"],
                "model": cloud_model_for(name),
                "console": spec["console"],
                "prefix": spec["prefix"],
                "configured": has_api_key(name),
                "masked": mask_key(api_key_for(name)),
            }
            for name, spec in CLOUD_PROVIDERS.items()
        },
    }


def extract_json(raw: str):
    """Parse a model reply that is supposed to be JSON.

    Small local models wrap JSON in prose or a markdown fence far more often
    than the 8B cloud model does, so the fence-stripping that already existed
    is not enough on its own. Falls back to the outermost bracket pair, which
    survives a leading "Here are the questions:". Returns None on failure so
    callers can use their own backup rather than crashing the request.
    """
    if not raw:
        return None
    text = raw.strip()

    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    logger.warning(f"Could not parse JSON from model reply: {raw[:160]!r}")
    return None


def normalize_coaching(value) -> dict:
    """Clamp an LLM coaching object to the report UI's stable contract."""
    if not isinstance(value, dict):
        value = {}

    def text(key: str, fallback: str, max_chars: int) -> str:
        raw = value.get(key)
        clean = " ".join(str(raw).split()) if raw is not None else ""
        return (clean or fallback)[:max_chars]

    def items(key: str, fallback: list[str], limit: int) -> list[str]:
        raw = value.get(key)
        if not isinstance(raw, list):
            raw = fallback
        clean = [" ".join(str(item).split())[:180] for item in raw if str(item).strip()]
        return (clean or fallback)[:limit]

    readiness = text("readiness", "Developing", 40)
    if readiness not in {"Strong", "Developing", "Needs targeted practice"}:
        readiness = "Developing"

    return {
        "verdict": text(
            "verdict",
            "Your interview showed useful foundations with room for more specific evidence.",
            180,
        ),
        "strengths": items(
            "strengths", ["You completed the interview with steady engagement."], 3
        ),
        "primary_improvement": text(
            "primary_improvement",
            "Support each answer with a concrete situation, decision, and measurable result.",
            240,
        ),
        "next_actions": items(
            "next_actions",
            [
                "Prepare two project examples using the STAR structure.",
                "State the result of each decision in one sentence.",
            ],
            3,
        ),
        "readiness": readiness,
    }
