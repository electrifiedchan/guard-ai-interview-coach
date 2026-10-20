"""Pins the LLM backend selection in `core_memory/llm_config.py`.

Two things here are easy to break silently and expensive to notice:

1. **Mode resolution.** `auto` decides between a local model and a cloud API by
   probing a port. If that probe were to wrongly report "down", the app keeps
   working — it just quietly sends the candidate's resume to a third party,
   which is the exact opposite of what Sovereign Mode promises. A privacy
   regression that still passes a smoke test is the kind that ships.

2. **JSON extraction.** Local 3B models wrap JSON in prose far more often than
   the 8B cloud model does. `generate_questions` falls back to canned questions
   on a parse failure, so a broken parser looks like a working app serving
   generic questions — the resume tailoring just silently stops happening.

Run: venv\\Scripts\\python.exe test_llm_config.py
"""
import os
import sys
import socket
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core_memory import llm_config

passed = failed = 0


def check(label: str, got, want):
    global passed, failed
    if got == want:
        passed += 1
        print(f"  ok   {label}")
    else:
        failed += 1
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")


class _EnvSandbox:
    """Set env vars and reset the probe cache, restoring both afterwards.

    The cache exists so `auto` does not pay a socket connect per LLM call, but
    it makes tests order-dependent unless cleared between cases.
    """

    KEYS = ("LLM_MODE", "OLLAMA_HOST", "OLLAMA_MODEL", "NVIDIA_MODEL")

    def __init__(self, **env):
        self.env = env

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in self.env.items() if v is not None})
        llm_config._probe_result = None
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        llm_config._probe_result = None
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeOllama:
    """A socket that accepts connections, standing in for the daemon.

    The probe is a TCP connect, so a bare listener is a faithful double — and
    it keeps the test hermetic whether or not Ollama is installed on the
    machine running it.
    """

    def __enter__(self):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.stop = threading.Event()

        def serve():
            while not self.stop.is_set():
                try:
                    conn, _ = self.sock.accept()
                    conn.close()
                except OSError:
                    return

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.sock.close()
        return False


print("\nurl normalisation — accepts what people actually type:")
for raw, want in [
    ("localhost:11434", "http://localhost:11434/v1"),
    ("http://localhost:11434", "http://localhost:11434/v1"),
    ("http://localhost:11434/", "http://localhost:11434/v1"),
    ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ("http://192.168.1.9:11434", "http://192.168.1.9:11434/v1"),
]:
    with _EnvSandbox(OLLAMA_HOST=raw):
        check(f"{raw!r}", llm_config.ollama_base_url(), want)


print("\nexplicit modes are obeyed regardless of what is running:")
dead = f"http://127.0.0.1:{_free_port()}"
with _EnvSandbox(LLM_MODE="ollama", OLLAMA_HOST=dead):
    # Forcing local must NOT silently fall back to the cloud. A user who set
    # this wants a hard failure, not a quiet upload of their resume.
    check("ollama stays local even when the port is dead",
          llm_config.effective_mode(), "ollama")

# `effective_mode` answers "where does it run", so every cloud vendor resolves
# to the single string "cloud" — whose cloud is `cloud_provider()`'s question,
# a separate axis added when Groq joined NVIDIA. Asserting "nvidia" here would
# be asserting that the two axes are one, which is the confusion this split
# exists to prevent; `configured_mode()` below is where the raw value is checked.
with _FakeOllama() as fake:
    with _EnvSandbox(LLM_MODE="nvidia", OLLAMA_HOST=f"http://127.0.0.1:{fake.port}"):
        check("nvidia stays cloud even when a local server is up",
              llm_config.effective_mode(), "cloud")
        check("...and the vendor axis still says which cloud",
              llm_config.cloud_provider(), "nvidia")


print("\nauto follows whether the daemon is actually listening:")
with _FakeOllama() as fake:
    with _EnvSandbox(LLM_MODE="auto", OLLAMA_HOST=f"http://127.0.0.1:{fake.port}"):
        check("server up   -> local", llm_config.effective_mode(), "ollama")
        check("is_local() agrees", llm_config.is_local(), True)

with _EnvSandbox(LLM_MODE="auto", OLLAMA_HOST=dead):
    check("server down -> cloud", llm_config.effective_mode(), "cloud")
    check("is_local() agrees", llm_config.is_local(), False)

with _EnvSandbox(OLLAMA_HOST=dead):
    check("unset LLM_MODE defaults to auto", llm_config.configured_mode(), "auto")


print("\nmodel choice follows the resolved backend:")
with _FakeOllama() as fake:
    host = f"http://127.0.0.1:{fake.port}"
    with _EnvSandbox(LLM_MODE="auto", OLLAMA_HOST=host):
        check("local default", llm_config.chat_model(), llm_config.OLLAMA_DEFAULT_MODEL)
    with _EnvSandbox(LLM_MODE="auto", OLLAMA_HOST=host, OLLAMA_MODEL="llama3.1:8b"):
        check("OLLAMA_MODEL override", llm_config.chat_model(), "llama3.1:8b")

with _EnvSandbox(LLM_MODE="nvidia"):
    check("cloud default", llm_config.chat_model(), llm_config.NVIDIA_MODEL)


print("\ncloud model list — primary first, then hot standby, deduped:")
with _EnvSandbox(LLM_MODE="nvidia"):
    nv = llm_config.cloud_models_for("nvidia")
    check("nvidia primary is first", nv[0], llm_config.NVIDIA_MODEL)
    check("nvidia hot standby follows", llm_config.NVIDIA_FALLBACK_MODEL in nv, True)
    check("nvidia list has no duplicates", len(nv), len(set(nv)))
    check("groq list is just its primary",
          llm_config.cloud_models_for("groq"),
          [llm_config.CLOUD_PROVIDERS["groq"]["model"]])
with _EnvSandbox(LLM_MODE="nvidia", NVIDIA_MODEL=llm_config.NVIDIA_FALLBACK_MODEL):
    # Pointing the primary env override at the standby collapses the list — the
    # dedupe must not leave the same model queued twice.
    check("primary==standby is not queued twice",
          llm_config.cloud_models_for("nvidia"),
          [llm_config.NVIDIA_FALLBACK_MODEL, llm_config.NVIDIA_FALLBACK_MODEL_2])


print("\nnemotron thinking is switched off, other models get no extra fields:")
check("nemotron primary gets thinking off",
      llm_config.extra_body_for(llm_config.NVIDIA_MODEL), llm_config.NEMOTRON_THINKING_OFF)
check("non-nemotron model gets nothing", llm_config.extra_body_for("qwen2.5:3b"), None)


print("\nclient points at the right endpoint:")
with _FakeOllama() as fake:
    host = f"http://127.0.0.1:{fake.port}"
    with _EnvSandbox(LLM_MODE="auto", OLLAMA_HOST=host):
        # The OpenAI SDK refuses to construct with an empty api_key, so local
        # mode must pass a placeholder — this asserts it does not regress to "".
        check("local base_url", str(llm_config.make_client().base_url).rstrip("/"),
              f"{host}/v1")

with _EnvSandbox(LLM_MODE="nvidia"):
    check("cloud base_url", str(llm_config.make_client().base_url).rstrip("/"),
          llm_config.NVIDIA_BASE_URL)


print("\njson extraction survives what small models actually return:")
cases = [
    ("clean array", '[{"question":"a"}]', [{"question": "a"}]),
    ("clean object", '{"questions":[1]}', {"questions": [1]}),
    ("json fence", '```json\n{"questions":[1]}\n```', {"questions": [1]}),
    ("bare fence", '```\n{"questions":[1]}\n```', {"questions": [1]}),
    ("leading prose", 'Here you go: [{"question":"a"}]', [{"question": "a"}]),
    ("trailing prose", '[{"question":"a"}] hope that helps!', [{"question": "a"}]),
    ("not json", "I cannot help with that", None),
    ("empty", "", None),
]
for label, raw, want in cases:
    check(label, llm_config.extract_json(raw), want)

# Malformed JSON must return None rather than a half-parsed object: callers
# treat None as "use the backup questions", and a truncated dict would sail
# through that check and reach the UI as a single broken question.
check("truncated json", llm_config.extract_json('[{"question": "a"'), None)

compact = llm_config.normalize_coaching({
    "verdict": "  Strong foundation. ",
    "strengths": [" Clear examples ", "Specific tradeoff", "Extra item", "Ignored"],
    "primary_improvement": " Add measurable outcomes ",
    "next_actions": ["Use STAR", "Name the result", "Practice once", "Ignored"],
    "readiness": "Strong",
})
check("coaching object is normalized", compact, {
    "verdict": "Strong foundation.",
    "strengths": ["Clear examples", "Specific tradeoff", "Extra item"],
    "primary_improvement": "Add measurable outcomes",
    "next_actions": ["Use STAR", "Name the result", "Practice once"],
    "readiness": "Strong",
})

fallback = llm_config.normalize_coaching(None)
check("invalid coaching uses stable fallback", set(fallback), {
    "verdict", "strengths", "primary_improvement", "next_actions", "readiness",
})


print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
