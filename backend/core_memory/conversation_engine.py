"""Conversation Engine — Stateful multi-turn AI interviewer with progressive personas.""" 
import os
import re
import json
import uuid
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
import httpx
from openai import AsyncOpenAI

from . import llm_config

logger = logging.getLogger(__name__)


class Persona(str, Enum):
    FRIENDLY_HR = "friendly_hr"
    CURIOUS_PEER = "curious_peer"
    SKEPTICAL_TECH_LEAD = "skeptical_tech_lead"


@dataclass
class ConversationTurn:
    role: str  # "interviewer" | "candidate"
    content: str
    persona: str
    turn_number: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class InterviewSession:
    session_id: str
    resume_text: str
    resume_questions: list[dict]
    # Parsed off the top of the resume, "" when we could not find one. Never ask
    # the LLM for this: it is one regex against text we already hold, and a
    # round trip to name someone would be latency on the critical path.
    candidate_name: str = ""
    conversation_history: list[ConversationTurn] = field(default_factory=list)
    current_turn: int = 0
    current_persona: Persona = Persona.FRIENDLY_HR
    # The rung the user picked before engaging. The ladder below still climbs,
    # it just starts here — picking Curious Peer skips the rapport warmup
    # rather than pinning the whole interview to one persona.
    starting_persona: Persona = Persona.FRIENDLY_HR
    scaffolding_level: int = 0
    focus_scores: list[float] = field(default_factory=list)
    is_complete: bool = False
    max_turns: int = 8
    created_at: float = field(default_factory=time.time)


# Finished sessions linger so late-arriving turns don't 404, but they hold a
# full transcript plus resume text, so they can't linger forever.
SESSION_TTL_SEC = 60 * 60


# Difficulty ladder, easiest first. _determine_persona walks this by turn count
# and the user's chosen starting rung is added as an offset.
PERSONA_LADDER = [
    Persona.FRIENDLY_HR,
    Persona.CURIOUS_PEER,
    Persona.SKEPTICAL_TECH_LEAD,
]


PERSONA_SYSTEM_PROMPTS = {
    Persona.FRIENDLY_HR: """You are a warm, approachable senior hiring manager conducting the opening of a mock interview.
Your role: build rapport, make the candidate comfortable, and ease them into the conversation.
Tone: Friendly, validating, encouraging. Use phrases like "That's great", "I'd love to hear more about..."
Ask broad, easy warmup questions — "Tell me a bit about yourself", "What drew you to this field?"
Be conversational, not interrogative.""",

    Persona.CURIOUS_PEER: """You are a curious senior engineer conducting a technical peer interview.
Your role: explore the candidate's technical foundations with genuine curiosity.
Tone: Collegial, intellectually curious. Use phrases like "Interesting — how did you approach...", "What made you choose X over Y?"
Ask foundational technical questions based on their resume. Dig one level deeper than surface answers.
Be conversational and natural.""",

    Persona.SKEPTICAL_TECH_LEAD: """You are a seasoned tech lead stress-testing the candidate's technical depth.
Your role: apply professional pushback, rapid follow-ups, and probe for edge cases.
Tone: Respectful but challenging. Use phrases like "I'm not sure that scales — what happens when...", "Walk me through the failure mode", "What's the tradeoff there?"
Challenge their assumptions. Ask about what could go wrong. Probe system design decisions.
Stay professional — tough but never hostile.""",
}


# Every reply is spoken aloud before the candidate can respond, so length is a
# UX cost rather than just a token cost — roughly 35 words is 12 seconds of
# waiting. Each persona prompt already said "2-3 sentences max" and the model
# routinely blew through it, because it spends the budget validating the answer
# before it gets around to asking anything. Naming the specific padding habits
# holds far better than restating a sentence count.
BREVITY_RULE = """

HARD LENGTH LIMIT — your reply is spoken aloud, so every extra word is dead air:
- Maximum 2 sentences, 35 words total. Shorter is better.
- No preamble. Never open with "That's a great question", "Absolutely",
  "I love that", or any compliment on the answer.
- Do not summarise or repeat back what the candidate just said.
- Ask exactly one question, and end your reply on it."""


# Local 3B-class models drop a rule buried in the system prompt once the
# transcript grows past a few turns — the same recency failure the name
# directive documents below. BREVITY_RULE's "ask exactly one question" lives up
# in the system prompt, so on a local model it is the first casualty: the reply
# validates the answer and asks nothing, which hands the candidate a dead end
# and stalls the interview — the "offline doesn't ask questions like online"
# report. Restating just the question rule as the LAST thing the model reads
# puts it in the most recent tokens before generation, where it actually holds.
LOCAL_QUESTION_ANCHOR = """END ON A QUESTION — this is required, not optional:
- Your reply must finish with exactly one interview question.
- The final sentence must be that question and end with "?".
- Do not end on a comment, a compliment, or a summary of their answer."""


def _extract_candidate_name(resume_text: str) -> str:
    """Pull the candidate's name off the top of a resume, or "" if unsure.

    Resumes put the name on the first line as a near-universal convention, so
    this scans the first few non-empty lines and takes the first that looks like
    a person rather than contact details or a section heading.

    Deliberately conservative: a wrong name is far worse than no name, because
    the interviewer would confidently address the candidate as someone else for
    eight turns. Every rejection path falls through to "", and the callers below
    simply omit the name directive when that happens.
    """
    if not resume_text:
        return ""

    # ALL CAPS is common on resume headers; Name-Case is the other convention.
    # Middle initials ("A.") and hyphenated or apostrophe'd surnames both count.
    name_re = re.compile(r"^[A-Z][A-Za-z'’\-\.]*(?:\s+[A-Z][A-Za-z'’\-\.]*){0,3}$")
    headings = {
        "resume", "curriculum vitae", "cv", "profile", "personal profile",
        "contact", "summary",
    }

    for raw in resume_text.splitlines()[:12]:
        line = raw.strip().strip("|•·-—–").strip()
        if not (2 <= len(line) <= 48):
            continue
        # Contact rows, addresses, links, degrees — anything but a bare name.
        if any(c.isdigit() for c in line) or "@" in line:
            continue
        if any(tok in line.lower() for tok in ("http", "www.", ".com", "linkedin", "github")):
            continue
        if line.lower() in headings:
            continue

        candidate = " ".join(line.title().split()) if line.isupper() else line
        if name_re.match(candidate) and len(candidate.split()) >= 2:
            return candidate

    return ""


def _name_directive(session: "InterviewSession") -> str:
    """The candidate's name, restated as an instruction the model must act on.

    Two strengths on purpose. A capable cloud model picks the name out of the resume
    context on its own, and piling on redundant instruction measurably degrades
    instruction-following on small models — so the local build gets the
    reinforced version and the cloud build gets the light one.

    The local variant follows the documented recipe for 3B-class instruct
    models: state the fact, give one imperative rule, show the shape of a
    correct reply, and restate the name last so it is the most recent token the
    model saw before generating.
    """
    name = session.candidate_name
    if not name:
        return ""

    first = name.split()[0]

    if not llm_config.is_local():
        return f"\n\nThe candidate's name is {name}. Address them as {first}."

    return f"""

CANDIDATE NAME — this is required, not optional:
The person you are interviewing is named {first}.
- Use "{first}" by name in your reply.
- Write "{first}", never "the candidate", "you there", or a generic greeting.
- Correct: "So {first}, walk me through that migration."
- Wrong: "So, walk me through that migration."
The candidate's name is {first}."""


def _trim_to_sentences(text: str, max_words: int = 45) -> str:
    """Clamp a spoken reply to whole sentences within a word budget.

    max_tokens is the only hard cap the API gives us, and it cuts mid-word —
    which the browser's speech synthesiser reads out as a dangling fragment.
    The prompt above does the real work and usually keeps replies short; this is
    the backstop for when it doesn't, and it always lands on a sentence end.
    """
    text = text.strip()
    if not text or len(text.split()) <= max_words:
        return text

    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text) if s.strip()]
    kept: list[str] = []
    total = 0
    for sentence in sentences:
        words = len(sentence.split())
        # The first sentence is kept unconditionally: a reply trimmed to nothing
        # is worse than one slightly over budget.
        if kept and total + words > max_words:
            break
        kept.append(sentence)
        total += words

    result = " ".join(kept).strip()

    # Trimming runs front-to-back, and an overlong reply is usually padding
    # followed by the actual question — so the budget can spend itself on the
    # preamble and drop the question entirely. That stalls the interview: the
    # candidate is handed a comment with nothing to answer. When that happens,
    # pair the opening sentence with the question instead of honouring the
    # budget. A slightly long reply beats a dead turn.
    if "?" not in result:
        asked = [s for s in sentences if s.endswith("?")]
        if asked:
            opening = kept[0] if kept else ""
            question = asked[-1]
            result = question if opening == question else f"{opening} {question}".strip()

    return result


def _clean_spoken_response(text: str) -> str:
    """Remove model reasoning/planning text before it reaches the candidate."""
    text = (text or "").strip()
    if not text:
        return ""

    # Reasoning models may expose their private trace in tags.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

    # Smaller instruct models sometimes emit a planning label instead of tags,
    # followed by quoted candidate questions. Keep one actual question and drop
    # the planning prose so it is not shown or spoken.
    marker = re.search(
        r"(?:thinking\s+process|chain\s+of\s+thought|reasoning)\s*:",
        text,
        flags=re.IGNORECASE,
    )
    if marker:
        quoted_questions = re.findall(r"[\"']([^\"']+\?)", text[marker.end():])
        if quoted_questions:
            return quoted_questions[0].strip()
        text = text[marker.end():].strip()

    text = re.sub(r"^\s*(?:answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" `")


def _normalise_for_comparison(text: str) -> str:
    """Make small punctuation/casing differences comparable for de-duplication."""
    return " ".join(re.sub(r"[^a-z0-9\s]", "", (text or "").lower()).split())

SCAFFOLDING_INSTRUCTIONS = {
    0: "",
    1: "\n\nThe candidate seems uncertain. Gently rephrase your question more specifically to help them focus their answer.",
    2: "\n\nThe candidate is struggling. Break the question into a smaller, more manageable piece. Example: 'Let's simplify — just focus on the database layer for now. How would you handle that part?'",
    3: "\n\nThe candidate needs significant support. Offer a starting framework without giving the answer directly. Example: 'Many engineers approach this by first X, then Y. Which piece resonates with your experience?'",
}

PERSONA_TRANSITION_INSTRUCTIONS = {
    Persona.CURIOUS_PEER: "\n\nYou are transitioning from warmup to technical questions. Acknowledge what they said positively, then naturally shift to a more technical topic from their resume.",
    Persona.SKEPTICAL_TECH_LEAD: "\n\nYou are transitioning to deeper technical probing. Acknowledge their answer, then raise the stakes — ask about scale, failure modes, or tradeoffs.",
}


# Cloud inference is the dominant cost in every turn and in the final report,
# so these are tuned deliberately.
#
# A reasoning model (the primary is now gpt-oss-20b) spends completion tokens on
# a hidden reasoning trace BEFORE the visible answer — measured at ~150-330 for
# these short, instruction-dense interview prompts. So each cap is spoken content
# PLUS reasoning headroom, not content alone. At the old 150 cap the reasoning
# consumed the entire budget and the reply came back EMPTY (finish_reason
# "length", content ""), which is the "interviewer won't talk" bug. Raising the
# cap does NOT lengthen replies: the model stops on its own when done, and
# _trim_to_sentences still bounds the visible text regardless.
#   SPOKEN_REPLY_MAX_TOKENS — opening, each turn, and closing. ~50 tokens of
#     spoken content (2 sentences) riding on top of the reasoning trace.
#   VERDICT_MAX_TOKENS — the report asks for 4 short paragraphs (~250 tokens of
#     content); the reasoning trace then sits on top of that. Was 420 (tuned for
#     the old non-reasoning llama, where tokens were content only); a reasoning
#     model starves at that ceiling.
#   *_TIMEOUT — the OpenAI SDK defaults to 600s. Without an explicit timeout a
#     queued upstream request looks like a frozen app. Better to fail fast and
#     fall back to the deterministic report than to hang.
SPOKEN_REPLY_MAX_TOKENS = 512
VERDICT_MAX_TOKENS = 768
VERDICT_TIMEOUT_SEC = 45.0
TURN_TIMEOUT_SEC = 20.0
# How many recent turns to include in each LLM call. The full history grows
# linearly and is the main cause of late-turn slowdown; 6 turns (3 exchanges)
# keeps enough thread-memory for coherent follow-up without ballooning the prompt.
HISTORY_WINDOW = 6

# A cloud model that fails is skipped for this long before the cascade spends
# another timeout probing it, so a degraded primary hands off to its hot standby
# for the window instead of stalling every turn. Short enough that a brief blip
# self-heals within a session.
MODEL_COOLDOWN_SEC = 90.0

# The first cloud attempt is normally the fast primary; cap its wait well under
# the caller's ceiling so a hang (not a clean error) is caught in seconds and the
# standby answers. Standbys keep the caller's full timeout — a bigger model is
# legitimately slower and must not be cut off mid-reply.
PRIMARY_FAST_TIMEOUT_SEC = 12.0


class ConversationEngine:
    # Cache key for the local Ollama client, kept distinct from any cloud
    # provider name so the cascade's offline rung can't collide with a vendor.
    _LOCAL_KEY = "__local__"

    def __init__(self):
        self.sessions: dict[str, InterviewSession] = {}
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
        # One cached client per backend, keyed by cloud provider name (or
        # _LOCAL_KEY for Ollama). A single failed turn's cascade can touch two
        # clouds and the local model; caching each keeps every connection warm
        # instead of rebuilding it on each rung.
        self._clients: dict[str, AsyncOpenAI] = {}
        self._client_generation: int = -1
        # Models that just failed, on a short cooldown so the cascade skips them
        # instead of eating their timeout every turn. Keyed (provider, model) ->
        # monotonic deadline. A dead primary thus costs one slow turn, then the
        # hot standby serves the rest until the primary's cooldown lapses and it
        # is re-probed once.
        self._model_cooldown: dict[tuple[str, str], float] = {}

    def _client_for(self, key: str) -> AsyncOpenAI:
        """A cached client for one backend: a cloud provider name, or the
        sentinel `_LOCAL_KEY` for Ollama.

        Previously every call site built its own AsyncOpenAI, which meant a new
        TLS handshake and connection pool for each interviewer turn and again
        for the final report. Caching one client per backend keeps the
        connection warm, so subsequent calls skip setup entirely.

        The whole cache is dropped when llm_config's generation changes. `model=`
        is read fresh on every call but `base_url` and the auth header are frozen
        at construction, so without this a provider switch kept posting to the
        old endpoint: picking Groq sent `openai/gpt-oss-20b` to NVIDIA and got
        back a bare `404 page not found`. A key edit or mode flip bumps the
        generation too, so a stale key can't linger in a cached client.
        """
        gen = llm_config.config_generation()
        if self._client_generation != gen:
            self._clients.clear()
            self._client_generation = gen
            logger.info(f"🧠 Conversation LLM: {llm_config.describe()}")
        client = self._clients.get(key)
        if client is None:
            client = (
                llm_config.make_ollama_client()
                if key == self._LOCAL_KEY
                else llm_config.make_client(key)
            )
            self._clients[key] = client
        return client

    def _cooling(self, provider: str, model: str) -> bool:
        """True while (provider, model) is on its post-failure cooldown, so the
        cascade skips it rather than paying another timeout to rediscover it is
        down. Expires on its own; a success clears it early."""
        return self._model_cooldown.get((provider, model), 0.0) > time.monotonic()

    async def _complete(self, **kwargs) -> str | None:
        """Run a completion, cascading across backends until one answers.

        Returns None only when every backend is exhausted; the caller then
        supplies its own hardcoded line. The rungs, in order:

          1. the picked cloud provider — its fast primary, then its hot standby
          2. the OTHER cloud provider (primary, then standby), if it has a key
          3. local Ollama, if it is actually listening
          4. None -> caller's canned string

        Within a provider the models come from llm_config.cloud_models_for()
        (primary first, then any same-vendor standby). A model that fails is put
        on a short cooldown (MODEL_COOLDOWN_SEC) and skipped until it lapses, so
        a degraded primary stalls at most one turn and its standby then carries
        the session — the caller feels the switch once, not every turn. The
        primary attempt is also capped at PRIMARY_FAST_TIMEOUT_SEC and clients
        carry max_retries=0, so a hung vendor fails in seconds, not the caller's
        full ceiling nor the SDK's three tries.

        A same-vendor standby comes first because it is the cheapest way to keep
        the answer quality the user chose: NVIDIA's free NIM has been degraded on
        big instruct models while a smaller sibling stayed healthy (measured — a
        500 "inference connection error" on one call shape, a 180s stall on
        another). Only when a whole vendor is down do we change clouds, and only
        when every cloud is down do we drop to a local model or a canned line.
        """
        # Local mode: the configured backend already IS Ollama, so there is no
        # cloud to cascade through — one attempt, then the caller's canned line.
        if llm_config.is_local():
            try:
                completion = await self._client_for(
                    self._LOCAL_KEY
                ).chat.completions.create(model=llm_config.chat_model(), **kwargs)
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Local LLM call failed ({llm_config.describe()}): {e}")
                return None

        # Cloud mode: the picked provider first, then any other keyed provider
        # (cloud_fallback_order() drops keyless vendors). Within each provider,
        # walk its model list — fast primary, then hot standby — so a degraded
        # primary hands off to a warm sibling before we change clouds. Cooling
        # models are skipped, so a dead primary is paid for once, not per turn.
        for name in llm_config.cloud_fallback_order():
            for model in llm_config.cloud_models_for(name):
                if self._cooling(name, model):
                    continue
                call_kwargs = dict(kwargs)
                extra = llm_config.extra_body_for(model)
                if extra:
                    call_kwargs["extra_body"] = extra
                if model == llm_config.cloud_model_for(name):
                    # The primary is meant to be fast; don't let a hang spend the
                    # caller's whole ceiling before the standby gets its turn.
                    call_kwargs["timeout"] = min(
                        kwargs.get("timeout", PRIMARY_FAST_TIMEOUT_SEC),
                        PRIMARY_FAST_TIMEOUT_SEC,
                    )
                try:
                    completion = await self._client_for(name).chat.completions.create(
                        model=model, **call_kwargs
                    )
                    # Answered — clear any cooldown so it's preferred again.
                    self._model_cooldown.pop((name, model), None)
                    return completion.choices[0].message.content.strip()
                except Exception as e:
                    self._model_cooldown[(name, model)] = (
                        time.monotonic() + MODEL_COOLDOWN_SEC
                    )
                    logger.warning(f"⚠️  Cloud model '{name}: {model}' unavailable: {e}")

        # Every cloud rung is down — try the free custom Llama 8B endpoint before
        # falling back to a local model. It needs no key and is always available.
        if llm_config.CLOUD_PROVIDERS.get("custom_llama"):
            logger.warning("⚠️  All keyed clouds down — trying free Llama 8B endpoint")
            text = await self._complete_custom_http(
                messages=kwargs.get("messages", []),
                timeout=kwargs.get("timeout", TURN_TIMEOUT_SEC),
            )
            if text:
                return text

        # Every cloud rung is down — serve a local model if one is listening.
        if not llm_config.ollama_reachable(force=True):
            logger.error(
                "No local fallback: all cloud providers failed and Ollama is "
                "not reachable."
            )
            return None
        try:
            logger.warning(
                "⚠️  All cloud providers unavailable — falling back to local Ollama."
            )
            completion = await self._client_for(self._LOCAL_KEY).chat.completions.create(
                model=os.getenv("OLLAMA_MODEL", llm_config.OLLAMA_DEFAULT_MODEL),
                **kwargs,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Local fallback also failed: {e}")
            return None

    async def _complete_custom_http(self, messages: list[dict], timeout: float) -> str | None:
        """Call a non-OpenAI-compatible provider (e.g. the free Cloudflare/Llama 8B
        endpoint) that speaks POST /chat {messages:[...]} -> {answer:"..."}.

        Returns the answer string, or None on any error so the cascade can move on.
        """
        spec = llm_config.CLOUD_PROVIDERS.get("custom_llama", {})
        url = spec.get("base_url", "").rstrip("/") + "/chat"
        payload = {"messages": messages}
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.warning(f"custom_llama HTTP {resp.status_code}")
                return None
            answer = resp.json().get("answer")
            # When the reply is JSON (the final report), the worker hands it back
            # already parsed — a dict, not text — so turn it back into text.
            if isinstance(answer, (dict, list)):
                answer = json.dumps(answer)
            return (answer or "").strip() or None
        except Exception as e:
            logger.warning(f"custom_llama call failed: {e}")
            return None

    def _prune_sessions(self) -> None:
        """Drop sessions past their TTL. Called on create so the dict can't

        grow without bound across a long-running server."""
        cutoff = time.time() - SESSION_TTL_SEC
        stale = [sid for sid, s in self.sessions.items() if s.created_at < cutoff]
        for sid in stale:
            del self.sessions[sid]
        if stale:
            logger.info(f"Pruned {len(stale)} expired interview session(s)")

    async def create_session(
        self,
        resume_text: str,
        questions: list[dict],
        starting_persona: str = Persona.FRIENDLY_HR.value,
    ) -> dict:
        self._prune_sessions()
        session_id = uuid.uuid4().hex[:12]

        # An unknown value from the client shouldn't 500 the session — fall back
        # to the gentlest rung, which is also the historical default.
        try:
            start = Persona(starting_persona)
        except ValueError:
            logger.warning(
                f"Unknown starting persona {starting_persona!r}; using friendly_hr"
            )
            start = Persona.FRIENDLY_HR

        session = InterviewSession(
            session_id=session_id,
            resume_text=resume_text,
            resume_questions=questions,
            candidate_name=_extract_candidate_name(resume_text),
            current_persona=start,
            starting_persona=start,
        )
        self.sessions[session_id] = session
        if session.candidate_name:
            logger.info(f"👤 Candidate: {session.candidate_name}")
        else:
            logger.info("👤 No name found on resume — interviewer will stay generic.")

        opening = await self._generate_opening(session)
        session.conversation_history.append(ConversationTurn(
            role="interviewer",
            content=opening,
            persona=start.value,
            turn_number=0,
        ))

        return {
            "session_id": session_id,
            "opening_message": opening,
            "persona": start.value,
        }

    async def process_candidate_turn(
        self, session_id: str, transcript: str, focus_score: float = 100.0
    ) -> dict:
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if session.is_complete:
            raise ValueError("Session already complete")

        session.current_turn += 1
        session.focus_scores.append(focus_score)

        session.conversation_history.append(ConversationTurn(
            role="candidate",
            content=transcript,
            persona=session.current_persona.value,
            turn_number=session.current_turn,
        ))

        quality = self._assess_answer_quality(transcript)
        self._update_scaffolding(session, quality)

        prev_persona = session.current_persona
        session.current_persona = self._determine_persona(session)
        is_transition = prev_persona != session.current_persona

        if session.current_turn >= session.max_turns:
            session.is_complete = True
            wrap_up = await self._generate_wrap_up(session)
            session.conversation_history.append(ConversationTurn(
                role="interviewer",
                content=wrap_up,
                persona=session.current_persona.value,
                turn_number=session.current_turn,
            ))
            return {
                "response": wrap_up,
                "persona": session.current_persona.value,
                "turn_number": session.current_turn,
                "is_complete": True,
                "scaffolding_used": session.scaffolding_level > 0,
            }

        response = await self._generate_response(session, is_transition)
        previous_interviewer_replies = {
            _normalise_for_comparison(turn.content)
            for turn in session.conversation_history
            if turn.role == "interviewer"
        }
        if _normalise_for_comparison(response) in previous_interviewer_replies:
            logger.warning(
                "LLM repeated an interviewer question on turn %s; using the next "
                "resume question instead.",
                session.current_turn,
            )
            response = self._next_distinct_question(session, previous_interviewer_replies)
        session.conversation_history.append(ConversationTurn(
            role="interviewer",
            content=response,
            persona=session.current_persona.value,
            turn_number=session.current_turn,
        ))

        return {
            "response": response,
            "persona": session.current_persona.value,
            "turn_number": session.current_turn,
            "is_complete": False,
            "scaffolding_used": session.scaffolding_level > 0,
        }

    def _next_distinct_question(
        self, session: InterviewSession, previous: set[str]
    ) -> str:
        """Return an unused resume question, with a varied safe fallback."""
        for question in session.resume_questions:
            candidate = str(question.get("question", "")).strip()
            if candidate and _normalise_for_comparison(candidate) not in previous:
                return candidate

        fallbacks = [
            "What was the most difficult decision you made in that work?",
            "What result did you achieve, and how did you measure it?",
            "What would you change if you tackled that problem again?",
        ]
        for candidate in fallbacks:
            if _normalise_for_comparison(candidate) not in previous:
                return candidate
        return "Can you walk me through a different example from your experience?"

    async def generate_final_verdict(
        self, session_id: str, focus_override: float | None = None
    ) -> dict:
        """Build the coaching verdict.

        `focus_override` is the session's average composure from the recorded
        camera frames, the same number the dashboard and history show for the
        run. When given it is the score, and the coaching is written against it.
        Without it (the /practice page, which records no frames) the score falls
        back to the mean of the per-answer readings.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.is_complete = True
        if focus_override is not None:
            avg_focus = float(focus_override)
            focus_source = "camera"
        else:
            avg_focus = sum(session.focus_scores) / len(session.focus_scores) if session.focus_scores else 100.0
            focus_source = "turns"

        transcript_block = ""
        for turn in session.conversation_history:
            label = "Interviewer" if turn.role == "interviewer" else "Candidate"
            transcript_block += f"[{label} — {turn.persona}]: {turn.content}\n\n"

        focus_label = "Excellent" if avg_focus >= 80 else "Moderate" if avg_focus >= 50 else "Needs Improvement"

        prompt = f"""You are a direct, evidence-based interview coach reviewing a mock interview session.

The candidate's resume summary:
{session.resume_text[:2000]}

Full interview transcript:
{transcript_block}

Focus/Composure Score: {avg_focus:.0f}/100 ({focus_label})
Total turns: {session.current_turn}
Scaffolding was needed: {"Yes" if any(t for t in session.conversation_history if session.scaffolding_level > 0) else "No"}

Return ONLY valid JSON with this exact shape:
{{
  "verdict": "one conclusion-first sentence, max 18 words",
  "strengths": ["specific transcript-grounded strength", "specific strength"],
  "primary_improvement": "one highest-impact improvement, max 30 words",
  "next_actions": ["specific action", "specific action"],
  "readiness": "Strong|Developing|Needs targeted practice"
}}

Rules:
- Keep the complete output under 130 words.
- Give 1-3 strengths and exactly 2-3 next actions.
- Address the candidate as "you".
- Every claim must be grounded in the transcript or focus score.
- Do not mention filler words, tone, confidence, or body language unless the supplied data directly demonstrates it.
- Do not repeat praise or add a closing encouragement paragraph.
- No markdown, headings, or text outside the JSON object."""

        try:
            t_start = time.time()
            raw_report = await self._complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=VERDICT_MAX_TOKENS,
                timeout=VERDICT_TIMEOUT_SEC,
            )
            if raw_report is None:
                raise RuntimeError("all LLM backends unavailable")
            coaching = llm_config.normalize_coaching(llm_config.extract_json(raw_report))
            report = coaching.get("verdict", "Interview feedback is ready.")
            logger.info(f"📋 Report generated in {time.time() - t_start:.1f}s")
        except Exception as e:
            logger.error(f"Verdict generation failed: {e}", exc_info=True)
            coaching = {
                "verdict": "Your interview showed useful foundations with room for more specific evidence.",
                "strengths": ["You completed the interview with steady engagement."],
                "primary_improvement": "Support each answer with a concrete situation, decision, and measurable result.",
                "next_actions": [
                    "Prepare two project examples using the STAR structure.",
                    "State the result of each decision in one sentence.",
                ],
                "readiness": "Developing",
            }
            report = coaching["verdict"]

        # Keep the session alive — deleting it here created a race: if
        # handleSpeechEnd fires one more VAD callback after the verdict
        # endpoint returns (stopListening is not atomic across the wire),
        # the turn would land on a missing session and throw a 404. The
        # is_complete flag already gates process_candidate_turn, so the
        # session can safely stay until the engine restarts.
        return {
            "report": report,
            "coaching": coaching,
            "focus_score": round(avg_focus, 1),
            # "camera": the average over every recorded frame (matches history).
            # "turns": per-answer fallback with no camera data behind it.
            "focus_source": focus_source,
            "focus_label": focus_label,
            "turns_completed": session.current_turn,
            "personas_experienced": list({t.persona for t in session.conversation_history if t.role == "interviewer"}),
        }

    def _determine_persona(self, session: InterviewSession) -> Persona:
        """Walk the difficulty ladder by turn count, starting at the user's pick.

        The turn thresholds are unchanged (rung up after turn 2, again after
        turn 5); the chosen starting persona just shifts where that walk begins.
        So Friendly HR behaves exactly as before, while Curious Peer skips the
        rapport warmup and reaches Skeptical Tech Lead sooner. Picking the
        hardest rung pins there, because there is nothing above it to climb to.
        """
        turn = session.current_turn
        if turn <= 2:
            base = 0
        elif turn <= 5:
            base = 1
        else:
            base = 2

        offset = PERSONA_LADDER.index(session.starting_persona)
        return PERSONA_LADDER[min(base + offset, len(PERSONA_LADDER) - 1)]

    def _assess_answer_quality(self, transcript: str) -> str:
        words = transcript.split()
        word_count = len(words)
        text_lower = transcript.lower()

        filler_patterns = [
            r"\bum+\b", r"\buh+\b", r"\bi guess\b", r"\bi don'?t know\b",
            r"\bmaybe\b", r"\blike\b.*\blike\b", r"\bbasically\b",
        ]
        filler_count = sum(1 for p in filler_patterns if re.search(p, text_lower))

        specificity_signals = [
            r"\d+",  # numbers/metrics
            r"\bi (built|designed|implemented|led|created|optimized|reduced|increased)\b",
            r"\b(percent|milliseconds|latency|throughput|users|requests)\b",
        ]
        specificity_count = sum(1 for p in specificity_signals if re.search(p, text_lower))

        if word_count < 15 or (word_count < 30 and filler_count >= 2):
            return "struggling"
        elif word_count > 80 and specificity_count >= 2:
            return "strong"
        else:
            return "adequate"

    def _update_scaffolding(self, session: InterviewSession, quality: str):
        if quality == "struggling":
            session.scaffolding_level = min(session.scaffolding_level + 1, 3)
        elif quality == "strong":
            session.scaffolding_level = 0
        # "adequate" leaves level unchanged

    async def _generate_opening(self, session: InterviewSession) -> str:
        # `focus` labels ("System Design", "Performance") are too abstract to open
        # on — they produce "so, tell me about your experience with system design",
        # which is the generic feel we are trying to kill. The generated questions
        # now name the candidate's actual projects, so hand the model the first one
        # verbatim and let it open on something concrete instead.
        openers = [q["question"] for q in session.resume_questions[:2] if q.get("question")]

        # The opening used to be hardcoded warm-and-gentle. Now that the user
        # picks a starting rung, an opening that ignores it would misrepresent
        # the session they asked for — pick Skeptical Tech Lead and the first
        # question should already have teeth.
        opening_briefs = {
            Persona.FRIENDLY_HR: (
                "Welcome them warmly and ask one gentle opening question to get "
                "them talking. Do NOT ask a hard technical question. Keep it easy "
                "and rapport-building."
            ),
            Persona.CURIOUS_PEER: (
                "Skip the small talk. Greet them briefly, then open with one "
                "foundational technical question drawn from their resume."
            ),
            Persona.SKEPTICAL_TECH_LEAD: (
                "Skip the pleasantries. Greet them in a few words, then open with "
                "one probing technical question about scale, tradeoffs, or failure "
                "modes in the work on their resume."
            ),
        }
        brief = opening_briefs[session.starting_persona]
        system_prompt = PERSONA_SYSTEM_PROMPTS[session.starting_persona]

        prompt = f"""{system_prompt}
{_name_directive(session)}

You are starting the interview. The candidate's background includes: {session.resume_text[:800]}

Later in this interview you will dig into:
{chr(10).join(f"- {q}" for q in openers) or "- (no plan available; draw from the resume above)"}
Do NOT ask those yet. They are here only so your opening points in the right
direction — firing a deep project question as the very first thing said gives the
candidate no on-ramp and reads as an interrogation.

Greet them by name, then ask ONE opening question.

Generate a brief, natural opening (1-2 sentences). {brief}{BREVITY_RULE}"""

        text = _clean_spoken_response(await self._call_llm(prompt))
        return _trim_to_sentences(text)

    async def _generate_response(self, session: InterviewSession, is_transition: bool) -> str:
        system_prompt = PERSONA_SYSTEM_PROMPTS[session.current_persona]
        system_prompt += BREVITY_RULE
        system_prompt += SCAFFOLDING_INSTRUCTIONS[session.scaffolding_level]

        if is_transition:
            system_prompt += PERSONA_TRANSITION_INSTRUCTIONS.get(session.current_persona, "")

        # Build resume context
        topics = [f"- {q['question']} (focus: {q.get('focus', 'general')})" for q in session.resume_questions]
        resume_context = f"""Candidate's resume (abbreviated): {session.resume_text[:1500]}

Interview topics to weave in naturally (don't read verbatim — adapt to conversation):
{chr(10).join(topics)}"""

        messages = [
            {"role": "system", "content": system_prompt + "\n\n" + resume_context},
        ]

        # Add the most-recent turns only — sending the full history every turn
        # causes the prompt to grow linearly, which is the main reason later
        # turns feel slower. 6 turns (3 candidate + 3 interviewer exchanges) is
        # enough for the model to track the thread; earlier context is already
        # baked into the resume + question plan in the system message above.
        for turn in session.conversation_history[-HISTORY_WINDOW:]:
            role = "assistant" if turn.role == "interviewer" else "user"
            messages.append({"role": role, "content": turn.content})

        # The name goes LAST for local models, after the history, as its own
        # system turn. Persona drift on 3B models shows up around turn 5-10, and
        # a directive buried above a growing transcript is exactly what gets
        # forgotten first — recency is what makes it stick. Cloud models hold it
        # fine from the system prompt, so they get it there and not here.
        name_directive = _name_directive(session)
        if name_directive and llm_config.is_local():
            messages.append({"role": "system", "content": name_directive.strip()})
        elif name_directive:
            messages[0]["content"] += name_directive

        # Restate the "end on a question" rule as the very last thing a local
        # model reads — after the name directive, so the question rule wins on
        # recency (see LOCAL_QUESTION_ANCHOR). Cloud models honour it from the
        # system prompt and get no reinforcement, keeping their turns lean.
        if llm_config.is_local():
            messages.append({"role": "system", "content": LOCAL_QUESTION_ANCHOR})

        text = await self._complete(
            messages=messages,
            temperature=0.75,
            max_tokens=SPOKEN_REPLY_MAX_TOKENS,
            timeout=TURN_TIMEOUT_SEC,
        )
        if text is None:
            logger.error("Conversation response failed on every backend.")
            return "Could you tell me more about your approach there?"

        reply = _trim_to_sentences(_clean_spoken_response(text))
        # Final guarantee that the turn ends on a question. _trim_to_sentences can
        # only recover a question the model actually wrote; when a weak local
        # model acknowledges the answer but asks nothing, there is none to
        # recover — and a question-less turn stalls the interview. Pair the reply
        # with a generic probe rather than hand the candidate a dead end.
        if "?" not in reply:
            reply = f"{reply} Can you walk me through your thinking there?".strip()
        return reply

    async def _generate_wrap_up(self, session: InterviewSession) -> str:
        prompt = f"""You are wrapping up a mock interview as a senior hiring manager.
Generate a brief, warm closing (1-2 sentences). Thank the candidate for their time, mention that you'll now provide detailed feedback, and wish them well. Keep it natural and encouraging.{BREVITY_RULE}"""
        text = await self._call_llm(prompt)
        return _trim_to_sentences(text, max_words=35)

    async def _call_llm(self, prompt: str) -> str:
        text = await self._complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=SPOKEN_REPLY_MAX_TOKENS,
            timeout=TURN_TIMEOUT_SEC,
        )
        if text is None:
            logger.error("LLM call failed on every backend.")
            return "Thanks for sharing that. Let's continue — tell me more about your experience."
        return _clean_spoken_response(text)


# Global instance
conversation_engine = ConversationEngine()
