"""AI Interviewer Module - Resume parsing and LLM question generation."""
import os
import json
import logging
import pdfplumber
from openai import AsyncOpenAI

from . import llm_config

logger = logging.getLogger(__name__)

# Reasoning models (gpt-oss-20b on both NVIDIA and Groq) spend completion tokens
# on a hidden reasoning trace BEFORE they emit a single visible character. At the
# old cap of 600 the trace consumed the entire budget: finish_reason came back
# "length", `content` came back None, extract_json failed, and every session
# silently fell through to _backup_questions() — which is why the interviewer
# sounded generic and never named a project. llama-3.1-8b hid this for a year
# because it had no trace and spent all 600 tokens on the answer itself.
# Measured: ~560 completion tokens for a 4-question plan, so this leaves room.
QUESTION_PLAN_MAX_TOKENS = 1500

# The SDK's default wait is 10 minutes, so a stalled vendor froze the upload
# screen. A healthy model answers in 2-10 s; past this, use the backup questions.
QUESTION_PLAN_TIMEOUT_SEC = 30


class AIInterviewer:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        self._client_generation: int | None = None

    def _llm(self) -> AsyncOpenAI:
        # Rebuilt when llm_config's generation changes (a brain switch, key edit
        # or mode flip). A client cached forever kept posting to the backend it
        # was built for, so switching from the dashboard sent the new model name
        # to the old endpoint and every plan fell back to the generic questions.
        gen = llm_config.config_generation()
        if self._client is None or self._client_generation != gen:
            self._client = llm_config.make_client()
            self._client_generation = gen
            logger.info(f"🧠 Interviewer LLM: {llm_config.describe()}")
        return self._client

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extracts and cleans text from an uploaded PDF resume."""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Resume parsing failed: {e}")
            return ""

    async def generate_questions(self, resume_text: str) -> list[dict]:
        """Passes resume to LLM to generate 3-5 tough STAR questions."""
        if not resume_text:
            return [{"question": "Can you walk me through your background and experience?"}]

        prompt = f"""You are a tough but constructive technical interview coach.
Read this candidate's resume and write exactly 4 questions that could ONLY be asked of THIS candidate.

Candidate Resume:
{resume_text[:4000]}

Rules for every question:
- Name something concrete from the resume out loud — the project, product, company, library, or a number they shipped.
- Dig into the decisions behind that specific work: what broke, what they traded away, what the result actually was.
- Still get at situation -> action -> result, but anchored to their real work, never as a template.
- Never write a question you could paste into a different candidate's interview.
- One or two sentences. No preamble, no "tell me about a time" boilerplate.

Return ONLY a valid JSON object with a "questions" array. Format:
{{
  "questions": [
    {{"question": "On [project named in the resume] you went with [specific choice they made] - what broke first, and what did you give up to fix it?", "focus": "System Design"}},
    ...
  ]
}}
Replace the bracketed parts with real details from the resume above.
Do not include markdown blocks, pleasantries, or extra text. Just the JSON object."""

        try:
            # `response_format` is honoured by both backends and is what keeps a
            # 3B local model from answering with prose. Without it the parse
            # fails often enough that every session falls back to the canned
            # questions — which looks like the local path "working".
            model = llm_config.chat_model()
            completion = await self._llm().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=QUESTION_PLAN_MAX_TOKENS,
                response_format={"type": "json_object"},
                extra_body=llm_config.extra_body_for(model),
                timeout=QUESTION_PLAN_TIMEOUT_SEC,
            )
            questions = llm_config.extract_json(completion.choices[0].message.content)

            # Asking for a JSON *object* to get schema adherence means the model
            # may wrap the array in a key of its choosing, so unwrap a lone
            # list-valued field before giving up on it.
            if isinstance(questions, dict):
                for value in questions.values():
                    if isinstance(value, list):
                        questions = value
                        break

            if not isinstance(questions, list) or not questions:
                logger.warning("LLM returned no usable questions; using backups")
                return self._backup_questions()

            valid = [q for q in questions if isinstance(q, dict) and q.get("question")]
            return valid or self._backup_questions()

        except Exception as e:
            logger.error(f"LLM Question Generation failed: {type(e).__name__}: {e}", exc_info=True)
            return self._backup_questions()

    def _backup_questions(self) -> list[dict]:
        return [
            {"question": "Describe a situation where you had to balance technical rigor and foresight in designing a system. What actions did you take, and what were the results?", "focus": "System Design"},
            {"question": "Walk me through a time when you encountered a complex challenge in low-level optimization. How did you approach the problem, and what was the outcome?", "focus": "Low-Level Optimization"},
            {"question": "Tell me about a project where you had to work with a large codebase and implement automated compliance checks. What tools or techniques did you use, and what were the benefits?", "focus": "Code Analysis"},
            {"question": "Recall a situation where you had to communicate technical information to a non-technical audience. How did you tailor your message, and what was the impact?", "focus": "Communication"},
        ]


# Global instance
ai_interviewer = AIInterviewer()
