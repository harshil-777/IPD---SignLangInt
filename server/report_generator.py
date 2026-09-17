"""Abstraction over the LLM used to turn compiled sign-language answers
into a formal incident-report paragraph.

Kept behind an interface (open/closed principle) so the backend isn't
locked to one provider — Groq today, something else tomorrow — without
changing `server/main.py`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

REPORT_PROMPT_TEMPLATE = """\
You are an expert Sign Language Translator drafting an Incident Report narrative.
A deaf complainant has provided the following signs across multiple questions.
"N/A" means the question was skipped.

{compiled_inputs}

Rules:
1. Single letters (e.g., K E N I L) must be combined into names.
2. Write ONE cohesive, formal, and legally appropriate paragraph summarizing the entire incident based on the inputs provided.
3. Do not invent details that were not signed. Ignore "N/A" entries.
4. Output ONLY the final formal paragraph. No greetings or bullet points.
"""


class ReportGenerator(ABC):
    @abstractmethod
    def generate(self, compiled_inputs: str) -> str: ...


class GroqReportGenerator(ReportGenerator):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    def generate(self, compiled_inputs: str) -> str:
        prompt = REPORT_PROMPT_TEMPLATE.format(compiled_inputs=compiled_inputs)
        chat_completion = self._client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self._model,
        )
        content = chat_completion.choices[0].message.content
        return content.strip() if content else ""


class UnavailableReportGenerator(ReportGenerator):
    """Used when no LLM API key is configured, so the endpoint stays
    functional with a clear message instead of raising.
    """

    def generate(self, compiled_inputs: str) -> str:
        return "Report generation is not configured on the backend (missing GROQ_API_KEY)."

