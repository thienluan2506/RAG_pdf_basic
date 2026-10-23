from __future__ import annotations

from contextvars import ContextVar

from google import genai
from google.genai import types

from config import settings

# Runtime API key override (optional)

_runtime_api_key: ContextVar[str | None] = ContextVar(
    "runtime_gemini_api_key", default=None
)


def set_runtime_gemini_api_key(api_key: str | None) -> None:
    """Override the Gemini API key """
    _runtime_api_key.set(api_key.strip() if isinstance(api_key, str) else None)

def invoke_llm(prompt: str) -> str:
    """Send *prompt* to Gemini and return the plain-text response.

    Args:
        prompt: Fully rendered prompt string (Jinja output).

    Returns:
        Stripped assistant text.

    Raises:
        RuntimeError: API key is missing, or the model returned an empty reply.
    """
    api_key = _runtime_api_key.get() or settings.gemini_api_key
    if not api_key:
        raise RuntimeError(
            "Gemini API key is missing.\n"
            "  • Set GEMINI_API_KEY in your .env file, or\n"
            "  • Visit https://aistudio.google.com/api-keys to get your API key.."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=settings.llm_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_new_tokens,
        ),
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text
