from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class GeminiError(RuntimeError):
    pass


def generate_content(
    *,
    api_key: str,
    model: str,
    user_text: str,
    timeout_s: float = 30.0,
    max_output_tokens: int = 600,
) -> dict[str, Any]:
    """
    Minimal Google Gemini (AI Studio) generateContent call without extra deps.
    https://ai.google.dev/api/rest/v1beta/models/generateContent
    """
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model)}:generateContent"
    url = f"{base}?key={urllib.parse.quote_plus(api_key)}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_output_tokens},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        raise GeminiError(f"HTTP {getattr(e, 'code', '?')}: {body or str(e)}") from e
    except Exception as e:
        raise GeminiError(str(e)) from e


def extract_text(response: dict[str, Any]) -> str:
    """
    Best-effort extraction of plain text from a generateContent payload.
    """
    candidates = response.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        if isinstance(content, dict):
            parts = content.get("parts")
            if isinstance(parts, list):
                chunks: list[str] = []
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        chunks.append(str(p["text"]))
                if chunks:
                    return "\n".join(chunks).strip()
    return ""

