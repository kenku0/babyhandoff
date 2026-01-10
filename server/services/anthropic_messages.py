from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class AnthropicError(RuntimeError):
    pass


def messages_create(
    *,
    api_key: str,
    model: str,
    user_text: str,
    timeout_s: float = 30.0,
    max_tokens: int = 600,
) -> dict[str, Any]:
    """
    Minimal Anthropic Messages API call without extra deps.
    https://docs.anthropic.com/en/api/messages
    """
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "messages": [{"role": "user", "content": user_text}],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
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
        raise AnthropicError(f"HTTP {getattr(e, 'code', '?')}: {body or str(e)}") from e
    except Exception as e:
        raise AnthropicError(str(e)) from e


def extract_text(response: dict[str, Any]) -> str:
    """
    Best-effort extraction of plain text from a Messages API payload.
    """
    content = response.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                chunks.append(str(item["text"]))
        if chunks:
            return "\n".join(chunks).strip()
    return ""

