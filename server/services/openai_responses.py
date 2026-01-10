from __future__ import annotations

import json
import urllib.request
from typing import Any


class OpenAIError(RuntimeError):
    pass


def responses_create(*, api_key: str, model: str, input_text: str, timeout_s: float = 30.0) -> dict[str, Any]:
    """
    Minimal OpenAI Responses API call without extra deps.
    Falls back to caller if the endpoint/model is unavailable.
    """
    url = "https://api.openai.com/v1/responses"
    payload = {"model": model, "input": input_text}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception as e:
        raise OpenAIError(str(e)) from e


def extract_text(response: dict[str, Any]) -> str:
    """
    Best-effort extraction of plain text from a Responses API payload.
    """
    output = response.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "output_text" and "text" in c:
                        chunks.append(str(c["text"]))
        if chunks:
            return "\n".join(chunks).strip()
    # Fallback: some SDKs/variants may return `output_text`
    if "output_text" in response:
        return str(response["output_text"]).strip()
    return ""

