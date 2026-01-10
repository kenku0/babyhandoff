from __future__ import annotations

import json
import re
from typing import Any


class LLMJSONError(ValueError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort parsing for LLM outputs that should be JSON.

    Handles common cases like fenced code blocks and extra leading/trailing text.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise LLMJSONError("Empty model response")

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            raise LLMJSONError(f"Invalid JSON object: {e}") from e

    raise LLMJSONError("Could not parse a JSON object from model output")

