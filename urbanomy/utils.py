"""Dependency-free payload and message conversions shared by domain modules."""

import math
from collections.abc import Mapping
from typing import Any


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return str(value)


def message_to_text(message: Any) -> str:
    """Read plain text or text blocks without depending on an LLM framework."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping) and item.get("text"):
                chunks.append(str(item["text"]))
        return "\n".join(chunks)
    return str(content)


__all__ = ["json_ready", "message_to_text"]
