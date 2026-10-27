"""Validated LLM scoring independent of geospatial and optimization packages."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from pydantic import BaseModel, Field

from urbanomy.utils import json_ready as _json_ready
from urbanomy.utils import message_to_text as _message_to_text


class Evaluation(BaseModel):
    score: float = Field(ge=0.0, le=1.0)


class StrategicAlignmentScorer:
    """Score scenarios with an LLM or baseline."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        baseline: Any | None = None,
        prompt: str,
    ) -> None:
        prompt_text = str(prompt).strip()
        if not prompt_text:
            raise ValueError("prompt must be a non-empty string.")
        if llm is None and baseline is None:
            raise ValueError("Either llm or baseline must be provided.")
        self._llm = llm
        self.baseline = baseline
        self.prompt = prompt_text
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def llm(self) -> Any | None:
        return self._llm

    def __deepcopy__(self, memo: dict[int, Any]):
        """Keep the underlying LLM by reference to avoid deepcopy/pickle failures in pymoo history."""
        copied = self.__class__.__new__(self.__class__)
        memo[id(self)] = copied
        copied._llm = self._llm
        copied.baseline = self.baseline
        copied.prompt = self.prompt
        copied._cache = copy.deepcopy(self._cache, memo)
        return copied

    def _invoke(self, prompt: str) -> Any:
        if self.baseline is not None:
            if hasattr(self.baseline, "invoke_state"):
                return self.baseline.invoke_state(prompt).get("output", "")
            response = self.baseline.invoke(prompt)
        else:
            response = self.llm.invoke(prompt)
        return response

    def _build_prompt(self, *, candidate_payload: Mapping[str, Any]) -> str:
        return (
            f"{self.prompt}\n\n"
            "Сценарий:\n"
            f"{json.dumps(_json_ready(candidate_payload), ensure_ascii=False, separators=(',', ':'))}"
        )

    def _parse_response(self, raw_response: Any) -> float:
        if hasattr(raw_response, "content"):
            payload = json.loads(_message_to_text(raw_response))
        elif isinstance(raw_response, BaseModel):
            payload = raw_response.model_dump()
        elif isinstance(raw_response, Mapping):
            payload = dict(raw_response)
        else:
            payload = json.loads(_message_to_text(raw_response))
        return float(Evaluation.model_validate(payload).score)

    def score_candidate(
        self,
        *,
        params_repaired: Mapping[str, Any],
        land_value_gain: float,
        investor_npv: float,
    ) -> dict[str, Any]:
        candidate_payload = {
            "params_repaired": _json_ready(params_repaired),
            "land_value_gain": float(land_value_gain),
            "investor_npv": float(investor_npv),
        }
        cache_key = json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(candidate_payload=candidate_payload)
        result = {
            "score": self._parse_response(self._invoke(prompt)),
            "candidate_payload": candidate_payload,
        }
        self._cache[cache_key] = result
        return result
