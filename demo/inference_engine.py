"""Utility helpers to run RevUtil inference inside interactive demos."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import get_prompt  # noqa: E402

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional heavy dependency
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover - enables --mock mode without GPU setup
    LLM = SamplingParams = LoRARequest = None  # type: ignore


ASPECTS: List[str] = [
    "actionability",
    "grounding_specificity",
    "verifiability",
    "helpfulness",
]


_MARKER_PATTERNS = [
    re.compile(r"^(?:[-*•‣▪◦·—–]+)\s+(?P<body>.+)$"),
    re.compile(r"^\(?\d+[)\.:\-]?\s+(?P<body>.+)$"),
    re.compile(r"^\(?[A-Za-z][)\.:]\s+(?P<body>.+)$"),
    re.compile(r"^\(?[IVXLCDM]+[)\.]\s+(?P<body>.+)$", re.IGNORECASE),
    re.compile(r"^\.+\s+(?P<body>.+)$"),
]

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z0-9])")

_ABBREVIATION_RE = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|fig|sec|al|mr|mrs|ms|dr|prof|dept|inc|co|jr|sr))\.$",
    re.IGNORECASE,
)

_MARKER_TOKENS = {
    "-",
    "--",
    "—",
    "–",
    "*",
    "•",
    "‣",
    "▪",
    "◦",
}

_SCHEMA_KEYS = [
    "actionability_label",
    "grounding_specificity_label",
    "verifiability_label",
    "helpfulness_label",
    "actionability_rationale",
    "grounding_specificity_rationale",
    "verifiability_rationale",
    "helpfulness_rationale",
]

_SCHEMA_FIELD_RE = re.compile(
    r'"?(?P<key>'
    + "|".join(_SCHEMA_KEYS)
    + r')"?\s*[:=]\s*(?P<value>"[^"]*"|\'[^\']*\'|[^,\n\r}]+)',
    re.IGNORECASE,
)


def split_review_text(raw_text: str) -> List[str]:
    """Split review text by bullet/number markers, then sentence cues, without loss."""

    if not raw_text:
        return []

    normalized = re.sub(r"\r\n?", "\n", raw_text).strip()
    if not normalized:
        return []

    points: List[str] = []
    current: List[str] = []

    def flush_current():
        if not current:
            return
        segment = " ".join(current).strip()
        if segment:
            points.append(segment)
        current.clear()

    def strip_marker(line: str) -> str | None:
        for pattern in _MARKER_PATTERNS:
            match = pattern.match(line)
            if match:
                return match.group("body").strip()
        return None

    for line in normalized.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_current()
            continue

        payload = strip_marker(stripped)
        if payload is not None:
            flush_current()
            if payload:
                current.append(payload)
            continue

        current.append(stripped)

    flush_current()

    if not points:
        points = [normalized]

    if len(points) <= 1:
        raw_chunks = [
            chunk.strip()
            for chunk in _SENTENCE_BOUNDARY_RE.split(normalized)
            if chunk.strip()
        ]

        merged_chunks: List[str] = []
        for chunk in raw_chunks:
            if merged_chunks and _ABBREVIATION_RE.search(merged_chunks[-1].rstrip()):
                merged_chunks[-1] = f"{merged_chunks[-1]} {chunk}".strip()
            else:
                merged_chunks.append(chunk)

        if len(merged_chunks) > len(points):
            points = merged_chunks

    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\S+", text)
        return [tok for tok in tokens if tok not in _MARKER_TOKENS]

    source_tokens = tokenize(normalized)
    segmented_tokens: List[str] = []
    for point in points:
        segmented_tokens.extend(tokenize(point))

    if len(segmented_tokens) < len(source_tokens):
        missing_tokens = source_tokens[len(segmented_tokens) :]
        if missing_tokens:
            remainder = " ".join(missing_tokens)
            if points:
                points[-1] = f"{points[-1]} {remainder}".strip()
            else:
                points = [remainder]

    return points


@dataclass
class DemoConfig:
    adapter_name: str = "boda/RevUtil_Llama-3.1-8B-Instruct_score_rationale"
    base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    aspect: str = "all"
    generation_type: str = "score_rationale"
    prompt_type: str = "instruction"
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0


class RevUtilEngine:
    """Wraps the RevUtil inference helpers for interactive usage."""

    def __init__(self, config: DemoConfig | None = None, *, mock: bool = False):
        self.config = config or DemoConfig()
        self.mock = mock or bool(os.getenv("REVUTIL_MOCK", ""))
        self._sampling_params = None
        self._llm = None
        self._cached_lora_request = None
        self._warmed_up = False

        if not self.mock:
            if LLM is None:
                raise RuntimeError(
                    "vLLM is not available. Install dependencies or use --mock mode."
                )
            self._sampling_params = SamplingParams(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
            )
            self._llm = LLM(
                model=self.config.base_model_name,
                enable_lora=True,
                max_lora_rank=64,
                tensor_parallel_size=int(os.getenv("REVUTIL_TENSOR_PARALLEL", 1)),
                gpu_memory_utilization=float(os.getenv("REVUTIL_GPU_UTIL", 0.9)),
                trust_remote_code=True,
            )

    def _get_lora_request(self):  # pragma: no cover - trivial cache
        if self.mock or LoRARequest is None:
            return None
        if self._cached_lora_request is None:
            self._cached_lora_request = LoRARequest(
                "revutil_adapter", 1, self.config.adapter_name
            )
        return self._cached_lora_request

    def _build_prompt(self, review_point: str) -> str:
        row = {"review_point": review_point}
        prompt_row = get_prompt(
            row,
            aspect=self.config.aspect,
            task="evaluation",
            generation_type=self.config.generation_type,
            prompt_type=self.config.prompt_type,
            finetuning_type="adapters",
            model=self.config.base_model_name,
        )
        return prompt_row["text"]

    def _mock_generation(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        fake_payload = {
            "actionability_label": "4",
            "grounding_specificity_label": "4",
            "verifiability_label": "3",
            "helpfulness_label": "4",
            "actionability_rationale": "Actionable next steps are clearly outlined.",
            "grounding_specificity_rationale": "Feedback cites concrete evidence from the submission.",
            "verifiability_rationale": "Most claims can be traced back to the manuscript.",
            "helpfulness_rationale": "Guidance is constructive and respectful.",
        }
        return [{"generated_text": json.dumps(fake_payload)} for _ in range(batch_size)]

    @staticmethod
    def _normalize_outputs(raw_outputs: List[Any]) -> List[Dict[str, Any]]:
        """Convert heterogeneous backend outputs into the legacy dict format."""

        normalized: List[Dict[str, Any]] = []
        for item in raw_outputs:
            if hasattr(item, "outputs"):
                text = item.outputs[0].text if getattr(item, "outputs", None) else ""
                normalized.append({"generated_text": text})
            elif isinstance(item, dict):
                if "generated_text" in item:
                    normalized.append(item)
                elif "outputs" in item and item["outputs"]:
                    normalized.append({"generated_text": item["outputs"][0]["text"]})
                else:
                    normalized.append({"generated_text": json.dumps(item)})
            else:
                normalized.append({"generated_text": str(item)})
        return normalized

    @staticmethod
    def _empty_prediction() -> Dict[str, Any]:
        return {
            "actionability_label": None,
            "grounding_specificity_label": None,
            "verifiability_label": None,
            "helpfulness_label": None,
            "actionability_rationale": None,
            "grounding_specificity_rationale": None,
            "verifiability_rationale": None,
            "helpfulness_rationale": None,
        }

    @staticmethod
    def _try_literal_or_json(text: str) -> Dict[str, Any] | None:
        candidates = [text]
        if text and "'" in text:
            candidates.append(text.replace("\\'", "'"))
            candidates.append(text.replace("'", '"'))

        for candidate in candidates:
            try:
                literal = ast.literal_eval(candidate)
                if isinstance(literal, dict):
                    return literal
            except (ValueError, SyntaxError):
                pass
            try:
                decoded = json.loads(candidate)
                if isinstance(decoded, dict):
                    return decoded
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _parse_generated_payload(payload: str) -> Dict[str, Any] | None:
        if not payload:
            return None

        text = payload.strip()
        if not text:
            return None

        parsed = RevUtilEngine._try_literal_or_json(text)
        if parsed:
            return parsed

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = RevUtilEngine._try_literal_or_json(match.group(0))
            if parsed:
                return parsed

        return None

    @staticmethod
    def _schema_fallback_parse(payload: str) -> Dict[str, Any] | None:
        if not payload:
            return None

        extracted: Dict[str, Any] = {}
        for match in _SCHEMA_FIELD_RE.finditer(payload):
            key = match.group("key").lower()
            value = match.group("value").strip()
            if value.endswith((",", ";")):
                value = value[:-1].strip()
            if (
                (value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))
            ) and len(value) >= 2:
                value = value[1:-1]
            extracted[key] = value

        return extracted or None

    def _extract_predictions(
        self, normalized_outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        predictions: List[Dict[str, Any]] = []
        for entry in normalized_outputs:
            payload = entry.get("generated_text", "")
            parsed = self._parse_generated_payload(payload)
            if not parsed:
                parsed = self._schema_fallback_parse(payload)
            if not parsed:
                predictions.append(self._empty_prediction())
                continue

            def _get(key: str):
                value = parsed.get(key)
                return str(value) if value is not None else None

            predictions.append(
                {
                    "actionability_label": _get("actionability_label"),
                    "grounding_specificity_label": _get("grounding_specificity_label"),
                    "verifiability_label": _get("verifiability_label"),
                    "helpfulness_label": _get("helpfulness_label"),
                    "actionability_rationale": _get("actionability_rationale"),
                    "grounding_specificity_rationale": _get(
                        "grounding_specificity_rationale"
                    ),
                    "verifiability_rationale": _get("verifiability_rationale"),
                    "helpfulness_rationale": _get("helpfulness_rationale"),
                }
            )
        return predictions

    @staticmethod
    def _structure_aspects(analysis: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            aspect: {
                "score": analysis.get(f"{aspect}_label"),
                "rationale": analysis.get(f"{aspect}_rationale"),
            }
            for aspect in ASPECTS
        }

    def _batched_predictions(self, prompts: List[str]) -> List[Dict[str, Any]]:
        if not prompts:
            return []

        if self.mock:
            outputs = self._mock_generation(len(prompts))
        else:
            outputs = self._llm.generate(  # type: ignore[union-attr]
                prompts=prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
                lora_request=self._get_lora_request(),
            )

        normalized_outputs = self._normalize_outputs(outputs)

        if len(normalized_outputs) != len(prompts):  # pragma: no cover - defensive
            logger.warning(
                "Mismatch between prompts (%s) and outputs (%s); padding/truncating.",
                len(prompts),
                len(normalized_outputs),
            )
            if len(normalized_outputs) < len(prompts):
                normalized_outputs.extend(
                    [
                        {"generated_text": ""}
                        for _ in range(len(prompts) - len(normalized_outputs))
                    ]
                )
            else:
                normalized_outputs = normalized_outputs[: len(prompts)]

        parsed = self._extract_predictions(normalized_outputs)

        if not self.mock:
            self._warmed_up = True

        return parsed

    def warm_up(self, review_point: str | None = None) -> bool:
        """Run a single dummy pass to ensure the model weights are loaded."""

        if self.mock or self._warmed_up or self._llm is None:
            return False

        placeholder = review_point or "Warm-up review input to initialize RevUtil."
        prompt = self._build_prompt(placeholder)

        try:
            self._llm.generate(  # type: ignore[union-attr]
                prompts=[prompt],
                sampling_params=self._sampling_params,
                use_tqdm=False,
                lora_request=self._get_lora_request(),
            )
            self._warmed_up = True
            logger.info("RevUtilEngine warm-up completed successfully.")
            return True
        except Exception as exc:  # pragma: no cover - warm-up best effort
            logger.warning("RevUtilEngine warm-up failed: %s", exc)
            return False

    def analyze_review(self, review_point: str) -> Dict[str, Any]:
        if not review_point.strip():
            raise ValueError("Review text cannot be empty.")

        prompt = self._build_prompt(review_point)

        parsed = self._batched_predictions([prompt])

        if not parsed:
            raise RuntimeError("Model returned no predictions.")

        analysis = parsed[0]
        structured = self._structure_aspects(analysis)
        return {"aspects": structured, "prompt": prompt}

    def analyze_points(self, review_points: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple review points using a single batched generation call."""

        normalized_points = [
            point.strip() for point in review_points if point and point.strip()
        ]
        if not normalized_points:
            raise ValueError("At least one non-empty review point is required.")

        prompts = [self._build_prompt(point) for point in normalized_points]
        parsed_batches = self._batched_predictions(prompts)

        results: List[Dict[str, Any]] = []
        for idx, (point, prompt, analysis) in enumerate(
            zip(normalized_points, prompts, parsed_batches), start=1
        ):
            structured = self._structure_aspects(analysis)
            results.append(
                {"index": idx, "text": point, "prompt": prompt, "aspects": structured}
            )

        return results


__all__ = ["RevUtilEngine", "DemoConfig", "ASPECTS", "split_review_text"]
