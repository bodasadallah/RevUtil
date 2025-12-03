"""Helpers for text preprocessing inside the demo layer."""

from __future__ import annotations

from typing import List

from inference_engine import split_review_text as _engine_split_review_text


def split_review_text(raw_text: str) -> List[str]:
    """Public shim that re-exports the robust splitter from `inference_engine`."""

    return _engine_split_review_text(raw_text)
