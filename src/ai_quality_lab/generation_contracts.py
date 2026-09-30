"""Contracts reserved for AI Quality Lab v0.2.

Generation evaluation is deliberately separated from retrieval metrics so that
retrieval can remain deterministic and provider-agnostic.
"""
from __future__ import annotations

from typing import Protocol


class GenerationEvaluator(Protocol):
    name: str

    def evaluate(self, *, question: str, answer: str, contexts: list[str]) -> dict[str, float]:
        """Return named generation-quality metrics in the [0, 1] interval."""
        ...
