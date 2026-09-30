from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ai_quality_lab.versioning.hashing import sha256_file


def build_manifest(
    *,
    ground_truth_path: str | Path,
    retrieval_path: str | Path,
    ks: tuple[int, ...],
    tool_version: str,
) -> dict:
    gt = Path(ground_truth_path)
    retrieval = Path(retrieval_path)
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": {"name": "ai-quality-lab", "version": tool_version},
        "evaluation": {"ks": list(ks)},
        "artifacts": {
            "ground_truth": {
                "path": str(gt),
                "sha256": sha256_file(gt),
            },
            "retrieval_results": {
                "path": str(retrieval),
                "sha256": sha256_file(retrieval),
            },
        },
    }
