"""
app/services/models/sequence.py
───────────────────────────────
Layer 1c — Sequence (Transformer) Model.

Treats each horse's career as a sequence of PP "tokens" (one per prior race
with surface, distance, speed figure, finish position, days-gap features)
and applies a small Transformer encoder to produce a fixed-length career
embedding. The embedding feeds the meta-learner.

STATUS — SCAFFOLDING ONLY (Phase 3 deferred work).

PyTorch is not yet a runtime dependency. Even with the parquet exported,
training a Transformer on 230K horse sequences requires:
  * GPU access (or a long CPU run)
  * a horse_id that's globally unique across years/jurisdictions — the
    current `horse_name + jurisdiction` key collides occasionally and
    needs the master DB `horses.dedup_key` to be exposed in the export.

The class skeleton lives here so the bootstrap pipeline can reference it
without ImportError. `fit()` raises NotImplementedError. `predict_proba()`
returns 0.5 uniformly — the meta-learner orthogonalisation step will treat
this as no-information until the model is actually trained.

Unblock path: add `torch>=2.2` to the optional `[gpu]` extras group, expose
horses.dedup_key in the parquet export, and implement the encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class SequenceModelConfig:
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_sequence_length: int = 30
    dropout: float = 0.1


class SequenceModel:
    """Placeholder — see module docstring for unblock criteria."""

    ARTIFACT_VERSION: str = "0"

    def __init__(self, config: Optional[SequenceModelConfig] = None):
        self.config = config or SequenceModelConfig()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "SequenceModel":
        raise NotImplementedError(
            "SequenceModel requires PyTorch + a globally-unique horse_id. "
            "See PROGRESS.md for the unblock path."
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.5, dtype=float)

    def save(self, path: Path) -> dict:
        log.warning("sequence.save_called_on_stub", path=str(path))
        return {"artifact_version": self.ARTIFACT_VERSION, "stub": True}

    @classmethod
    def load(cls, path: Path) -> "SequenceModel":
        return cls()


__all__ = ["SequenceModelConfig", "SequenceModel"]
