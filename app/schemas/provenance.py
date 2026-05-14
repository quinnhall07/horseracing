"""
app/schemas/provenance.py
─────────────────────────
ModelProvenance: a compact description of which bootstrap run produced the
on-disk model artifacts. Surfaced on every prediction-bearing response so the
UI can render a clear "these models are synthetic" banner when needed.

The bootstrap scripts (`scripts/bootstrap_models.py`, `scripts/quick_bootstrap.py`)
write `models/<run>/BOOTSTRAP_PROVENANCE.json` with this exact shape.
`InferenceArtifacts.load` reads it back at FastAPI startup and the API
endpoints copy it into their response models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelProvenance(BaseModel):
    """Description of the bootstrap run that produced the loaded models."""

    is_synthetic: bool
    trained_at: Optional[str] = None
    n_train_rows: Optional[int] = None
    n_calib_rows: Optional[int] = None
    n_test_rows: Optional[int] = None
    sub_models: list[str] = Field(default_factory=list)
    stub_sub_models: list[str] = Field(default_factory=list)
    meta_learner_test_ece: Optional[float] = None
    meta_learner_test_brier: Optional[float] = None
    bootstrap_script: Optional[str] = None
    bootstrap_seed: Optional[int] = None
    parquet_path: Optional[str] = None
    warning: Optional[str] = None


__all__ = ["ModelProvenance"]
