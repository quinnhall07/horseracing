"""Stream A — server-side inference pipeline.

Re-exports the public surface from `pipeline.py` for ergonomic imports:

    from app.services.inference import (
        InferenceArtifacts,
        analyze_card,
        infer_calibrated_win_probs,
        race_card_to_features,
    )
"""

from app.services.inference.pipeline import (
    InferenceArtifacts,
    analyze_card,
    build_inference_features,
    build_portfolio_from_candidates,
    infer_calibrated_win_probs,
    race_card_to_features,
)

__all__ = [
    "InferenceArtifacts",
    "analyze_card",
    "build_inference_features",
    "build_portfolio_from_candidates",
    "infer_calibrated_win_probs",
    "race_card_to_features",
]
