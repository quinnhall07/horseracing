"""
app/api/v1/cards.py
───────────────────
GET /api/v1/cards/{card_id} — hydrate a persisted RaceCard with
calibrated per-horse win probabilities and edge metrics.

The endpoint reads the persisted card from the live ingestion DB,
runs the Stream A inference pipeline (sub-models → meta-learner →
calibrator), and fills `model_prob`, `market_prob`, `edge` on each
HorseEntry in the response. The hydration is in-memory only — the
DB row is not touched.

Caching
───────
Because inference is deterministic given (card, artifacts), the
per-race calibrated probabilities are memoised on `app.state.inference_cache`
keyed by `card_id`. Repeated GETs reuse the cached vectors.

Failure modes
─────────────
  404 — card_id not found in DB
  503 — `app.state.artifacts` is None (models not loaded at startup)
  502 — inference failed unexpectedly mid-card (sub-model crash, etc.)
"""

from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.persistence import load_card
from app.db.session import get_session
from app.schemas.race import RaceCard
from app.services.inference.pipeline import (
    InferenceArtifacts,
    build_inference_features,
    infer_calibrated_win_probs,
)

router = APIRouter()
log = get_logger(__name__)


def _require_artifacts(request: Request) -> InferenceArtifacts:
    """Pull artifacts off app.state; raise 503 if missing."""
    artifacts: Optional[InferenceArtifacts] = getattr(
        request.app.state, "artifacts", None
    )
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Inference artifacts not loaded. Ensure models/baseline_full "
                "exists at startup or set HRBS_MODELS_DIR."
            ),
        )
    return artifacts


def _hydrate_card(
    card: RaceCard, race_probs: dict[str, np.ndarray]
) -> RaceCard:
    """Fill model_prob / market_prob / edge on every horse in-place.

    The race_probs vectors are ordered by post_position (Stream A
    invariant from analyze_card / infer_calibrated_win_probs).
    """
    for race in card.races:
        race_id = (
            race.header.race_date.strftime("%Y%m%d")
            + "|" + str(race.header.track_code or "??")
            + "|" + str(race.header.race_number)
        )
        probs = race_probs.get(race_id)
        if probs is None:
            continue
        ordered = sorted(race.entries, key=lambda e: e.post_position)
        for entry, p in zip(ordered, probs):
            entry.model_prob = float(p)
            if entry.morning_line_odds and entry.morning_line_odds > 0:
                mp = 1.0 / float(entry.morning_line_odds)
                entry.market_prob = mp
                entry.edge = float(p) - mp
            else:
                entry.market_prob = None
                entry.edge = None
    return card


def _compute_race_probs_sync(
    card: RaceCard, artifacts: InferenceArtifacts
) -> dict[str, np.ndarray]:
    """Synchronous (CPU-bound) inference path — invoked in a thread."""
    feats = build_inference_features(card)
    if feats.empty:
        return {}

    race_probs: dict[str, np.ndarray] = {}
    for race_id, group in feats.groupby("race_id", sort=False):
        ordered = group.sort_values("post_position").reset_index(drop=True)
        try:
            probs = infer_calibrated_win_probs(ordered, artifacts, str(race_id))
        except Exception as exc:  # noqa: BLE001
            log.error(
                "cards.inference_failed",
                race_id=race_id,
                error=str(exc),
            )
            continue
        race_probs[str(race_id)] = probs
    return race_probs


@router.get("/{card_id}", response_model=RaceCard, status_code=status.HTTP_200_OK)
async def get_card(
    card_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> RaceCard:
    """Return the persisted RaceCard with per-horse calibrated probabilities."""
    artifacts = _require_artifacts(request)

    try:
        card_pk = int(card_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"card_id {card_id!r} is not a valid integer",
        )

    card = await load_card(session, card_pk)
    if card is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"card_id {card_id} not found",
        )

    # In-memory cache (per FastAPI app instance). We always read the dict
    # back off app.state so mutations propagate — using a local `or {}`
    # creates an orphan dict that the next request can't see.
    if not hasattr(request.app.state, "inference_cache") or \
            request.app.state.inference_cache is None:
        request.app.state.inference_cache = {}
    cache: dict[str, dict[str, np.ndarray]] = request.app.state.inference_cache

    cached = cache.get(card_id)
    if cached is None:
        loop = asyncio.get_running_loop()
        try:
            cached = await loop.run_in_executor(
                None, _compute_race_probs_sync, card, artifacts
            )
        except Exception as exc:  # noqa: BLE001
            log.error("cards.inference_unhandled", error=str(exc), card_id=card_id)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"inference failed: {type(exc).__name__}: {exc}",
            )
        cache[card_id] = cached

    hydrated = _hydrate_card(card, cached)
    hydrated.model_provenance = artifacts.provenance
    return hydrated


__all__ = ["router"]
