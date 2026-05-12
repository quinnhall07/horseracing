"""Synthetic training-data fixtures shared across model tests.

Builds a plausible long-form parquet-shaped DataFrame so we can exercise the
training pipeline end-to-end without touching the real 2.3M-row parquet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_results(
    n_horses: int = 40,
    n_races_per_horse: int = 12,
    seed: int = 7,
) -> pd.DataFrame:
    """Return a DataFrame mimicking the columns of training_<date>.parquet.

    Speed figures are correlated with finish position so a trained model has
    actual signal to learn — this lets us assert val AUC > 0.5 rather than
    chasing random noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start_date = pd.Timestamp("2020-01-01")

    # Each horse gets a latent ability; speed_figure ≈ ability + noise, and the
    # field's winner is the highest-speed-figure horse (with some noise).
    for h in range(n_horses):
        ability = rng.normal(80, 12)
        jurisdiction = rng.choice(["UK", "HK", "JP"], p=[0.45, 0.20, 0.35])
        jockey = f"jockey_{rng.integers(0, 12)}"
        trainer = f"trainer_{rng.integers(0, 8)}"
        for r in range(n_races_per_horse):
            race_date = start_date + pd.Timedelta(days=int(rng.integers(0, 1500)))
            rows.append({
                "horse_name": f"horse_{h:03d}",
                "jurisdiction": jurisdiction,
                "ability": ability,
                "race_date": race_date,
                "track_code": rng.choice(["AAA", "BBB", "CCC"]),
                "race_number": int(rng.integers(1, 10)),
                "distance_furlongs": float(rng.choice([5.0, 6.0, 8.0, 10.0])),
                "surface": rng.choice(["dirt", "turf"]),
                "condition": rng.choice(["fast", "good", "soft"]),
                "race_type": rng.choice(["claiming", "allowance", "stakes"]),
                "claiming_price": None,
                "purse_usd": float(rng.choice([30_000, 60_000, 100_000])),
                "field_size": None,  # let prepare_training_features derive it
                "post_position": int(rng.integers(1, 13)),
                "finish_position": None,  # set below
                "weight_lbs": float(rng.normal(120, 5)),
                "odds_final": float(np.clip(rng.exponential(8.0), 1.5, 80.0)),
                "speed_figure": float(np.clip(ability + rng.normal(0, 8), 0, 130)),
                "speed_figure_source": "brisnet",
                "fraction_q1_sec": None,
                "fraction_q2_sec": None,
                "fraction_finish_sec": float(rng.normal(70.0, 2.0)),
                "beaten_lengths_q1": None,
                "beaten_lengths_q2": None,
                "data_quality_score": 0.9,
                "foaling_year": None,
                "sire": None,
                "dam_sire": None,
                "jockey_name": jockey,
                "trainer_name": trainer,
            })

    df = pd.DataFrame(rows)
    # Group into races: pair horses by date+track+race_number into shared races
    # so the prepare_training_features field_size derivation finds plural rows.
    df["_race_key"] = (
        df["race_date"].dt.strftime("%Y%m%d") + "|"
        + df["track_code"] + "|" + df["race_number"].astype(str)
    )

    # For each race, assign finish positions ordered by (speed_figure descending)
    # with some randomness so the win label has nontrivial signal.
    rng2 = np.random.default_rng(seed + 1)
    new_finish = np.empty(len(df), dtype=float)
    for _, idx in df.groupby("_race_key").indices.items():
        idx = list(idx)
        speeds = df.loc[idx, "speed_figure"].to_numpy() + rng2.normal(0, 3, len(idx))
        order = np.argsort(-speeds)
        ranks = np.empty_like(order)
        for rank, pos in enumerate(order, start=1):
            ranks[pos] = rank
        for j, raw_idx in enumerate(idx):
            new_finish[raw_idx] = float(ranks[j])
    df["finish_position"] = new_finish.astype(int)

    df = df.drop(columns=["_race_key", "ability"])
    return df
