"""
app/services/models/sequence.py
───────────────────────────────
Layer 1c — Sequence (Transformer) Model.

Each horse's career is modelled as a sequence of past-performance "tokens"
(one per prior race; surface / distance / speed-figure / finish-pct /
days-gap features). A small Transformer encoder produces a fixed-length
career embedding that maps to a single per-row P(win) — same output shape
as every other Layer-1 sub-model, so the meta-learner stacks it identically.

Status — TRAINABLE (ADR-046, supersedes the ADR-026 stub for this layer).

Key invariants:
  * Per ADR-029: only PRIOR races feed the encoder for any given row. Today's
    race contributes scalar "today context" features but never its outcome.
  * Per ADR-003: train / calib / test is time-split externally — this module
    does not re-split.
  * Per ADR-027: per-horse grouping uses `horse_dedup_key` (the master-DB
    SHA-256 key) when present, falling back to the legacy
    `horse_name|jurisdiction` compromise only when the parquet predates the
    column.

Optional dep: torch (declared under `pyproject.toml :: [project.optional-dependencies.gpu]`).
The base install path (no torch) still works — `is_trainable_with(df)`
returns False without torch and `predict_proba()` falls back to 0.5.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.core.logging import get_logger

log = get_logger(__name__)


# ── Public dataclass: config ────────────────────────────────────────────────


@dataclass
class SequenceModelConfig:
    embedding_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    max_sequence_length: int = 30
    dropout: float = 0.1

    # Training hyperparameters.
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    n_epochs: int = 10
    warmup_frac: float = 0.05  # fraction of total steps spent in linear warmup
    grad_clip: float = 1.0
    eval_every_steps: int = 0  # 0 = once per epoch

    # Feature engineering.
    today_context_columns: tuple[str, ...] = (
        "distance_furlongs",
        "field_size",
        "weight_lbs",
        "purse_usd",
    )
    seq_feature_columns: tuple[str, ...] = (
        "speed_figure",
        "distance_furlongs",
        "finish_pct",
        "days_gap",
        "weight_lbs",
        "log_odds_final",
    )
    surface_categories: tuple[str, ...] = ("dirt", "turf", "synthetic", "unknown")

    # Reproducibility.
    seed: int = 42

    # Runtime device override; None → auto-detect.
    device: Optional[str] = None

    @property
    def n_seq_features(self) -> int:
        return len(self.seq_feature_columns) + len(self.surface_categories)

    @property
    def n_today_features(self) -> int:
        return len(self.today_context_columns) + len(self.surface_categories)


# ── Trainability gate (importable without torch present) ────────────────────


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ── Public model class ─────────────────────────────────────────────────────


class SequenceModel:
    """Layer-1c Transformer over a horse's prior-race history.

    Lifecycle:
        m = SequenceModel(config).fit(train_df, val_df=calib_df)
        m.save(path)
        m2 = SequenceModel.load(path)
        proba = m2.predict_proba(any_df)  # 0..1 per row
    """

    ARTIFACT_VERSION: str = "1"

    def __init__(self, config: Optional[SequenceModelConfig] = None):
        self.config: SequenceModelConfig = config or SequenceModelConfig()
        self.is_fitted: bool = False
        # Populated by fit() / load().
        self._encoder = None  # torch.nn.Module
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None
        self._today_scaler_mean: Optional[np.ndarray] = None
        self._today_scaler_std: Optional[np.ndarray] = None
        self._metrics: dict = {}

    # ── Trainability check ────────────────────────────────────────────────

    @classmethod
    def is_trainable_with(cls, df: pd.DataFrame) -> bool:
        """True when (a) torch is importable, and (b) the parquet has a
        horse-identifier column the model can group on."""
        if not _torch_available():
            return False
        if df is None or len(df) == 0:
            return False
        # Either horse_dedup_key (preferred) or the legacy compromise pair.
        has_dedup = "horse_dedup_key" in df.columns and df["horse_dedup_key"].notna().any()
        has_legacy = {"horse_name", "jurisdiction"}.issubset(df.columns)
        return bool(has_dedup or has_legacy)

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "SequenceModel":
        if not _torch_available():
            raise ImportError(
                "torch is required to fit SequenceModel; install with "
                "`pip install -e .[gpu]` (or `pip install torch>=2.2`)."
            )
        import torch  # noqa: E402
        import torch.nn as nn  # noqa: E402

        cfg = self.config
        device = _resolve_device(cfg.device)
        log.info("sequence.fit.start", n_rows=len(train_df), device=str(device))

        # 1. Build per-row sequences + labels + today-context for train.
        Xs_tr, mask_tr, Xt_tr, y_tr = _build_sequences(train_df, cfg)
        if Xs_tr.size == 0:
            raise ValueError("No sequences could be built from train_df.")

        # 2. Fit per-feature standardiser on TRAIN slice only.
        seq_mean, seq_std = _fit_seq_scaler(Xs_tr, mask_tr)
        today_mean, today_std = _fit_today_scaler(Xt_tr)
        self._scaler_mean, self._scaler_std = seq_mean, seq_std
        self._today_scaler_mean, self._today_scaler_std = today_mean, today_std

        Xs_tr_n = _apply_seq_scaler(Xs_tr, mask_tr, seq_mean, seq_std)
        Xt_tr_n = _apply_today_scaler(Xt_tr, today_mean, today_std)

        # 3. Build val tensors if provided. Pre-move to device so the eval
        # loop doesn't re-transfer every call (the MPS backend in torch 2.12
        # has a slow path for repeated host→device copies of identical buffers).
        Xs_va_t = mask_va_t = Xt_va_t = y_va_t = None
        if val_df is not None and len(val_df) > 0:
            Xs_va, mask_va, Xt_va, y_va = _build_sequences(val_df, cfg)
            if Xs_va.size > 0:
                Xs_va_n = _apply_seq_scaler(Xs_va, mask_va, seq_mean, seq_std)
                Xt_va_n = _apply_today_scaler(Xt_va, today_mean, today_std)
                Xs_va_t = torch.as_tensor(Xs_va_n, dtype=torch.float32, device=device)
                mask_va_t = torch.as_tensor(mask_va, dtype=torch.bool, device=device)
                Xt_va_t = torch.as_tensor(Xt_va_n, dtype=torch.float32, device=device)
                y_va_t = torch.as_tensor(y_va, dtype=torch.float32, device=device)

        # 4. Construct + initialise the encoder.
        torch.manual_seed(cfg.seed)
        encoder = _SequenceEncoder(cfg).to(device)
        opt = torch.optim.AdamW(
            encoder.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        # 5. Pre-move TRAIN tensors to device once. Avoids the DataLoader
        # per-batch host→device transfer path that hangs the MPS backend on
        # multi-million-row datasets.
        Xs_t = torch.as_tensor(Xs_tr_n, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask_tr, dtype=torch.bool, device=device)
        Xt_t = torch.as_tensor(Xt_tr_n, dtype=torch.float32, device=device)
        y_t = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
        n_train = Xs_t.shape[0]
        bs = cfg.batch_size
        n_batches = (n_train + bs - 1) // bs

        # Cosine LR schedule with linear warmup.
        total_steps = max(1, cfg.n_epochs * n_batches)
        warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: _cosine_warmup(step, warmup_steps, total_steps),
        )

        # 6. Train loop. Manual index-based shuffle keeps the BatchSampler /
        # DataLoader generator chain out of the picture.
        encoder.train()
        epoch_metrics: list[dict] = []
        global_step = 0
        for epoch in range(cfg.n_epochs):
            running = 0.0
            n_seen = 0
            perm = torch.randperm(n_train, device=device)
            for b in range(n_batches):
                idx = perm[b * bs : (b + 1) * bs]
                xs = Xs_t.index_select(0, idx)
                mk = mask_t.index_select(0, idx)
                xt = Xt_t.index_select(0, idx)
                y = y_t.index_select(0, idx)
                logits = encoder(xs, mk, xt)
                loss = criterion(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                running += float(loss.item()) * xs.shape[0]
                n_seen += xs.shape[0]
                global_step += 1
                # Periodic progress so we know the model is moving even on a
                # multi-hour training run.
                if global_step % 500 == 0:
                    log.info(
                        "sequence.fit.step",
                        epoch=epoch + 1,
                        step=global_step,
                        running_loss=round(running / max(1, n_seen), 4),
                    )
            train_loss = running / max(1, n_seen)

            val_loss = val_brier = val_auc = float("nan")
            if Xs_va_t is not None:
                val_loss, val_brier, val_auc = _eval(
                    encoder, Xs_va_t, mask_va_t, Xt_va_t, y_va_t, device, cfg
                )
                encoder.train()  # _eval flipped to eval mode
            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_brier": val_brier,
                    "val_auc": val_auc,
                }
            )
            log.info(
                "sequence.fit.epoch",
                epoch=epoch + 1,
                train_loss=round(train_loss, 6),
                val_brier=round(val_brier, 6) if not np.isnan(val_brier) else None,
                val_auc=round(val_auc, 4) if not np.isnan(val_auc) else None,
            )

        self._encoder = encoder.cpu()  # keep CPU copy for portable save/load
        self.is_fitted = True
        self._metrics = {
            "n_train_rows": int(len(Xs_t)),
            "n_val_rows": int(0 if Xs_va_t is None else len(Xs_va_t)),
            "epochs": epoch_metrics,
            "device": str(device),
        }
        return self

    # ── Predict ───────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self._encoder is None:
            return np.full(len(df), 0.5, dtype=float)
        if not _torch_available():
            return np.full(len(df), 0.5, dtype=float)
        import torch

        cfg = self.config
        Xs, mask, Xt, _y = _build_sequences(df, cfg, with_labels=False)
        if Xs.size == 0:
            return np.full(len(df), 0.5, dtype=float)
        Xs_n = _apply_seq_scaler(Xs, mask, self._scaler_mean, self._scaler_std)
        Xt_n = _apply_today_scaler(Xt, self._today_scaler_mean, self._today_scaler_std)

        device = _resolve_device(cfg.device)
        encoder = self._encoder.to(device).eval()
        bs = cfg.batch_size
        n = Xs_n.shape[0]
        out = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            xs_all = torch.as_tensor(Xs_n, dtype=torch.float32, device=device)
            mk_all = torch.as_tensor(mask, dtype=torch.bool, device=device)
            xt_all = torch.as_tensor(Xt_n, dtype=torch.float32, device=device)
            for start in range(0, n, bs):
                end = min(n, start + bs)
                logits = encoder(xs_all[start:end], mk_all[start:end], xt_all[start:end])
                out[start:end] = torch.sigmoid(logits).cpu().numpy()
        # Move encoder back to CPU so subsequent .save() picks up CPU tensors.
        self._encoder = encoder.cpu()
        # Rows that contributed no sequence (zero priors) → neutral 0.5.
        # _build_sequences marks them with mask==True for all timesteps.
        no_priors = mask.all(axis=1)
        out[no_priors] = 0.5
        return out.astype(float)

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, path: Path) -> dict:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self.is_fitted or self._encoder is None:
            (path / "metadata.json").write_text(json.dumps(
                {"artifact_version": self.ARTIFACT_VERSION, "stub": True}, indent=2
            ))
            log.warning("sequence.save_called_on_stub", path=str(path))
            return {"artifact_version": self.ARTIFACT_VERSION, "stub": True}

        if not _torch_available():
            raise ImportError("torch required to save a fitted SequenceModel.")
        import torch

        torch.save(self._encoder.state_dict(), path / "encoder.pt")
        np.savez(
            path / "scalers.npz",
            seq_mean=self._scaler_mean,
            seq_std=self._scaler_std,
            today_mean=self._today_scaler_mean,
            today_std=self._today_scaler_std,
        )
        meta = {
            "artifact_version": self.ARTIFACT_VERSION,
            "stub": False,
            "config": asdict(self.config),
            "metrics": self._metrics,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    @classmethod
    def load(cls, path: Path) -> "SequenceModel":
        path = Path(path)
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            return cls()
        meta = json.loads(meta_path.read_text())
        if meta.get("stub", True):
            return cls()
        if not _torch_available():
            log.warning("sequence.load_without_torch", path=str(path))
            return cls()
        import torch

        cfg = SequenceModelConfig(**meta["config"])
        m = cls(cfg)
        encoder = _SequenceEncoder(cfg)
        state = torch.load(path / "encoder.pt", map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)
        m._encoder = encoder
        scalers = np.load(path / "scalers.npz")
        m._scaler_mean = scalers["seq_mean"]
        m._scaler_std = scalers["seq_std"]
        m._today_scaler_mean = scalers["today_mean"]
        m._today_scaler_std = scalers["today_std"]
        m._metrics = meta.get("metrics", {})
        m.is_fitted = True
        return m


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers — torch-dependent code lives below the conditional import so
# the module can be imported on systems without torch (predict_proba falls back
# to 0.5, fit() raises a clean error).
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_device(override: Optional[str]):
    import torch
    if override is not None:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math_cos(math_pi * progress))


# Reach math via numpy so we don't carry `math` as a separate dep just for two
# constants used at scheduler-step granularity.
def math_cos(x: float) -> float:
    return float(np.cos(x))


math_pi = float(np.pi)


def _build_sequences(
    df: pd.DataFrame,
    cfg: SequenceModelConfig,
    with_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (Xs, mask, Xt, y).

    Xs: (N, T, F_seq)  — per-row prior-race tokens, oldest → newest.
    mask: (N, T) bool  — True where padding (no prior in that slot).
    Xt: (N, F_today)   — today-race scalar context.
    y:  (N,)           — labels (0/1). Empty if with_labels=False.

    Rows with zero priors get an all-padding sequence (mask.all() per row) and
    their predicted proba is replaced with 0.5 downstream.
    """
    if df is None or len(df) == 0:
        return (
            np.zeros((0, cfg.max_sequence_length, cfg.n_seq_features), dtype=np.float32),
            np.zeros((0, cfg.max_sequence_length), dtype=bool),
            np.zeros((0, cfg.n_today_features), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Stash the input order so we can restore it before returning.
    work = df.copy()
    # Use a stable positional id that survives sort_values + reset_index.
    work["__orig_pos"] = np.arange(len(work))

    # Per-row horse key (matches prepare_training_features._horse_key fallback chain).
    if "horse_dedup_key" in work.columns and work["horse_dedup_key"].notna().any():
        legacy = work["horse_name"].astype(str) + "|" + work["jurisdiction"].astype(str)
        work["__horse_key"] = work["horse_dedup_key"].astype("string").fillna(legacy)
    else:
        work["__horse_key"] = (
            work["horse_name"].astype(str) + "|" + work["jurisdiction"].astype(str)
        )

    work["race_date"] = pd.to_datetime(work["race_date"]).dt.normalize()

    # Derive scalar columns the sequence tokens need.
    if "field_size" in work.columns:
        fs = pd.to_numeric(work["field_size"], errors="coerce")
    else:
        fs = pd.Series(np.nan, index=work.index)
    if "finish_position" in work.columns:
        fp = pd.to_numeric(work["finish_position"], errors="coerce")
    else:
        fp = pd.Series(np.nan, index=work.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        work["finish_pct"] = (fp / fs).astype(float)

    if "odds_final" in work.columns:
        odds = pd.to_numeric(work["odds_final"], errors="coerce")
        work["log_odds_final"] = np.log(odds.clip(lower=1.01))
    else:
        work["log_odds_final"] = np.nan

    # days_gap = days since the horse's prior race. Filled in below via
    # groupby+shift; today's row's days_gap is NaN (we never use it).
    work = work.sort_values(["__horse_key", "race_date"]).reset_index(drop=True)
    prev_date = work.groupby("__horse_key")["race_date"].shift(1)
    work["days_gap"] = (work["race_date"] - prev_date).dt.days

    # Surface one-hot.
    surf = work.get("surface", pd.Series(["unknown"] * len(work), index=work.index))
    surf = surf.astype(str).str.lower().where(
        surf.astype(str).str.lower().isin(cfg.surface_categories), "unknown"
    )
    surf_dummies = pd.get_dummies(
        pd.Categorical(surf, categories=list(cfg.surface_categories)),
        prefix="surface",
    )
    for cat in cfg.surface_categories:
        col = f"surface_{cat}"
        work[col] = surf_dummies[col].to_numpy().astype(np.float32)

    seq_cols = list(cfg.seq_feature_columns) + [f"surface_{c}" for c in cfg.surface_categories]
    today_cols = list(cfg.today_context_columns) + [f"surface_{c}" for c in cfg.surface_categories]

    # Make sure every seq col exists (parquet caveats: speed_figure / odds_final
    # may be missing in some sources).
    for col in seq_cols + today_cols:
        if col not in work.columns:
            work[col] = np.nan

    # Replace NaN with 0 for downstream tensor work; the scaler is fit on
    # non-padding rows only so NaN-fill bias is absorbed.
    seq_mat = work[seq_cols].to_numpy(dtype=np.float32, copy=True)
    today_mat = work[today_cols].to_numpy(dtype=np.float32, copy=True)
    seq_mat = np.nan_to_num(seq_mat, nan=0.0, posinf=0.0, neginf=0.0)
    today_mat = np.nan_to_num(today_mat, nan=0.0, posinf=0.0, neginf=0.0)

    if with_labels and "finish_position" in work.columns:
        y = (pd.to_numeric(work["finish_position"], errors="coerce") == 1).to_numpy(dtype=np.float32)
    else:
        y = np.zeros(len(work), dtype=np.float32)

    # Build per-row sequences via groupby (oldest → newest) using shift trick:
    # for the i-th row of a horse, the sequence is rows [max(0, i-T) .. i-1].
    T = cfg.max_sequence_length
    F = len(seq_cols)
    n = len(work)
    Xs = np.zeros((n, T, F), dtype=np.float32)
    mask = np.ones((n, T), dtype=bool)  # True = pad; flipped to False as we fill

    # Group indices: for each horse, list of row indices in chronological order.
    group_idx = work.groupby("__horse_key", sort=False).indices
    for _, idxs in group_idx.items():
        idxs = np.asarray(idxs)
        for pos, row_idx in enumerate(idxs):
            if pos == 0:
                continue  # no priors → all-pad row
            start = max(0, pos - T)
            priors = idxs[start:pos]
            seq = seq_mat[priors]  # (P, F)
            P = seq.shape[0]
            # Pack newest-last (chronological): place priors at the end of T slots.
            Xs[row_idx, T - P:T, :] = seq
            mask[row_idx, T - P:T] = False

    # Restore input ordering — `work` was sorted by (__horse_key, race_date),
    # so Xs/mask/today_mat/y currently align to that sorted order. Use the
    # stashed __orig_pos column to invert the permutation in-place.
    orig_pos = work["__orig_pos"].to_numpy()
    inv = np.empty(n, dtype=np.int64)
    inv[orig_pos] = np.arange(n)
    return Xs[inv], mask[inv], today_mat[inv], y[inv]


def _fit_seq_scaler(
    seq: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std over non-padding tokens."""
    flat = seq.reshape(-1, seq.shape[-1])
    flat_mask = (~mask).reshape(-1)
    if flat_mask.sum() == 0:
        F = seq.shape[-1]
        return np.zeros(F, dtype=np.float32), np.ones(F, dtype=np.float32)
    obs = flat[flat_mask]
    mean = obs.mean(axis=0).astype(np.float32)
    std = obs.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def _apply_seq_scaler(
    seq: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    out = (seq - mean) / std
    # Zero out padding so it doesn't poison attention via residual norm noise.
    out[mask] = 0.0
    return out.astype(np.float32)


def _fit_today_scaler(today: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if today.shape[0] == 0:
        F = today.shape[-1]
        return np.zeros(F, dtype=np.float32), np.ones(F, dtype=np.float32)
    mean = today.mean(axis=0).astype(np.float32)
    std = today.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def _apply_today_scaler(
    today: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((today - mean) / std).astype(np.float32)


def _eval(encoder, Xs_t, mask_t, Xt_t, y_t, device, cfg) -> tuple[float, float, float]:
    """Evaluate on validation tensors. Tensors are assumed pre-moved to device."""
    import torch
    encoder.eval()
    bs = cfg.batch_size
    n = Xs_t.shape[0]
    probs = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(n, start + bs)
            logits = encoder(Xs_t[start:end], mask_t[start:end], Xt_t[start:end])
            probs[start:end] = torch.sigmoid(logits).cpu().numpy()
    y = y_t.cpu().numpy()
    eps = 1e-7
    bce = float(-np.mean(y * np.log(np.clip(probs, eps, 1 - eps))
                         + (1 - y) * np.log(np.clip(1 - probs, eps, 1 - eps))))
    brier = float(np.mean((y - probs) ** 2))
    auc = _auc(y, probs)
    return bce, brier, auc


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U → ROC AUC. No sklearn dep."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer encoder (defined at module top-level so it can be imported under
# torch presence). We protect the nn.Module subclass behind a lazy class
# factory so a torch-less import of `app.services.models.sequence` still
# succeeds (the meta-learner needs the SequenceModel class even when torch is
# absent and the layer is a stub).
# ──────────────────────────────────────────────────────────────────────────────


def _SequenceEncoder(*args, **kwargs):
    """Lazy constructor returning a fresh `nn.Module` instance.

    Defining the subclass inside this function means the file is importable on
    a torch-less install; only callers that reach the training/inference path
    pay for the import.
    """
    import torch
    import torch.nn as nn

    class _Impl(nn.Module):
        def __init__(self, cfg: SequenceModelConfig):
            super().__init__()
            self.cfg = cfg
            self.feature_proj = nn.Linear(cfg.n_seq_features, cfg.embedding_dim)
            self.today_proj = nn.Linear(cfg.n_today_features, cfg.embedding_dim)
            self.pos_emb = nn.Embedding(cfg.max_sequence_length + 1, cfg.embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.embedding_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.embedding_dim * 4,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # enable_nested_tensor + pre-LN is a known incompatibility that
            # produces a UserWarning and silently disables the nested-tensor
            # fast path. Explicitly disable so the warning is gone and the
            # behaviour is documented.
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.num_layers,
                enable_nested_tensor=False,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embedding_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            self.head = nn.Linear(cfg.embedding_dim * 2, 1)
            self.layer_norm = nn.LayerNorm(cfg.embedding_dim)

        def forward(
            self,
            seq: "torch.Tensor",
            pad_mask: "torch.Tensor",
            today: "torch.Tensor",
        ) -> "torch.Tensor":
            # seq: (B, T, F_seq), pad_mask: (B, T) True where pad,
            # today: (B, F_today).
            B, T, _ = seq.shape
            h = self.feature_proj(seq)  # (B, T, D)
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            h = torch.cat([cls, h], dim=1)  # (B, T+1, D)
            cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=seq.device)
            full_mask = torch.cat([cls_pad, pad_mask], dim=1)  # (B, T+1)
            # All-pad rows would zero-out attention entirely; nn.Transformer
            # propagates NaN in that case. Guard: at least the CLS isn't padded
            # by construction, but with all-pad keys the softmax over keys is
            # 0/0. Reset the mask so keys are unmasked when ALL keys are pad —
            # the CLS attends to the (zero) padding tokens, producing the
            # learned bias of CLS. We then overwrite the output of those rows
            # downstream via the "no priors → 0.5" guard in predict_proba.
            all_pad = full_mask.all(dim=1)
            if all_pad.any():
                full_mask = full_mask.clone()
                full_mask[all_pad] = False
            positions = torch.arange(T + 1, device=seq.device).unsqueeze(0).expand(B, -1)
            h = h + self.pos_emb(positions)
            h = self.encoder(h, src_key_padding_mask=full_mask)
            cls_out = self.layer_norm(h[:, 0])  # (B, D)
            today_emb = self.today_proj(today)  # (B, D)
            combined = torch.cat([cls_out, today_emb], dim=-1)  # (B, 2D)
            logit = self.head(combined).squeeze(-1)  # (B,)
            return logit

    return _Impl(*args, **kwargs)


__all__ = ["SequenceModelConfig", "SequenceModel"]
