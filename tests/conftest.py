"""
tests/conftest.py
─────────────────
Process-wide test setup applied before any test module imports.

The only thing this file does is suppress the OpenMP duplicate-library abort
that happens when both `torch` (Sequence model) and `lightgbm` (Speed/Form,
Meta-learner) are loaded into the same Python interpreter.

Both libraries ship their own copy of `libomp.dylib` (macOS) / `libgomp.so`
(Linux); when a single process loads both, the second import fails with
``OMP: Error #15`` and the process can also segfault later during LightGBM
training (it dereferences a stale OpenMP thread-context). Setting
``KMP_DUPLICATE_LIB_OK=TRUE`` defers to whichever runtime loaded first and
is the documented Intel-recommended workaround. It is safe for our use
because torch + lightgbm never share the same OpenMP parallel region.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# On macOS/arm64 the segfault is reproducible when torch is imported BEFORE
# lightgbm initialises its internal thread pool. Force-single-thread both so
# they don't fight over OpenMP context.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "1")

# Phase 9 / ADR-049 — placeholder API key so the LLM parser module imports
# cleanly under test. Real API calls in tests are mocked; this avoids the
# "ANTHROPIC_API_KEY not set" early-return path from masking real failures
# when tests want to exercise the API call shape. Set BEFORE pytest collects
# any module that reads the env at import time.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
