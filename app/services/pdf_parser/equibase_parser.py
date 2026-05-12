"""
app/services/pdf_parser/equibase_parser.py
──────────────────────────────────────────
Parser for Equibase past performance PDFs.

Equibase PDFs share most structural conventions with Brisnet UP — same
column-oriented PP tables, same race header layout, similar typography —
so the v1 implementation subclasses `BrisnetParser` and inherits all its
regexes. Genuine divergences (different speed-figure column header, slight
fraction-time formatting differences, workouts table position) will be
addressed as method overrides here once the first real-PDF validation run
surfaces them.

Until then, this thin subclass exists so the dispatcher in `extractor.py`
can route on format without falling back through an "unknown" path.
"""

from __future__ import annotations

from app.services.pdf_parser.brisnet_parser import BrisnetParser


class EquibaseParser(BrisnetParser):
    """Equibase PP parser — currently identical to Brisnet UP at the regex layer."""


__all__ = ["EquibaseParser"]
