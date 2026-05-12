"""Dedup key stability + collision-resistance tests.

DATA_PIPELINE.md §3 hashes are part of the DB contract. Changing the inputs
to the hash invalidates every existing dedup_key — these tests freeze the
exact hash inputs so a refactor cannot silently break the contract.
"""

from __future__ import annotations

from datetime import date

from scripts.db.dedup import (
    horse_dedup_key,
    normalize_name,
    person_dedup_key,
    race_dedup_key,
    result_dedup_key,
)


# ─── normalize_name ───────────────────────────────────────────────────────

def test_normalize_lowercases_strips_punct_collapses_whitespace():
    assert normalize_name("Lovely Words") == "lovely words"
    assert normalize_name("O'Brien") == "obrien"
    assert normalize_name("  John   A.  Ortiz  ") == "john a ortiz"
    assert normalize_name("hello-world!!") == "helloworld"


# ─── race_dedup_key ───────────────────────────────────────────────────────

def test_race_key_stable_across_date_and_string_input():
    """Calling with `date` vs ISO string must produce the same hash."""
    k1 = race_dedup_key("CD", "2026-05-10", 4, 4.5, "dirt")
    k2 = race_dedup_key("CD", date(2026, 5, 10), 4, 4.5, "dirt")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex length


def test_race_key_normalizes_track_case_and_surface_case():
    assert race_dedup_key("cd", "2026-05-10", 4, 4.5, "DIRT") == \
           race_dedup_key("CD", "2026-05-10", 4, 4.5, "dirt")


def test_race_key_distance_rounding_is_two_decimals():
    """4.50 == 4.5 == 4.500001 (rounded to 4.50) → same key."""
    assert race_dedup_key("CD", "2026-05-10", 4, 4.50, "dirt") == \
           race_dedup_key("CD", "2026-05-10", 4, 4.5,  "dirt")


def test_race_key_changes_with_each_field():
    base = race_dedup_key("CD", "2026-05-10", 4, 4.5, "dirt")
    assert base != race_dedup_key("SA", "2026-05-10", 4, 4.5, "dirt")  # track
    assert base != race_dedup_key("CD", "2026-05-11", 4, 4.5, "dirt")  # date
    assert base != race_dedup_key("CD", "2026-05-10", 5, 4.5, "dirt")  # number
    assert base != race_dedup_key("CD", "2026-05-10", 4, 6.0, "dirt")  # distance
    assert base != race_dedup_key("CD", "2026-05-10", 4, 4.5, "turf")  # surface


# ─── horse_dedup_key ──────────────────────────────────────────────────────

def test_horse_key_unknown_year_and_country_use_unknown_token():
    """Same name with unknown year+country gives a stable hash."""
    k1 = horse_dedup_key("Lovely Words", None, None)
    k2 = horse_dedup_key("Lovely Words", None, None)
    assert k1 == k2


def test_horse_key_distinguishes_different_horses_with_same_name():
    """Two horses named the same but foaled in different years are distinct."""
    k1 = horse_dedup_key("Northern Dancer", 2020, "US")
    k2 = horse_dedup_key("Northern Dancer", 2021, "US")
    assert k1 != k2


def test_horse_key_normalizes_punctuation_and_case():
    assert horse_dedup_key("LOVELY WORDS", 2022, "US") == \
           horse_dedup_key("lovely words", 2022, "US")


# ─── person_dedup_key ─────────────────────────────────────────────────────

def test_person_key_handles_unknown_jurisdiction():
    k = person_dedup_key("Mike Smith", None)
    assert len(k) == 64


def test_person_key_distinguishes_jurisdictions():
    """Same name in different countries = different ID (the design intent)."""
    assert person_dedup_key("John Smith", "US") != person_dedup_key("John Smith", "UK")


# ─── result_dedup_key ─────────────────────────────────────────────────────

def test_result_key_is_deterministic_combination():
    rk = race_dedup_key("CD", "2026-05-10", 4, 4.5, "dirt")
    hk = horse_dedup_key("Lovely Words", 2022, "US")
    k1 = result_dedup_key(rk, hk)
    k2 = result_dedup_key(rk, hk)
    assert k1 == k2
    # Swapping order changes the hash — guards against accidental commutativity.
    assert k1 != result_dedup_key(hk, rk)
