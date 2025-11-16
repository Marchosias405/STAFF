# src/rules.py
# -----------------------------------------------------------------------------
# Simple rule engine for shipping compliance "flags".
# - Loads keyword rules from CSVs in data/
# - Matches case-insensitive terms against your structured inputs:
#     * tags: comma-separated keywords (primary signal)
#     * description: optional fallback for extra matches (not required)
# - Returns a list of human-readable flags to display with predictions.
#
# CSV formats:
#   data/rules_restricted.csv  -> columns: term,category,notes
#   data/rules_gift.csv        -> columns: term,category,notes
#
# Example usage:
#   flags = apply_rules(tags="electronics,headphones,wireless,lithium",
#                       gift=1, hs_pred=8518, description="sony xm4 ...")
#   # -> ["Restricted: Dangerous Goods (lithium) – Special packing/labels for air",
#         "Gift: Exemption (gift) – Check threshold/personal use criteria"]
# -----------------------------------------------------------------------------

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set

# ------------------------------- configuration ------------------------------ #

DATA_DIR = Path("data")
RESTRICTED_CSV = DATA_DIR / "rules_restricted.csv"
GIFT_CSV = DATA_DIR / "rules_gift.csv"

# ------------------------------- data classes -------------------------------- #

@dataclass(frozen=True)
class Rule:
    term: str           # keyword to match (lowercased)
    category: str       # e.g., "Dangerous Goods", "Prohibited", "Exemption"
    notes: str          # short guidance to show user

@dataclass(frozen=True)
class RuleFlag:
    kind: str           # "Restricted" or "Gift"
    category: str       # copied from rule.category
    term: str           # the matched keyword
    notes: str          # copied from rule.notes

    def pretty(self) -> str:
        """Human-friendly single-line message for CLI/UI."""
        prefix = f"{self.kind}: {self.category} ({self.term})"
        return f"{prefix} – {self.notes}" if self.notes else prefix

# ------------------------------- load & cache -------------------------------- #

class _RuleStore:
    """Loads and caches rules from CSVs (singleton-ish)."""
    def __init__(self):
        self._restricted: List[Rule] = []
        self._gift: List[Rule] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self._restricted = _load_rules_csv(RESTRICTED_CSV)
        self._gift = _load_rules_csv(GIFT_CSV)
        self._loaded = True

    @property
    def restricted(self) -> List[Rule]:
        self.load()
        return self._restricted

    @property
    def gift(self) -> List[Rule]:
        self.load()
        return self._gift

_RULES = _RuleStore()

def _load_rules_csv(path: Path) -> List[Rule]:
    """Read a rules CSV with columns: term,category,notes."""
    rules: List[Rule] = []
    if not path.exists():
        return rules
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = (row.get("term") or "").strip().lower()
            if not term:
                continue
            category = (row.get("category") or "").strip()
            notes = (row.get("notes") or "").strip()
            rules.append(Rule(term=term, category=category, notes=notes))
    return rules

# ------------------------------ matching utils ------------------------------ #

def _tokenize_tags(tags: Optional[str]) -> Set[str]:
    """
    Turn a comma-separated tag string into a set of lowercase tokens.
    Example: "electronics,audio,Wireless" -> {"electronics","audio","wireless"}
    """
    if not tags:
        return set()
    return {t.strip().lower() for t in tags.split(",") if t.strip()}

def _contains_term_in_text(term: str, text: Optional[str]) -> bool:
    """
    Fallback check in description. Case-insensitive substring match.
    Keep it simple (no NLP): "lithium" matches "lithium-ion".
    """
    if not text:
        return False
    return term in text.lower()

def _match_terms(terms: Set[str], term: str, description: Optional[str]) -> bool:
    """
    Primary: exact term match in tags.
    Fallback: substring match in description (optional).
    """
    if term in terms:
        return True
    return _contains_term_in_text(term, description)

# ------------------------------- public API --------------------------------- #

def apply_rules(
    tags: str,
    gift: int | bool,
    hs_pred: Optional[int | str] = None,
    description: Optional[str] = None,
) -> List[str]:
    """
    Apply both restricted and gift rules to a single item.

    Parameters
    ----------
    tags : str
        Comma-separated keywords from your structured input.
    gift : int|bool
        1/True if user marked as gift; 0/False otherwise.
    hs_pred : Optional[int|str]
        Predicted HS code (not used for matching right now, but useful if you
        later add HS-range specific rules).
    description : Optional[str]
        Optional free-text. We *don't* rely on this, but use it as a fallback.

    Returns
    -------
    List[str]
        A list of human-readable flag messages to show in CLI/UI.
    """
    termset = _tokenize_tags(tags)

    flags: List[RuleFlag] = []

    # 1) Restricted / Prohibited / Special Handling
    for r in _RULES.restricted:
        if _match_terms(termset, r.term, description):
            flags.append(RuleFlag(kind="Restricted", category=r.category, term=r.term, notes=r.notes))

    # 2) Gift indicators (only if the input gift flag is set OR term detected)
    gift_flag = bool(gift)
    if gift_flag:
        for r in _RULES.gift:
            # If the item is already marked gift, add generic gift rule.
            # We still try to associate a concrete term; if none present, use the first row term.
            matched_term = r.term if _match_terms(termset, r.term, description) else r.term
            flags.append(RuleFlag(kind="Gift", category=r.category, term=matched_term, notes=r.notes))
    else:
        # If not marked gift, still add a hint if terms appear in tags/description.
        for r in _RULES.gift:
            if _match_terms(termset, r.term, description):
                flags.append(RuleFlag(kind="Gift", category=r.category, term=r.term, notes=r.notes))

    # De-duplicate (same kind+category+term)
    dedup: Dict[Tuple[str, str, str], RuleFlag] = {}
    for f in flags:
        key = (f.kind, f.category, f.term)
        dedup[key] = f
    return [f.pretty() for f in dedup.values()]

# ------------------------------- batch helper -------------------------------- #

def apply_rules_df(
    tags_col: List[str],
    gift_col: List[int],
    hs_col: Optional[List[int]] = None,
    desc_col: Optional[List[str]] = None,
) -> List[List[str]]:
    """
    Vectorized-ish application over lists/Series. Returns a list of flag lists.
    Example:
        flags_list = apply_rules_df(df['tags'], df['gift'])
        # flags_list[i] -> list[str] for row i
    """
    n = len(tags_col)
    out: List[List[str]] = []
    for i in range(n):
        hs_val = hs_col[i] if hs_col is not None else None
        desc_val = desc_col[i] if desc_col is not None else None
        out.append(apply_rules(tags=str(tags_col[i]), gift=int(gift_col[i]), hs_pred=hs_val, description=desc_val))
    return out

# -------------------------------- self test ---------------------------------- #

if __name__ == "__main__":
    # Quick smoke test (will work if your CSVs exist; otherwise prints empty flags)
    sample_tags = "electronics,headphones,wireless,lithium"
    sample_desc = "Sony XM4 headphones with lithium-ion battery"
    msgs = apply_rules(tags=sample_tags, gift=1, hs_pred=8518, description=sample_desc)
    print("Flags:")
    for m in msgs:
        print("-", m)
