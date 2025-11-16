# src/tokenizers.py
def split_commas(s: str):
    """Tokenize a comma-separated string into trimmed tokens."""
    if s is None:
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]
