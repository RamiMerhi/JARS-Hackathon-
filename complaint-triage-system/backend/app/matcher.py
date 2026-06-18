"""
Stable establishment matcher.

The previous version keyed live establishments by normalized name, which can
collapse different establishments into the same key. This version always uses
`establishment_id` as the stable identity and only uses normalized names for
search/matching.
"""
import re
import pandas as pd
from rapidfuzz import process, fuzz

_STOPWORDS = {
    "the", "and", "co", "company", "ltd", "llc", "sarl", "est", "establishment",
    "restaurant", "rest", "market", "supermarket", "mini", "super", "store",
    "shop", "butchery", "butcher", "bakery", "cafe", "coffee", "grocery",
    "pharmacy", "of", "for", "al", "el",
}

MATCH_THRESHOLD = 80


def normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower()
    # Keep Arabic/French letters instead of deleting everything outside a-z.
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in _STOPWORDS]
    return " ".join(tokens) if tokens else s


class EstablishmentMatcher:
    def __init__(self, establishments: pd.DataFrame):
        self.df = establishments.reset_index(drop=True).copy()
        self.df["_norm"] = self.df["name"].apply(normalize)
        self._choices = {i: n for i, n in enumerate(self.df["_norm"]) if n}

    def match(self, purchase_place: str, province: str | None = None):
        query = normalize(purchase_place)
        if not query:
            return None

        exact = self.df[self.df["_norm"] == query]
        if len(exact):
            # Prefer same province if multiple names normalize identically.
            if province:
                same = exact[exact["province"].astype(str).str.lower() == str(province).lower()]
                if len(same):
                    return self._row_to_dict(same.iloc[0])
            return self._row_to_dict(exact.iloc[0])

        results = process.extract(query, self._choices, scorer=fuzz.token_set_ratio, limit=8)
        if not results:
            return None

        best = None
        for _matched_string, score, idx in results:
            if score < MATCH_THRESHOLD:
                continue
            row = self.df.iloc[idx]
            same_province = bool(
                province and str(row.get("province", "")).strip().lower() == str(province).strip().lower()
            )
            adj = score + (5 if same_province else 0)
            # Tie-breaker: if scores are equal, prefer more violations/open complaints
            # because those are more relevant for triage safety.
            risk_tie = int(row.get("violations", 0) or 0) + int(row.get("open_complaints", 0) or 0)
            candidate = (adj, risk_tie, row)
            if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate

        return self._row_to_dict(best[2]) if best else None

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "establishment_id": row.get("establishment_id", "") or None,
            "name": row["name"],
            "zone": row.get("zone", "UNKNOWN") or "UNKNOWN",
            "province": row.get("province", "") or None,
            "violations": int(row.get("violations", 0) or 0),
            "open_complaints": int(row.get("open_complaints", 0) or 0),
        }
