"""Stage 2 — automatic data profiling.

Takes a cleaned DataFrame and produces a `DatasetProfile`: a structured,
fact-only description of the data. Every number here is computed by pandas.

The profile is used in two places:
  * the UI (Overview and Column details tabs), and
  * later (Stage 4) as the *only* information about the data that we send to
    the AI — which is why `to_dict()` returns plain, JSON-safe values.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- data shapes

# A column's "role" describes how it should be analyzed, not just its dtype.
#   measure    – numbers you can sum/average (Revenue, Units)
#   identifier – unique labels you shouldn't do math on (Order ID, Zip code)
#   category   – a small set of repeating labels to group by (Region, Product)
#   date       – dates/times
#   boolean    – yes/no
#   text       – free text with many different values (comments, names)
ROLES = ("measure", "identifier", "category", "date", "boolean", "text")


@dataclass
class ColumnProfile:
    name: str
    role: str
    dtype: str
    count: int               # non-missing values
    missing: int
    missing_pct: float
    unique: int
    stats: dict[str, Any] = field(default_factory=dict)  # role-specific statistics


@dataclass
class QualityIssue:
    severity: str            # "warning" (worth fixing) or "info" (worth knowing)
    message: str
    column: str | None = None


@dataclass
class Correlation:
    column_a: str
    column_b: str
    r: float                 # Pearson correlation, -1..1
    strength: str            # "strong" or "moderate"
    n: int                   # number of rows used


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    total_cells: int
    missing_cells: int
    completeness_pct: float
    duplicate_rows: int
    columns: list[ColumnProfile]
    issues: list[QualityIssue]
    correlations: list[Correlation]

    def by_role(self, role: str) -> list[ColumnProfile]:
        return [c for c in self.columns if c.role == role]

    def column(self, name: str) -> ColumnProfile:
        return next(c for c in self.columns if c.name == name)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of built-in types (safe for JSON / sending to an API)."""
        return asdict(self)


# ---------------------------------------------------------------- helpers

_ID_NAME = re.compile(r"(^|[\s_\-])(id|code|zip|postal|phone|sku|uuid)$|id$", re.IGNORECASE)
_ADDITIVE_NAME = re.compile(
    r"revenue|sales|amount|profit|cost|units|quantity|qty|total|income|expense|spend|orders|volume",
    re.IGNORECASE,
)
_NUMBER_JUNK = re.compile(r"[\s,$€£¥%()]")


def _num(x: Any, digits: int = 4) -> float | None:
    """numpy/pandas number -> plain Python float (or None for NaN)."""
    if x is None or pd.isna(x):
        return None
    return round(float(x), digits)


def _pct(part: int, whole: int) -> float:
    return round(part / whole * 100, 2) if whole else 0.0


def classify_column(name: str, s: pd.Series) -> str:
    """Decide the analytical role of a column using its type, values and name."""
    non_null = s.dropna()
    n, unique = len(non_null), non_null.nunique()

    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        if _ID_NAME.search(name.strip()):
            return "identifier"
        # Every value a different whole number (e.g. 1001, 1002, …) looks like an ID
        is_whole = n > 0 and bool((non_null % 1 == 0).all())
        if is_whole and n >= 20 and unique == n and not _ADDITIVE_NAME.search(name):
            return "identifier"
        return "measure"
    # Text columns
    if n == 0:
        return "text"
    if n >= 20 and unique / n >= 0.95:
        return "identifier"
    # Few labels, or labels that repeat a lot (≤ 1 new label per 5 rows)
    if unique <= 20 or unique / n <= 0.2:
        return "category"
    return "text"


def _iqr_outliers(s: pd.Series) -> dict[str, Any]:
    """Count values far outside the middle 50% (the standard 1.5×IQR rule)."""
    if len(s) < 10:
        return {"outliers": 0}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return {"outliers": 0}
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_low, n_high = int((s < low).sum()), int((s > high).sum())
    return {
        "outliers": n_low + n_high,
        "outliers_low": n_low,
        "outliers_high": n_high,
        "outlier_low_bound": _num(low, 2),
        "outlier_high_bound": _num(high, 2),
    }


def _measure_stats(s: pd.Series) -> dict[str, Any]:
    s = s.dropna().astype(float)
    if s.empty:
        return {}
    return {
        "sum": _num(s.sum()),
        "mean": _num(s.mean()),
        "median": _num(s.median()),
        "std": _num(s.std()) if len(s) > 1 else None,
        "min": _num(s.min()),
        "q1": _num(s.quantile(0.25)),
        "q3": _num(s.quantile(0.75)),
        "max": _num(s.max()),
        "negatives": int((s < 0).sum()),
        "zeros": int((s == 0).sum()),
        **_iqr_outliers(s),
    }


def _category_stats(s: pd.Series, top_n: int = 10) -> dict[str, Any]:
    counts = s.dropna().astype(str).value_counts()
    if counts.empty:
        return {}
    total = int(counts.sum())
    return {
        "top_value": str(counts.index[0]),
        "top_count": int(counts.iloc[0]),
        "top_pct": _pct(int(counts.iloc[0]), total),
        "top_values": [
            {"value": str(v), "count": int(c), "pct": _pct(int(c), total)}
            for v, c in counts.head(top_n).items()
        ],
    }


def _date_stats(s: pd.Series) -> dict[str, Any]:
    s = s.dropna()
    if s.empty:
        return {}
    has_time = not bool((s.dt.normalize() == s).all())
    fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
    return {
        "min": s.min().strftime(fmt),
        "max": s.max().strftime(fmt),
        "span_days": int((s.max() - s.min()).days),
        "distinct_dates": int(s.dt.normalize().nunique()),
    }


def _text_stats(s: pd.Series) -> dict[str, Any]:
    lengths = s.dropna().astype(str).str.len()
    if lengths.empty:
        return {}
    return {"avg_length": _num(lengths.mean(), 1), "examples": s.dropna().astype(str).head(3).tolist()}


def _numeric_share(s: pd.Series) -> float:
    """Share of a text column's values that would be numbers once '$', ',' etc. are removed."""
    text = s.dropna().astype(str).str.replace(_NUMBER_JUNK, "", regex=True)
    if text.empty:
        return 0.0
    return float(pd.to_numeric(text, errors="coerce").notna().mean())


# ---------------------------------------------------------------- quality checks

def _quality_issues(df: pd.DataFrame, columns: list[ColumnProfile], duplicate_rows: int) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    n_rows = len(df)

    if duplicate_rows:
        issues.append(QualityIssue(
            "warning",
            f"{duplicate_rows:,} row(s) are exact duplicates of another row "
            f"({_pct(duplicate_rows, n_rows)}% of rows). They may be double-counted in totals.",
        ))

    for c in columns:
        if c.missing_pct >= 50:
            issues.append(QualityIssue("warning", f"'{c.name}' is mostly empty ({c.missing_pct}% missing).", c.name))
        elif c.missing:
            issues.append(QualityIssue("info", f"'{c.name}' has {c.missing:,} missing value(s) ({c.missing_pct}%).", c.name))

        if c.unique == 1 and n_rows > 1:
            only = str(df[c.name].dropna().iloc[0])
            issues.append(QualityIssue(
                "info", f"'{c.name}' contains only one distinct value ('{only}'), so it can't explain differences.", c.name))

        if c.role == "measure" and c.stats.get("outliers"):
            parts = []
            if c.stats["outliers_high"]:
                parts.append(f"{c.stats['outliers_high']:,} unusually high (above {c.stats['outlier_high_bound']:,.2f})")
            if c.stats["outliers_low"]:
                parts.append(f"{c.stats['outliers_low']:,} unusually low (below {c.stats['outlier_low_bound']:,.2f})")
            issues.append(QualityIssue(
                "info",
                f"'{c.name}' has {' and '.join(parts)} value(s) compared with the rest of the column. "
                "They may be genuine or data-entry errors — worth a look.",
                c.name,
            ))

        if c.role in ("text", "category", "identifier") and c.dtype != "datetime":
            share = _numeric_share(df[c.name])
            if 0.5 <= share < 0.95:
                bad = df[c.name].dropna().astype(str)
                bad = bad[pd.to_numeric(bad.str.replace(_NUMBER_JUNK, "", regex=True), errors="coerce").isna()]
                examples = ", ".join(f"'{v}'" for v in bad.unique()[:3])
                issues.append(QualityIssue(
                    "warning",
                    f"'{c.name}' is mostly numbers ({share:.0%}) but also contains text such as {examples}, "
                    "so it's being treated as text and can't be summed or averaged.",
                    c.name,
                ))

    # Warnings first, then info
    issues.sort(key=lambda i: i.severity != "warning")
    return issues


def _correlations(df: pd.DataFrame, measures: list[str], threshold: float = 0.5, top_n: int = 5) -> list[Correlation]:
    if len(measures) < 2:
        return []
    found: list[Correlation] = []
    for i, a in enumerate(measures):
        for b in measures[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < 10 or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                continue
            r = float(pair[a].corr(pair[b]))
            if np.isfinite(r) and abs(r) >= threshold:
                found.append(Correlation(a, b, round(r, 3), "strong" if abs(r) >= 0.7 else "moderate", len(pair)))
    found.sort(key=lambda c: abs(c.r), reverse=True)
    return found[:top_n]


# ---------------------------------------------------------------- public API

def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Build a complete, fact-only profile of a DataFrame."""
    n_rows, n_cols = df.shape
    columns: list[ColumnProfile] = []

    for name in df.columns:
        s = df[name]
        role = classify_column(str(name), s)
        missing = int(s.isna().sum())
        stats_fn = {
            "measure": _measure_stats,
            "date": _date_stats,
            "category": _category_stats,
            "boolean": _category_stats,
            "text": _text_stats,
            "identifier": lambda x: {"examples": x.dropna().astype(str).head(3).tolist()},
        }[role]
        if pd.api.types.is_datetime64_any_dtype(s):
            dtype = "datetime"
        elif pd.api.types.is_bool_dtype(s):
            dtype = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            dtype = "number"
        else:
            dtype = "text"
        columns.append(ColumnProfile(
            name=str(name), role=role, dtype=dtype,
            count=n_rows - missing, missing=missing, missing_pct=_pct(missing, n_rows),
            unique=int(s.nunique(dropna=True)), stats=stats_fn(s),
        ))

    total_cells = int(df.size)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    measures = [c.name for c in columns if c.role == "measure"]

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=n_cols,
        total_cells=total_cells,
        missing_cells=missing_cells,
        completeness_pct=round(100 - _pct(missing_cells, total_cells), 2) if total_cells else 0.0,
        duplicate_rows=duplicate_rows,
        columns=columns,
        issues=_quality_issues(df, columns, duplicate_rows),
        correlations=_correlations(df, measures),
    )


def is_additive(column_name: str) -> bool:
    """True if a column's *total* is usually meaningful (Revenue, Units — not Price, Rating)."""
    return bool(_ADDITIVE_NAME.search(column_name))


def suggest_kpi_columns(profile: DatasetProfile, limit: int = 4) -> list[str]:
    """Pick measures whose *totals* are meaningful (Revenue, Units — not Price or %).

    Falls back to the first few measures if no column name looks additive.
    The user can always change the selection in the UI.
    """
    measures = [c.name for c in profile.by_role("measure")]
    additive = [m for m in measures if _ADDITIVE_NAME.search(m)]
    # Most business-relevant first; ties keep the file's column order (sorted() is stable).
    additive = sorted(additive, key=_kpi_priority)
    return (additive or measures)[:limit]


# Lower number = shown first. Revenue matters more to a business reader than units shipped.
_KPI_TIERS = [
    re.compile(r"revenue|sales|income", re.IGNORECASE),
    re.compile(r"profit|margin", re.IGNORECASE),
    re.compile(r"amount|total|spend", re.IGNORECASE),
    re.compile(r"cost|expense", re.IGNORECASE),
]


def _kpi_priority(name: str) -> int:
    return next((i for i, tier in enumerate(_KPI_TIERS) if tier.search(name)), len(_KPI_TIERS))
