"""Stage 1 — reading, validating and lightly cleaning uploaded files.

Everything happens in memory: we receive the file's raw bytes and never
write them to disk. Every change we make to the data is recorded in a
`cleaning_log` so the user can see exactly what happened.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core import config


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file can't be loaded."""


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    file_name: str
    sheet_name: str | None = None
    cleaning_log: list[str] = field(default_factory=list)
    # Currency symbol found in the data (e.g. "$" in "$1,200"), used to format money columns.
    # None when the file shows no symbol — the app then never assumes a currency.
    currency_symbol: str | None = None


# ---------------------------------------------------------------- validation

def _validate_file(file_name: str, data: bytes) -> str:
    ext = Path(file_name).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise DataLoadError(f"Unsupported file type '{ext or 'none'}'. Please upload one of: {allowed}.")
    if len(data) == 0:
        raise DataLoadError("The file is empty.")
    if len(data) > config.MAX_FILE_BYTES:
        raise DataLoadError(f"File is larger than the {config.MAX_FILE_MB} MB limit.")
    return ext


# ---------------------------------------------------------------- reading

def list_excel_sheets(data: bytes) -> list[str]:
    """Return the sheet names of an .xlsx file."""
    try:
        with pd.ExcelFile(io.BytesIO(data), engine="openpyxl") as xls:
            return [str(s) for s in xls.sheet_names]
    except Exception as exc:  # corrupted / not really an xlsx
        raise DataLoadError("Couldn't open this Excel file. Is it a valid .xlsx?") from exc


def _decode_text(data: bytes) -> tuple[str, str]:
    """Try common encodings. latin-1 accepts any bytes, so it's the last resort."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DataLoadError("Couldn't read the text encoding of this CSV file.")  # pragma: no cover


def _detect_delimiter(text: str) -> str:
    """Guess whether the CSV uses , ; tab or | between values."""
    sample = text[:64_000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_csv(data: bytes, log: list[str]) -> pd.DataFrame:
    text, encoding = _decode_text(data)
    if encoding not in ("utf-8", "utf-8-sig"):
        log.append(f"File was read using the '{encoding}' text encoding.")
    delimiter = _detect_delimiter(text)
    if delimiter != ",":
        names = {";": "semicolon", "\t": "tab", "|": "pipe"}
        log.append(f"Detected {names.get(delimiter, repr(delimiter))}-separated values.")
    try:
        return pd.read_csv(io.StringIO(text), sep=delimiter, low_memory=False)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The CSV file has no data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"The CSV file looks malformed: {exc}") from exc


def _read_excel(data: bytes, sheet_name: str | None) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name or 0, engine="openpyxl")
    except ValueError as exc:
        raise DataLoadError(f"Couldn't read sheet '{sheet_name}'.") from exc
    except Exception as exc:
        raise DataLoadError("Couldn't open this Excel file. Is it a valid .xlsx?") from exc


# ---------------------------------------------------------------- cleaning

def _clean_column_names(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    new_names: list[str] = []
    seen: dict[str, int] = {}
    renamed = 0
    for i, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        if name != str(col):
            renamed += 1
        new_names.append(name)
    if renamed:
        log.append(f"Tidied {renamed} column name(s) (blank, duplicate or extra spaces).")
    df.columns = new_names
    return df


def _drop_empty(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    rows_before, cols_before = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    dropped_rows = rows_before - df.shape[0]
    dropped_cols = cols_before - df.shape[1]
    if dropped_rows:
        log.append(f"Removed {dropped_rows:,} completely empty row(s).")
    if dropped_cols:
        log.append(f"Removed {dropped_cols:,} completely empty column(s).")
    return df.reset_index(drop=True)


_NUMBER_JUNK = re.compile(r"[\s,$€£¥%]")
_DATE_HINT = re.compile(
    r"\d{1,4}[-/.]\d{1,2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)


def _try_numeric(s: pd.Series) -> pd.Series | None:
    """'$1,200', '15%', '(300)' -> 1200, 15, -300. Returns None if it doesn't fit."""
    text = s.astype(str).str.strip()
    negative = text.str.match(r"^\(.*\)$")
    cleaned = text.str.replace(r"[()]", "", regex=True).str.replace(_NUMBER_JUNK, "", regex=True)
    numbers = pd.to_numeric(cleaned, errors="coerce")
    if numbers.notna().mean() < config.CONVERSION_THRESHOLD:
        return None
    return numbers.where(~negative, -numbers)


def _try_datetime(s: pd.Series) -> pd.Series | None:
    text = s.astype(str).str.strip()
    if text.str.contains(_DATE_HINT).mean() < config.CONVERSION_THRESHOLD:
        return None
    dates = pd.to_datetime(text, errors="coerce", format="mixed")
    if dates.notna().mean() < config.CONVERSION_THRESHOLD:
        return None
    return dates


_CURRENCY = re.compile(r"[$€£¥]")


def _infer_types(df: pd.DataFrame, log: list[str], symbols: dict[str, int]) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]
        is_text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        if not is_text:
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue

        numbers = _try_numeric(non_null)
        if numbers is not None:
            df[col] = numbers.reindex(df.index)
            note = " (values were percentages, e.g. 15% → 15)" if non_null.astype(str).str.contains("%").any() else ""
            log.append(f"Converted '{col}' from text to numbers{note}.")
            for sym in non_null.astype(str).str.findall(_CURRENCY).explode().dropna():
                symbols[sym] = symbols.get(sym, 0) + 1
            continue

        dates = _try_datetime(non_null)
        if dates is not None:
            df[col] = dates.reindex(df.index)
            log.append(f"Converted '{col}' from text to dates.")
            continue

        # Plain text: trim stray spaces so "North " and "North" are the same group
        trimmed = s.where(s.isna(), s.astype(str).str.strip())
        if not trimmed.equals(s):
            df[col] = trimmed
            log.append(f"Trimmed extra spaces in '{col}'.")
    return df


# ---------------------------------------------------------------- public API

def load_dataset(file_name: str, data: bytes, sheet_name: str | None = None) -> LoadedDataset:
    """Validate, read and clean an uploaded file. Raises DataLoadError on problems."""
    ext = _validate_file(file_name, data)
    log: list[str] = []

    df = _read_csv(data, log) if ext == ".csv" else _read_excel(data, sheet_name)

    if df.shape[1] > config.MAX_COLUMNS:
        raise DataLoadError(f"This file has {df.shape[1]:,} columns; the limit is {config.MAX_COLUMNS}.")
    if df.shape[0] > config.MAX_ROWS:
        raise DataLoadError(f"This file has {df.shape[0]:,} rows; the limit is {config.MAX_ROWS:,}.")

    df = _clean_column_names(df, log)
    df = _drop_empty(df, log)
    if df.empty:
        raise DataLoadError("No data found after removing empty rows and columns.")
    symbols: dict[str, int] = {}
    df = _infer_types(df, log, symbols)
    currency = max(symbols, key=symbols.get) if symbols else None
    if currency:
        log.append(f"Found the currency symbol '{currency}' in the data; money columns are shown in {currency}.")

    return LoadedDataset(df=df, file_name=file_name, sheet_name=sheet_name, cleaning_log=log,
                         currency_symbol=currency)
