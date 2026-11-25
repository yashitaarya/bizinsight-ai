"""Unit tests for core/data_loader.py.  Run with:  pytest"""

import io

import pandas as pd
import pytest

from core.data_loader import DataLoadError, list_excel_sheets, load_dataset


def csv_bytes(text: str, encoding: str = "utf-8") -> bytes:
    return text.encode(encoding)


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------- validation

def test_rejects_unsupported_extension():
    with pytest.raises(DataLoadError, match="Unsupported file type"):
        load_dataset("notes.txt", b"hello")


def test_rejects_empty_file():
    with pytest.raises(DataLoadError, match="empty"):
        load_dataset("data.csv", b"")


def test_rejects_too_large(monkeypatch):
    monkeypatch.setattr("core.config.MAX_FILE_BYTES", 10)
    with pytest.raises(DataLoadError, match="limit"):
        load_dataset("data.csv", b"a,b\n1,2\n3,4\n")


def test_rejects_corrupt_excel():
    with pytest.raises(DataLoadError):
        load_dataset("broken.xlsx", b"this is not a zip file")


# ---------------------------------------------------------------- reading

def test_basic_csv():
    ds = load_dataset("s.csv", csv_bytes("Region,Sales\nNorth,10\nSouth,20\n"))
    assert ds.df.shape == (2, 2)
    assert ds.df["Sales"].sum() == 30


def test_semicolon_csv_and_latin1():
    text = "Ville;Ventes\nMontréal;5\nQuébec;7\n"
    ds = load_dataset("fr.csv", csv_bytes(text, "latin-1"))
    assert list(ds.df.columns) == ["Ville", "Ventes"]
    assert ds.df["Ville"].tolist() == ["Montréal", "Québec"]
    assert any("semicolon" in m for m in ds.cleaning_log)


def test_excel_multiple_sheets():
    data = xlsx_bytes({
        "Q1": pd.DataFrame({"A": [1, 2]}),
        "Q2": pd.DataFrame({"B": [3, 4, 5]}),
    })
    assert list_excel_sheets(data) == ["Q1", "Q2"]
    ds = load_dataset("book.xlsx", data, sheet_name="Q2")
    assert ds.df["B"].tolist() == [3, 4, 5]


# ---------------------------------------------------------------- cleaning

def test_currency_percent_and_negative_parentheses():
    text = 'Amount,Rate\n"$1,200.50",15%\n(300),5%\n"$45",0%\n'
    ds = load_dataset("m.csv", csv_bytes(text))
    assert ds.df["Amount"].tolist() == [1200.50, -300.0, 45.0]
    assert ds.df["Rate"].tolist() == [15, 5, 0]
    assert any("percentages" in m for m in ds.cleaning_log)


def test_dates_converted():
    ds = load_dataset("d.csv", csv_bytes("When,X\n2025-01-05,1\n2025-02-10,2\n"))
    assert pd.api.types.is_datetime64_any_dtype(ds.df["When"])


def test_text_ids_are_not_turned_into_dates_or_numbers():
    ds = load_dataset("i.csv", csv_bytes("Order ID,Name\nORD-1,Ann\nORD-2,Bob\n"))
    assert not pd.api.types.is_numeric_dtype(ds.df["Order ID"])
    assert not pd.api.types.is_datetime64_any_dtype(ds.df["Order ID"])


def test_mostly_text_column_left_alone():
    # Only 2 of 4 values are numbers -> below threshold, keep as text
    ds = load_dataset("t.csv", csv_bytes("Code\n100\n200\nabc\nxyz\n"))
    assert not pd.api.types.is_numeric_dtype(ds.df["Code"])


def test_empty_rows_columns_and_bad_headers():
    text = "Name,,Name,Score\nAnn,,x,1\n,,,\nBob,,y,2\n"
    ds = load_dataset("e.csv", csv_bytes(text))
    assert ds.df.shape == (2, 3)
    assert len(set(ds.df.columns)) == 3          # no duplicate names
    assert any("empty row" in m for m in ds.cleaning_log)


def test_whitespace_trimmed():
    ds = load_dataset("w.csv", csv_bytes("Region,V\n North ,1\nNorth,2\n"))
    assert ds.df["Region"].nunique() == 1


def test_sample_data_loads():
    from pathlib import Path
    sample = Path(__file__).parent.parent / "sample_data" / "sales_sample.csv"
    ds = load_dataset(sample.name, sample.read_bytes())
    assert len(ds.df) == 1200
    assert pd.api.types.is_numeric_dtype(ds.df["Unit Price"])
    assert pd.api.types.is_datetime64_any_dtype(ds.df["Order Date"])
