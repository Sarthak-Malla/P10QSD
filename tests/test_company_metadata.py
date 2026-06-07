import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.update_sp500_metadata import normalize_constituents
from src.dataloader.company_metadata import (
    get_sector_etf,
    get_ticker_metadata,
    get_ticker_sector,
    load_company_metadata,
    normalize_ticker,
)
from src.dataloader.filing_dataset import resolve_section_columns


def _metadata_csv(tmp_path):
    path = tmp_path / "sp500_constituents.csv"
    pd.DataFrame({
        "ticker": ["AAPL", "JPM", "XOM", "GOOGL", "BRK-B"],
        "company_name": ["Apple", "JPMorgan", "Exxon", "Alphabet", "Berkshire"],
        "cik": ["0000320193", "0000019617", "0000034088", "0001652044", "0001067983"],
        "gics_sector": [
            "Information Technology",
            "Financials",
            "Energy",
            "Communication Services",
            "Financials",
        ],
        "gics_sub_industry": ["Technology Hardware", "Banks", "Oil", "Interactive Media", "Multi-Sector"],
        "sector_etf": ["XLK", "XLF", "XLE", "XLC", "XLF"],
        "source": ["test"] * 5,
        "source_date": ["2026-06-07"] * 5,
    }).to_csv(path, index=False)
    load_company_metadata.cache_clear()
    return str(path)


def test_ticker_normalization_handles_share_class_dots():
    assert normalize_ticker("brk.b") == "BRK-B"
    assert normalize_ticker(" BF.B ") == "BF-B"


def test_metadata_lookup_maps_gics_sector_to_sector_etf(tmp_path):
    metadata_path = _metadata_csv(tmp_path)

    assert get_ticker_sector("AAPL", metadata_path) == "Information Technology"
    assert get_sector_etf("AAPL", metadata_path) == "XLK"
    assert get_sector_etf("JPM", metadata_path) == "XLF"
    assert get_sector_etf("XOM", metadata_path) == "XLE"
    assert get_sector_etf("GOOGL", metadata_path) == "XLC"
    assert get_sector_etf("BRK.B", metadata_path) == "XLF"


def test_missing_ticker_falls_back_to_unknown_and_spy(tmp_path):
    metadata_path = _metadata_csv(tmp_path)

    metadata = get_ticker_metadata("NOPE", metadata_path)

    assert metadata["gics_sector"] == "Unknown"
    assert metadata["sector_etf"] == "SPY"


def test_update_script_normalizes_source_columns():
    source = pd.DataFrame({
        "Symbol": ["BRK.B", "AAPL"],
        "Security": ["Berkshire Hathaway", "Apple Inc."],
        "GICS Sector": ["Financials", "Information Technology"],
        "GICS Sub-Industry": ["Multi-Sector Holdings", "Technology Hardware"],
        "CIK": [1067983, 320193],
    })

    metadata = normalize_constituents(source, "test-source")

    assert metadata.loc[metadata["ticker"] == "BRK-B", "sector_etf"].iloc[0] == "XLF"
    assert metadata.loc[metadata["ticker"] == "AAPL", "cik"].iloc[0] == "0000320193"


def test_filing_dataset_uses_explicit_mda_section_not_second_section():
    sec_cfg = {
        "sections": ["part_ii_item_1a", "part_ii_item_1", "part_i_item_2", "part_i_item_3"],
        "primary_section": "part_ii_item_1a",
        "mda_section": "part_i_item_2",
    }

    text_col, mda_col = resolve_section_columns(sec_cfg)

    assert text_col == "section_part_ii_item_1a"
    assert mda_col == "section_part_i_item_2"
