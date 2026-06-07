import os
import logging
from collections import defaultdict
from typing import Optional

import hydra
import pandas as pd
from edgar import Company, set_identity
from edgar import httpclient
from edgar.entity.filings import EntityFilings
from omegaconf import DictConfig
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "httpxthrottlecache",
    "edgar",
    "edgar.documents",
    "edgar.documents.extractors",
]


SECTION_TITLES = {
    "part_i_item_1": "Part I, Item 1 - Financial Statements",
    "part_i_item_2": "Part I, Item 2 - MD&A",
    "part_i_item_3": "Part I, Item 3 - Market Risk",
    "part_i_item_4": "Part I, Item 4 - Controls and Procedures",
    "part_ii_item_1": "Part II, Item 1 - Legal Proceedings",
    "part_ii_item_1a": "Part II, Item 1A - Risk Factors",
    "part_ii_item_2": "Part II, Item 2 - Unregistered Sales",
    "part_ii_item_3": "Part II, Item 3 - Defaults",
    "part_ii_item_4": "Part II, Item 4 - Mine Safety",
    "part_ii_item_5": "Part II, Item 5 - Other Information",
    "part_ii_item_6": "Part II, Item 6 - Exhibits",
}

SECTION_SHORT_NAMES = {
    "part_i_item_1": "I.1",
    "part_i_item_2": "I.2",
    "part_i_item_3": "I.3",
    "part_i_item_4": "I.4",
    "part_ii_item_1": "II.1",
    "part_ii_item_1a": "II.1A",
    "part_ii_item_2": "II.2",
    "part_ii_item_3": "II.3",
    "part_ii_item_4": "II.4",
    "part_ii_item_5": "II.5",
    "part_ii_item_6": "II.6",
}


SECTION_KEYWORDS = {
    "part_i_item_1": [
        "Part I, Item 1", "PART I, ITEM 1", "Financial Statements",
        "Item 1.", "ITEM 1.",
    ],
    "part_i_item_2": [
        "Part I, Item 2", "PART I, ITEM 2",
        "Management's Discussion and Analysis",
        "Item 2.", "ITEM 2.",
    ],
    "part_i_item_3": [
        "Part I, Item 3", "PART I, ITEM 3",
        "Quantitative and Qualitative Disclosures", "Item 3.", "ITEM 3.",
    ],
    "part_i_item_4": [
        "Part I, Item 4", "PART I, ITEM 4", "Controls and Procedures",
        "Item 4.", "ITEM 4.",
    ],
    "part_ii_item_1": [
        "Part II, Item 1", "PART II, ITEM 1", "Legal Proceedings",
        "Item 1.", "ITEM 1.",
    ],
    "part_ii_item_1a": [
        "Part II, Item 1A", "PART II, ITEM 1A", "Risk Factors",
        "RISK FACTORS", "Item 1A.", "ITEM 1A.",
    ],
    "part_ii_item_2": [
        "Part II, Item 2", "PART II, ITEM 2",
        "Unregistered Sales of Equity Securities", "Item 2.", "ITEM 2.",
    ],
    "part_ii_item_3": [
        "Part II, Item 3", "PART II, ITEM 3",
        "Defaults Upon Senior Securities", "Item 3.", "ITEM 3.",
    ],
    "part_ii_item_4": [
        "Part II, Item 4", "PART II, ITEM 4", "Mine Safety Disclosures",
        "Item 4.", "ITEM 4.",
    ],
    "part_ii_item_5": [
        "Part II, Item 5", "PART II, ITEM 5", "Other Information",
        "Item 5.", "ITEM 5.",
    ],
    "part_ii_item_6": [
        "Part II, Item 6", "PART II, ITEM 6", "Exhibits",
        "Item 6.", "ITEM 6.",
    ],
}


def _quiet_external_loggers() -> None:
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _get_company_filings(
    cik_or_name: str, form: str, start: str, end: str
) -> Optional[EntityFilings]:
    """Fetch filings for a company within a date range."""
    try:
        company = Company(cik_or_name)
        filings = company.get_filings(form=form, filing_date=f"{start}:{end}")
        if filings is None or len(filings) == 0:
            return None
        return filings
    except Exception as exc:
        logger.error("Failed to fetch filings for %s: %s", cik_or_name, exc)
        return None


def _extract_section_text(doc, section_name: str) -> Optional[str]:
    """Extract section text (e.g. Item 1A) from a filing document."""
    try:
        section = doc.get_section(section_name)
        if section:
            return section.text(clean=True, table_max_col_width=500)
    except Exception:
        pass
    return None


def _configure_cache(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    def get_cache_directory() -> str:
        return cache_dir

    httpclient.get_cache_directory = get_cache_directory


def _get_document_entity(filing):
    doc = filing.obj()
    return doc.document


def _extract_section_with_fallback(filing, sections: list) -> dict:
    """Try section extraction, then fall back to full-text search for missing sections."""
    try:
        doc = _get_document_entity(filing)
        section_texts = {}

        if hasattr(doc, "get_section"):
            for section in sections:
                text = _extract_section_text(doc, section)
                section_texts[f"section_{section}"] = text

        missing_sections = [
            section for section in sections
            if not section_texts.get(f"section_{section}")
        ]
        if missing_sections:
            try:
                full_text = doc.text(clean=True, include_tables=False, table_max_col_width=200)
                for section in missing_sections:
                    keywords = SECTION_KEYWORDS.get(section, [f"ITEM {section}", f"Item {section}"])
                    start_idx = -1
                    for kw in keywords:
                        idx = full_text.find(kw)
                        if idx != -1:
                            start_idx = idx
                            break

                    if start_idx != -1:
                        # Take up to 15000 chars from that point
                        section_texts[f"section_{section}"] = full_text[start_idx:start_idx + 15000]
                    else:
                        section_texts[f"section_{section}"] = None
            except Exception:
                for section in missing_sections:
                    section_texts[f"section_{section}"] = section_texts.get(f"section_{section}")

        return section_texts
    except Exception as exc:
        logger.error("Document extraction failed: %s", exc)
        return {f"section_{s}": None for s in sections}


def _empty_section_metrics(sections: list) -> dict:
    return {
        section: {"extracted": 0, "missing": 0}
        for section in sections
    }


def _update_section_metrics(metrics: dict, records, sections: list) -> None:
    for record in records:
        _update_section_metrics_for_record(metrics, record, sections)


def _update_section_metrics_for_record(metrics: dict, record: dict, sections: list) -> None:
    for section in sections:
        value = record.get(f"section_{section}")
        if isinstance(value, str) and value.strip():
            metrics[section]["extracted"] += 1
        else:
            metrics[section]["missing"] += 1


def _format_live_status(stats: dict, section_metrics: dict, sections: list) -> str:
    section_bits = []
    for section in sections:
        counts = section_metrics[section]
        short = SECTION_SHORT_NAMES.get(section, section)
        section_bits.append(f"{short}={counts['extracted']}/{counts['missing']}")

    return (
        f"dl={stats['tickers_downloaded']} "
        f"skip={stats['tickers_skipped']} "
        f"none={stats['tickers_no_filings']} "
        f"filings={stats['filings_available']} | "
        + " ".join(section_bits)
    )


def _print_summary(stats: dict, section_metrics: dict) -> None:
    print("\nSEC loader summary")
    print("=" * 72)
    print(f"Tickers total:      {stats['tickers_total']}")
    print(f"Tickers downloaded: {stats['tickers_downloaded']}")
    print(f"Tickers skipped:    {stats['tickers_skipped']}")
    print(f"Tickers no filings: {stats['tickers_no_filings']}")
    print(f"Filings saved:      {stats['filings_saved']}")
    print(f"Filings available:  {stats['filings_available']}")
    print("\nSection extraction")
    print("-" * 72)
    print(f"{'section':20s} {'title':34s} {'extracted':>9s} {'missing':>8s} {'rate':>7s}")
    for section, counts in section_metrics.items():
        extracted = counts["extracted"]
        missing = counts["missing"]
        total = extracted + missing
        rate = (extracted / total * 100) if total else 0.0
        title = SECTION_TITLES.get(section, section)
        print(f"{section:20s} {title[:34]:34s} {extracted:9d} {missing:8d} {rate:6.1f}%")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _quiet_external_loggers()
    cache_dir = os.path.join(cfg.data.raw_dir, "sec_cache")
    _configure_cache(cache_dir)
    set_identity(cfg.sec.identity)

    # Load tickers from file
    tickers_file = cfg.data.tickers_file
    if not os.path.exists(tickers_file):
        raise FileNotFoundError(f"Tickers file not found: {tickers_file}")
    with open(tickers_file) as f:
        tickers = [line.strip() for line in f if line.strip()]

    start_date = cfg.data.start_date
    end_date = cfg.data.end_date
    raw_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    os.makedirs(raw_dir, exist_ok=True)

    forms = list(cfg.sec.forms)
    sections = list(cfg.sec.sections)

    stats = defaultdict(int)
    stats["tickers_total"] = len(tickers)
    section_metrics = _empty_section_metrics(sections)

    print(f"SEC loader: {len(tickers)} tickers from {tickers_file}")
    print(f"Forms: {', '.join(forms)} | Sections: {', '.join(sections)}")
    print(f"Date range: {start_date} to {end_date}\n")

    progress = tqdm(tickers, desc="SEC filings", unit="ticker")
    progress.set_postfix_str(_format_live_status(stats, section_metrics, sections))

    for ticker in progress:
        output_path = os.path.join(raw_dir, f"{ticker}_filings.parquet")
        if os.path.exists(output_path):
            stats["tickers_skipped"] += 1
            try:
                existing_df = pd.read_parquet(output_path)
                existing_records = existing_df.to_dict("records")
                _update_section_metrics(section_metrics, existing_records, sections)
                stats["filings_available"] += len(existing_records)
            except Exception as exc:
                logger.warning("Could not read existing file for %s: %s", ticker, exc)
            progress.set_postfix_str(_format_live_status(stats, section_metrics, sections))
            continue

        records = []

        for form in forms:
            filings = _get_company_filings(ticker, form, start_date, end_date)
            if filings is None:
                continue

            for filing in filings:
                try:
                    section_texts = _extract_section_with_fallback(filing, sections)

                    record = {
                        "ticker": ticker,
                        "cik": getattr(filing, "cik", None),
                        "company": getattr(filing, "company", ticker),
                        "accession_number": getattr(filing, "accession_number", None),
                        "form_type": getattr(filing, "form", form),
                        "filed_at": str(getattr(filing, "filing_date", ""))
                        if getattr(filing, "filing_date", None)
                        else None,
                        "period_of_report": str(getattr(filing, "period_of_report", ""))
                        if getattr(filing, "period_of_report", None)
                        else None,
                        "filing_url": getattr(filing, "url", None),
                    }

                    for section in sections:
                        record[f"section_{section}"] = section_texts.get(
                            f"section_{section}", None
                        )

                    records.append(record)
                    _update_section_metrics_for_record(section_metrics, record, sections)
                    stats["filings_available"] += 1
                    progress.set_postfix_str(_format_live_status(stats, section_metrics, sections))
                except Exception as exc:
                    stats["filing_errors"] += 1
                    logger.error("Failed to process filing for %s: %s", ticker, exc)

        if records:
            pd.DataFrame(records).to_parquet(output_path, index=False)
            stats["tickers_downloaded"] += 1
            stats["filings_saved"] += len(records)
        else:
            stats["tickers_no_filings"] += 1

        progress.set_postfix_str(_format_live_status(stats, section_metrics, sections))

    _print_summary(stats, section_metrics)


if __name__ == "__main__":
    main()
