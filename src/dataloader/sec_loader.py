import os
import logging
from typing import Optional

import hydra
import pandas as pd
from edgar import Company, set_identity
from edgar import httpclient
from edgar.entity.filings import EntityFilings
from omegaconf import DictConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _get_company_filings(
    cik_or_name: str, form: str, start: str, end: str
) -> Optional[EntityFilings]:
    """Fetch filings for a company within a date range."""

    try:
        company = Company(cik_or_name)
        filings = company.get_filings(form=form, filing_date=f"{start}:{end}")
        if filings is None or len(filings) == 0:
            logger.warning("No filings found for %s", cik_or_name)
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
    except Exception as exc:
        logger.warning("Failed to extract %s: %s", section_name, exc)
    return None


def _configure_cache(cache_dir: str) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    def get_cache_directory() -> str:
        return cache_dir

    httpclient.get_cache_directory = get_cache_directory

def _get_document_entity(filing):
    doc = filing.obj()
    return doc.document

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cache_dir = os.path.join(cfg.data.raw_dir, "sec_cache")
    _configure_cache(cache_dir)
    set_identity(cfg.sec.identity)

    tickers = cfg.data.tickers
    start_date = cfg.data.start_date
    end_date = cfg.data.end_date
    raw_dir = os.path.join(cfg.data.raw_dir, "sec_filings")
    os.makedirs(raw_dir, exist_ok=True)

    forms = cfg.sec.forms
    sections = cfg.sec.sections

    for ticker in tickers:
        records = []
        logger.info("Fetching SEC filings for %s", ticker)

        for form in forms:
            filings = _get_company_filings(ticker, form, start_date, end_date)
            if filings is None:
                continue

            for filing in filings:
                try:
                    doc = _get_document_entity(filing)
                    text_content = None
                    section_texts = {}

                    if hasattr(doc, "get_section"):
                        for section in sections:
                            section_texts[f"section_{section}"] = _extract_section_text(
                                doc, section
                            )

                        # Fallback to full text if no specific sections
                        if not any(section_texts.values()):
                            text_content = doc.text(
                                clean=True, include_tables=True, table_max_col_width=500
                            )
                    else:
                        logger.warning(
                            "Document for %s does not support section extraction",
                            ticker,
                        )
                        text_content = doc.text(
                                clean=True, include_tables=True, table_max_col_width=500
                                )
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

                    record["document_text"] = text_content

                    records.append(record)
                except Exception as exc:
                    logger.error("Failed to process filing for %s: %s", ticker, exc)

        if records:
            output_path = os.path.join(raw_dir, f"{ticker}_filings.parquet")
            pd.DataFrame(records).to_parquet(output_path, index=False)
            logger.info("Saved %d filings for %s", len(records), ticker)
        else:
            logger.warning("No filings saved for %s", ticker)


if __name__ == "__main__":
    main()
