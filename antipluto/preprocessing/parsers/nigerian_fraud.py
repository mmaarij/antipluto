"""
parsers/nigerian_fraud.py
=========================
Parser for the CLAIR Nigerian Fraud (419 Advance-Fee) Email Corpus.

Dataset Overview
----------------
The CLAIR corpus (Radev, 2008) is a collection of approximately 3,977
Nigerian advance-fee fraud emails (419 scam) compiled by the Computational
Linguistics and Information Retrieval (CLAIR) group at the University of
Michigan.  It is one of the five source datasets used in the MeAJOR corpus
(Mendes et al., 2025).

All records are labelled ``1`` (phishing / fraud).

File Layout
-----------
::

    nigerian_fraud/
      fradulent_emails.txt   <- single Unix mbox (~17.3 MB, ~3,977 emails)

Note: the filename ``fradulent_emails.txt`` preserves the original typo
("fradulent" instead of "fraudulent") from the CLAIR Kaggle release.

Format
------
Standard Unix mbox format -- RFC-4155 compliant, identical to the Nazario
archive format.  Parsed using ``mailbox.mbox`` for correct RFC-compliant
message separation.

Parallelism
-----------
The corpus is small (~3,977 emails).  Sequential single-process parsing is
sufficient; no worker pool is needed.

Citation
--------
Radev, D. (2008). CLAIR collection of fraud email.
ACL Data and Code Repository, ADCR2008T001.
https://aclweb.org/aclwiki/CLAIR_collection_of_fraud_email_(Repository)
"""

from __future__ import annotations

import logging
import mailbox
from pathlib import Path
from typing import Iterator

from antipluto.preprocessing.cleaners.html_stripper import HTMLStripper
from antipluto.preprocessing.cleaners.header_slicer import HeaderSlicer
from antipluto.preprocessing.parsers.base import extract_email_fields

logger = logging.getLogger(__name__)

_SOURCE = "nigerian_fraud"
_LABEL = 1  # All records are phishing / advance-fee fraud.

# Known filename for the CLAIR mbox file (preserves original typo).
_MBOX_FILENAME = "fradulent_emails.txt"


# ---------------------------------------------------------------------------
# NigerianFraudParser
# ---------------------------------------------------------------------------


class NigerianFraudParser:
    """
    Ingests the CLAIR Nigerian Fraud email corpus from a single mbox file.

    Parameters
    ----------
    source_config : dict
        Source configuration block (``config["sources"]["nigerian_fraud"]``).
    processing_config : dict
        Global processing settings.
    project_root : Path
        Absolute path to the thesis project root.
    """

    def __init__(
        self,
        source_config: dict,
        processing_config: dict,
        project_root: Path,
    ) -> None:
        self.min_body_length: int = processing_config.get("min_body_length", 50)

    def parse(self, source_path: Path) -> Iterator[dict]:
        """
        Parse the CLAIR Nigerian fraud mbox and yield cleaned, labelled records.

        Parameters
        ----------
        source_path : Path
            Absolute path to the ``nigerian_fraud/`` directory.

        Yields
        ------
        dict
            Raw pre-validation record with keys:
            ``{subject, body, label, source}``.
        """
        stripper = HTMLStripper()
        slicer = HeaderSlicer()

        # Resolve the mbox file path.  Accept the source_path pointing either
        # to the directory or directly to the mbox file for flexibility.
        if source_path.is_file():
            mbox_path = source_path
        else:
            mbox_path = source_path / _MBOX_FILENAME

        if not mbox_path.exists():
            logger.error(
                "NigerianFraud > mbox file not found at %s -- skipping.", mbox_path
            )
            return

        logger.info("NigerianFraud > Parsing mbox: %s", mbox_path)

        kept = 0
        dropped = 0
        errors = 0

        try:
            mbox = mailbox.mbox(str(mbox_path), create=False)
        except Exception as exc:
            logger.error(
                "NigerianFraud > Failed to open %s: %s", mbox_path.name, exc
            )
            return

        try:
            # Iterate by key (not by message object directly) to safely catch
            # per-message UnicodeDecodeError on non-ASCII From_ separator lines,
            # matching the pattern established in NazarioParser.
            for key in mbox.keys():
                try:
                    msg = mbox[key]
                    result = extract_email_fields(
                        msg,
                        stripper=stripper,
                        slicer=slicer,
                        min_body_length=self.min_body_length,
                        label=_LABEL,
                        source=_SOURCE,
                    )
                    if result is not None:
                        kept += 1
                        yield result
                    else:
                        dropped += 1
                except UnicodeDecodeError:
                    # Non-ASCII bytes in the mbox From_ separator line.
                    errors += 1
                except Exception as exc:
                    logger.debug(
                        "NigerianFraud > Error parsing message key=%s: %s",
                        key,
                        exc,
                    )
                    errors += 1
        finally:
            mbox.close()

        logger.info(
            "NigerianFraud > Complete. kept=%d dropped=%d errors=%d",
            kept,
            dropped,
            errors,
        )
