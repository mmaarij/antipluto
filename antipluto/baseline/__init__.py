"""
antipluto.baseline
================
Spam filter baseline comparison suite.

Evaluates the unseen test partition against industry-standard open-source
email security engines (Apache SpamAssassin and Rspamd) alongside Anti-PLUTO:
  - Apache SpamAssassin (spamd on port 783 via SPAMC/1.5 protocol)
  - Rspamd (HTTP REST API on localhost:11333 via /checkv2)
  - Anti-PLUTO (XGBoost Fusion Head)
"""

from __future__ import annotations

from antipluto.baseline.mime_envelope import build_mime_message
from antipluto.baseline.unmasked_loader import (
    UnmaskedEmailRecord,
    load_unmasked_test_records,
)
from antipluto.baseline.clients import (
    BaselineFilterResult,
    SpamAssassinClient,
    RspamdClient,
)
from antipluto.baseline.evaluator import (
    evaluate_external_baselines,
    compute_binary_metrics,
)

__all__ = [
    "build_mime_message",
    "UnmaskedEmailRecord",
    "load_unmasked_test_records",
    "BaselineFilterResult",
    "SpamAssassinClient",
    "RspamdClient",
    "evaluate_external_baselines",
    "compute_binary_metrics",
]
