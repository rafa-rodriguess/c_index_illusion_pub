"""
Assumption-aligned survival metrics for Block E (anchor ladder).

Metric implementations are filled in incrementally; E0n scripts call these
APIs and may write ``implementation_status: stub`` until each degrau is live.
"""

from src.metrics.freeze import (
    FROZEN_MANIFEST_NAME,
    load_frozen_manifest,
    require_protocol_freeze,
    sha256_file,
)

__all__ = [
    "FROZEN_MANIFEST_NAME",
    "load_frozen_manifest",
    "require_protocol_freeze",
    "sha256_file",
]
