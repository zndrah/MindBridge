"""
MindBridge 2.0 - Orchestrator Fallbacks & Disclaimers Re-Export
==============================================================

Re-exports authoritative fallback messages, non-diagnostic disclaimers,
and clinical constraint checkers from backend.safety.fallbacks.
"""

from backend.safety.fallbacks import (
    MANDATORY_SCREENING_DISCLAIMER,
    NEUTRAL_SUPPORTIVE_FALLBACK,
    PROHIBITED_DIAGNOSTIC_PATTERNS,
    contains_prohibited_clinical_claims,
)

__all__ = [
    "MANDATORY_SCREENING_DISCLAIMER",
    "NEUTRAL_SUPPORTIVE_FALLBACK",
    "PROHIBITED_DIAGNOSTIC_PATTERNS",
    "contains_prohibited_clinical_claims",
]
