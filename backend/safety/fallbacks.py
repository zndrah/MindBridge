"""
MindBridge 2.0 - Vetted Safety Fallbacks and Disclaimers
========================================================

Authoritative, pre-approved fallback messages, non-diagnostic disclaimers,
and clinical constraint lexicons grounded in:
- docs/AGENTS.md §9 (Response Generation Node & Verification Gate)
- docs/PRD.md §5.2 (NFR-2: Privacy-by-Design & Input Safety)
- docs/SAFETY_POLICY.md §4 (Non-Clinical Safety Boundaries)
"""

import re
from typing import Tuple

# Standard neutral supportive fallback template (docs/AGENTS.md §9.2)
# Strictly prohibits unconditional breathing exercises or unsolicited somatic prescriptions
NEUTRAL_SUPPORTIVE_FALLBACK = (
    "I am here with you. I hear what you are experiencing, and I am ready to support "
    "you with helpful information or resources whenever you are ready."
)

# Mandatory screening disclaimer for clinical screening rubrics (docs/AGENTS.md §9.1)
MANDATORY_SCREENING_DISCLAIMER = (
    "CRITICAL: This screening rubric is an informational educational tool, not a diagnostic confirmation. "
    "It cannot replace evaluation by a qualified medical professional."
)

# Prohibited clinical and diagnostic phrasing patterns
PROHIBITED_DIAGNOSTIC_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\b(?:you\s+have|diagnosed\s+with|i\s+diagnose\s+you\s+with)\s+(?:major\s+depressive\s+disorder|depression|bipolar|panic\s+disorder|schizophrenia|ptsd|generalized\s+anxiety\s+disorder|clinical\s+insomnia)\b", re.IGNORECASE),
    re.compile(r"\b(?:take|prescribe|administer|dose|dosage\s+of)\s+\d+\s*(?:mg|milligrams|g|ml)\b", re.IGNORECASE),
    re.compile(r"\b(?:you\s+should\s+take|start\s+taking|prescribe\s+you)\s+(?:ssri|lexapro|prozac|zoloft|xanax|ativan|klonopin|valium|ambien|sertraline|fluoxetine|escitalopram)\b", re.IGNORECASE),
    re.compile(r"\b(?:we\s+have\s+dispatched|dispatching\s+emergency|calling\s+(?:911|112|emergency)\s+for\s+you|sending\s+an\s+ambulance)\b", re.IGNORECASE),
)


def contains_prohibited_clinical_claims(text: str) -> bool:
    """
    Checks if a draft text contains prohibited diagnostic claims,
    pharmacological dosing, or automated emergency dispatch assertions.
    """
    if not text:
        return False
    for pattern in PROHIBITED_DIAGNOSTIC_PATTERNS:
        if pattern.search(text):
            return True
    return False
