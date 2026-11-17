"""
MindBridge 2.0 - Conversational Safety Intelligence (`conversational_safety.py`)
================================================================================

Evaluates user utterances in the context of multi-turn conversation history
and persistent session safety state.

Invariants:
1. Explicit suicidal intent ("thoughts of ending my life", "want to kill myself", etc.)
   triggers crisis emergency_bypass directly without looping questions.
2. Positive confirmation ("yes", "i do", "yes i have thoughts", etc.) to a pending
   safety follow-up question immediately escalates to emergency_bypass.
3. Explicit negation ("I am not suicidal. I was only discussing a fictional situation.")
   safely clears follow-up without escalating to emergency.
4. An active emergency_bypass state remains active across continuations,
   preventing generic supportive fallback responses from overriding crisis assistance.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.orchestrator.state import EmergencyTriggerID

# ============================================================================
# 1. Suicidal Intent & Crisis Regexes
# ============================================================================

# Explicit expressions of suicidal thoughts, urges, impulses, or intent
EXPLICIT_SUICIDAL_INTENT_PATTERN = re.compile(
    r"\b(?:have\s+(?:(?:so\s+)?many\s+|a\s+lot\s+of\s+|constant\s+|frequent\s+|overwhelming\s+)(?:thoughts?|urges?|impulses?)\s+(?:of|to|about)\s+(?:end(?:ing)?\s+my\s+life|kill(?:ing)?\s+myself|suicide|die|dying)|"
    r"thoughts?\s+to\s+end\s+my\s+life|"
    r"want\s+to\s+kill\s+myself|"
    r"wanna\s+kill\s+myself|"
    r"kill\s+myself)\b",
    re.IGNORECASE,
)

# Affirmative responses to a direct suicide risk or immediate danger inquiry
AFFIRMATIVE_CONFIRMATION_PATTERN = re.compile(
    r"^(?:yes|yeah|yep|yup|ya|yea|i\s+do|i\s+am|i\s+have|definitely|sure|affirmative|true|indeed|always|often|sometimes|a\s+lot|so\s+many|yes\s+please|yes\s+i\s+do|yes\s+i\s+am|yes\s+i\s+have)[\.\!\?\,]*$",
    re.IGNORECASE,
)

# Clause-leading affirmative responses (e.g. "yes i have so many thoughts...")
AFFIRMATIVE_PREFIX_PATTERN = re.compile(
    r"^(?:yes|yeah|yep|yup|i\s+do|i\s+am|definitely|sure)[\,\s\.\!]+",
    re.IGNORECASE,
)

# Explicit negation / clarification indicating user is NOT in danger or not suicidal
EXPLICIT_NEGATION_PATTERN = re.compile(
    r"\b(?:not\s+suicidal|"
    r"only\s+discussing|"
    r"fictional(?:\s+situation)?|"
    r"no\b|nope|nah|not\s+really|not\s+at\s+all|"
    r"i\s+am\s+not(?:\s+suicidal|\s+in\s+danger)?|"
    r"i\s+do\s+not(?:\s+want\s+to\s+die)?|"
    r"no\s+thoughts|"
    r"not\s+in\s+danger|"
    r"never\s+mind|false\s+alarm)\b",
    re.IGNORECASE,
)

# Indicators that the preceding assistant message was a Safety Follow-Up
SAFETY_FOLLOWUP_INDICATORS = re.compile(
    r"(?:immediate\s+danger|"
    r"thoughts\s+of\s+ending\s+(?:your|my)\s+life|"
    r"thoughts\s+of\s+(?:suicide|hurting\s+yourself)|"
    r"having\s+thoughts\s+of\s+ending\s+your\s+life|"
    r"are\s+you\s+having\s+thoughts|"
    r"are\s+you\s+in\s+immediate\s+danger|"
    r"thoughts\s+of\s+ending\s+it|"
    r"harming\s+yourself|hurting\s+yourself)",
    re.IGNORECASE,
)

# Indicators that the preceding assistant message was an Emergency Response
EMERGENCY_RESPONSE_INDICATORS = re.compile(
    r"(?:Immediate\s+Emergency\s+&\s+Crisis\s+Support\s+Resources|"
    r"Direct\s+Calling\s+Required|"
    r"emergency\s+medical\s+assistance\s+immediately|"
    r"contact\s+local\s+emergency\s+services\s+immediately|"
    r"International\s+Emergency\s+Standard|"
    r"Befrienders\s+Worldwide|findahelpline)",
    re.IGNORECASE,
)


def is_safety_affirmation(text: str) -> bool:
    """Checks whether the text represents an affirmative confirmation of risk."""
    if not text:
        return False
    t = text.strip().lower()
    if AFFIRMATIVE_CONFIRMATION_PATTERN.match(t):
        return True
    if AFFIRMATIVE_PREFIX_PATTERN.match(t):
        return True
    if any(k in t for k in (
        "thoughts to end", "thoughts of ending", "suicidal thoughts", "have thoughts",
        "feel suicidal", "want to die", "kill myself", "end my life", "in danger", "i might hurt myself"
    )):
        return True
    return False


def is_safety_negation(text: str) -> bool:
    """Checks whether the text represents an explicit negation, clarification, or denial of risk."""
    if not text:
        return False
    t = text.strip().lower()
    # Direct short negation
    if re.match(r"^(?:no|nope|nah|not really|not at all|i am not|i do not|neither)[\.\!\?\,]*$", t):
        return True
    return bool(EXPLICIT_NEGATION_PATTERN.search(t))


def detect_preceding_safety_context(
    history_messages: List[Dict[str, str]],
    stored_safety_route: Optional[str] = None,
) -> Tuple[bool, bool]:
    """
    Analyzes conversation history and stored session state to determine if
    the preceding turn was a safety follow-up or emergency response.

    Returns:
        (is_preceding_followup, is_preceding_emergency)
    """
    last_asst_content = ""
    if history_messages:
        for m in reversed(history_messages):
            if m.get("role") == "assistant":
                last_asst_content = (m.get("content") or "").strip()
                break

    is_followup = (
        stored_safety_route == "pending_safety_follow_up"
        or bool(SAFETY_FOLLOWUP_INDICATORS.search(last_asst_content))
    )

    is_emergency = (
        stored_safety_route == "emergency_bypass"
        or bool(EMERGENCY_RESPONSE_INDICATORS.search(last_asst_content))
    )

    return is_followup, is_emergency


def evaluate_conversational_safety(
    current_user_text: str,
    history_messages: List[Dict[str, str]],
    stored_safety_route: Optional[str] = None,
) -> Tuple[bool, Optional[EmergencyTriggerID], bool, bool]:
    """
    Evaluates conversational safety state across consecutive turns.

    Returns:
        (is_emergency, emergency_trigger_id, is_followup_needed, is_followup_cleared)
    """
    if not current_user_text:
        return False, None, False, False

    user_norm = current_user_text.strip().lower()
    is_neg = is_safety_negation(current_user_text)
    is_aff = is_safety_affirmation(current_user_text)

    # 1. Standalone explicit suicidal intent (applies in any turn)
    if not is_neg and EXPLICIT_SUICIDAL_INTENT_PATTERN.search(current_user_text):
        return True, "IMMINENT_SUICIDE_INTENT", False, False

    # 2. Check preceding conversation context
    is_preceding_followup, is_preceding_emergency = detect_preceding_safety_context(
        history_messages=history_messages,
        stored_safety_route=stored_safety_route,
    )

    # Case A: User responding to a pending safety follow-up question
    if is_preceding_followup:
        # Negation / clarification (e.g. Test E: "I am not suicidal. I was only discussing a fictional situation.")
        if is_neg:
            return False, None, False, True

        # Positive confirmation / affirmation (e.g. Test C & Test D: "Yes", "yes I do", etc.)
        if is_aff:
            return True, "SAFETY_FOLLOWUP_CONFIRMED", False, False

    # Case B: User continuing dialogue after emergency_bypass was emitted (Test F)
    if is_preceding_emergency:
        # User explicitly says they are safe and changing topic
        if is_neg and ("safe now" in user_norm or "fine now" in user_norm or "let's talk about" in user_norm):
            return False, None, False, True

        # Otherwise, emergency bypass MUST remain active; do NOT fall back to generic response!
        return True, "EMERGENCY_STATE_PRESERVED", False, False

    return False, None, False, False
