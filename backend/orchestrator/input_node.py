"""
MindBridge 2.0 - Input Processing Node (`input_node`)
=====================================================

This module implements the LangGraph input ingestion, pre-truncation safety scanning,
message history validation, and PII sanitization boundary for MindBridge 2.0.

CRITICAL PRIVACY & SAFETY RULES:
--------------------------------
1. EPHEMERAL INPUT BOUNDARY:
   Raw user text is accepted ONLY as an ephemeral function argument (`raw_user_text: str`).
   It is NEVER added to, read from, or persisted in `MindBridgeState`.

2. PRE-TRUNCATION SAFETY SCANNING & EMERGENCY PRECISION:
   Safety evaluation takes absolute priority over message truncation or length limits.
   Deterministic emergency regex scanning runs across the COMPLETE, un-truncated input string
   before any length reduction, chunking, or sanitization occurs.
   - EXPLICIT IMMEDIATE DANGER: Active intent, possession of lethal means, imminent timeframe,
     suicide note posting, or direct self-harm activate `emergency_bypass`.
   - PASSIVE / NON-IMMINENT DISCLOSURES: Generic phrases ("want to die", "suicidal thoughts",
     "better off dead", "wish I wouldn't wake up") do NOT trigger emergency bypass.
     Instead, they initiate structured multi-turn follow-up (`pending_safety_follow_up`).

3. VALIDATION OF EXISTING CONVERSATION HISTORY:
   Never blindly copy existing state messages. Every message in existing history must strictly
   satisfy the `SanitizedMessage` contract (valid dictionary with only 'role' and 'content',
   valid role value, string content). If existing history fails validation, a controlled error
   is returned without copying untrusted or malformed content.

4. PRESERVATION OF EMERGENCY MEANING IN PII SCRUBBING:
   PII masking scrubs structured personal data (emails, personal phone numbers, national IDs, URLs)
   while explicitly exempting verified emergency dispatch numbers (112, 911, 999), mental health
   crisis lifelines (988, 14416, 741741, 85258), and medical/clinical quantities.

5. CONTROLLED TRACING & ZERO DATA LEAKAGE:
   Zero raw user text, matched user substrings, or raw exception strings (`str(e)`) may be
   stored in execution traces, error codes, or disk logs.

6. PRIVACY LIMITATION ACKNOWLEDGMENT:
   Deterministic regex scrubbing provides strong pattern-based minimization for structured
   identifiers. It does NOT guarantee complete anonymization of arbitrary personal names or
   unstructured narratives across diverse linguistic backgrounds without machine learning NER.

Authoritative References:
- docs/PRD.md §5.2 (NFR-2: Privacy-by-Design & Input Safety)
- docs/ARCHITECTURE.md §4.2 (RoBERTa Sequence Length & Long-Message Processing)
- docs/ARCHITECTURE.md §4.5 (PII Lifecycle & Ephemeral Input-Boundary Isolation)
- docs/AGENTS.md §2 (Input Processing Node Interface)
- docs/SAFETY_POLICY.md §1 & §4 (Emergency Precedence & PHQ-9 Seven Dimensions)
"""

from datetime import datetime, timezone
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

from backend.orchestrator.state import (
    EmergencyTriggerID,
    MindBridgeState,
    SanitizedMessage,
    TraceEntry,
)

# ============================================================================
# Configuration Constants
# ============================================================================

MAX_INPUT_LENGTH = 2000
"""Maximum standard character limit before long-message partitioning or truncation."""

HARD_MAX_INPUT_LENGTH = 10000
"""Absolute ceiling for input strings; payloads exceeding this are flagged as oversized."""

SAFETY_SCAN_CEILING = 25000
"""Maximum character boundary for pre-truncation safety scanning to prevent unbounded ReDoS/DoS while covering long multi-page messages."""

# Verified emergency hotline numbers and crisis short codes from backend/safety/hotlines.json.
# These must NEVER be masked by PII regexes to preserve critical emergency context.
EMERGENCY_WHITELIST = frozenset({
    "000",
    "102",
    "111",
    "112",
    "116123",
    "131114",
    "1300224636",
    "1300659467",
    "14416",
    "18005990019",
    "686868",
    "741741",
    "85258",
    "911",
    "988",
    "999",
    "9999666555",
})

# ============================================================================
# Pre-Truncation Emergency Trigger Regexes (Automatic emergency_bypass)
# ============================================================================

# These patterns strictly require explicit immediate peril, active intent,
# possession/access to lethal mechanisms, immediate crisis timeframe,
# suicide notes, or direct acute self-harm threats.
# Generic phrases ("want to die", "suicidal thoughts", "better off dead")
# are EXCLUDED from this list to prevent inappropriate emergency takeover.
EMERGENCY_TRIGGERS: List[Tuple[EmergencyTriggerID, re.Pattern[str]]] = [
    (
        "SUICIDE_NOTE_POSTING",
        re.compile(
            r"\b(?:this\s+is\s+my\s+(?:suicide\s+note|last\s+note|final\s+goodbye|goodbye\s+note)|"
            r"writing\s+my\s+suicide\s+note|posting\s+my\s+suicide\s+note)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "LETHAL_MEANS_DISCLOSURE",
        re.compile(
            r"\b(?:have\s+(?:a\s+gun|a\s+weapon|pills|a\s+noose|a\s+knife)|"
            r"bottle\s+of\s+pills|swallow(?:ing)?\s+(?:all\s+the\s+)?pills|overdos(?:e|ing)|"
            r"jump(?:ing)?\s+off\s+(?:a\s+)?(?:bridge|building|roof)|"
            r"knife\s+to\s+my\s+(?:wrist|throat)|hanging\s+from\s+the\s+ceiling)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "IMMINENT_TIMEFRAME_CRISIS",
        re.compile(
            r"\b(?:tonight\s+(?:is\s+the\s+night|i\s+(?:will|am\s+going\s+to)\s+end\s+it)|"
            r"in\s+a\s+few\s+minutes|goodbye\s+forever|won'?t\s+be\s+here\s+tomorrow|"
            r"last\s+(?:day|hours?|night)\s+alive)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "IMMINENT_SUICIDE_INTENT",
        re.compile(
            r"\b(?:(?:going\s+to|planning\s+to|ready\s+to|decided\s+to|about\s+to|want\s+to|wanna|plan\s+to|will)\s+(?:kill\s+myself|end\s+my\s+life|hang\s+myself|take\s+my\s+own\s+life|(?:(?:commit|do|attempt)\s+)?suicide(?:\s+myself)?)|"
            r"i\s+(?:will|am\s+going\s+to|want\s+to|wanna|'m\s+going\s+to|m\s+going\s+to|plan\s+to)\s+(?:kill\s+myself|end\s+my\s+life|(?:(?:commit|do|attempt)\s+)?suicide(?:\s+myself)?)|"
            r"(?:want|going|plan)\s+to\s+(?:(?:commit|do|attempt)\s+)?suicide|"
            r"(?:commit|do|attempt)\s+suicide|"
            r"have\s+(?:(?:so\s+)?many\s+|a\s+lot\s+of\s+|constant\s+|frequent\s+|overwhelming\s+)(?:thoughts?|urges?|impulses?)\s+(?:of|to|about)\s+(?:end(?:ing)?\s+my\s+life|kill(?:ing)?\s+myself)|"
            r"thoughts?\s+to\s+end\s+my\s+life|"
            r"want\s+to\s+kill\s+myself|"
            r"wanna\s+kill\s+myself|"
            r"kill\s+myself|"
            r"ending\s+my\s+life\s+(?:tonight|today|now)|going\s+to\s+hang\s+myself|"
            r"am\s+going\s+to\s+end\s+my\s+life|going\s+to\s+end\s+my\s+life)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SELF_HARM_DIRECT_THREAT",
        re.compile(
            r"\b(?:cutting\s+my(?:self|\s+wrists?)|slitting\s+my\s+wrists?|bleed(?:ing)?\s+out|"
            r"burning\s+my(?:self|\s+skin)|"
            r"(?:(?:feel\s+like\s+i\s+might|might|feel\s+like|going\s+to|about\s+to)\s+(?:hurt|harm|injure|cut)\s+myself\s+(?:right\s+now|at\s+this\s+moment))|"
            r"(?:hurt|harm|injure)\s+myself\s+right\s+now|"
            r"(?:feel\s+like\s+i\s+might\s+hurt\s+myself\s+right\s+now))\b",
            re.IGNORECASE,
        ),
    ),
]

# ============================================================================
# Non-Emergency Safety Follow-Up Regexes (pending_safety_follow_up)
# ============================================================================

# These patterns detect passive death wishes, non-imminent suicidal ideation,
# and direct self-harm disclosures without explicit immediate lethality.
# They initiate structured multi-turn follow-up (pending_safety_follow_up)
# without triggering disruptive emergency bypass screens.

PASSIVE_DEATH_WISHES_REGEX = re.compile(
    r"\b(?:wish(?:ed)?\s+(?:i\s+(?:would\s+not|wouldn'?t)\s+wake\s+up|i\s+were\s+dead|i\s+was\s+dead|i\s+had\s+never\s+been\s+born)|"
    r"better\s+off\s+dead|"
    r"want\s+to\s+die|"
    r"don'?t\s+want\s+to\s+(?:wake\s+up|live\s+anymore|exist\s+anymore)|"
    r"wish\s+(?:it\s+would\s+all\s+(?:just\s+)?stop|i\s+could\s+disappear)|"
    r"no\s+reason\s+to\s+(?:live|keep\s+going|stay\s+alive))\b",
    re.IGNORECASE,
)

SUICIDAL_IDEATION_REGEX = re.compile(
    r"\b(?:suicidal\s+thoughts?|"
    r"thoughts?\s+of\s+(?:suicide|ending\s+(?:my\s+life|it\s+all)|killing\s+myself)|"
    r"thinking\s+(?:about|of)\s+(?:suicide|ending\s+it\s+all|killing\s+myself)|"
    r"feel\s+suicidal|"
    r"suicidality)\b",
    re.IGNORECASE,
)

SELF_HARM_DISCLOSURE_PATTERNS = re.compile(
    r"\b(?:(?:going\s+to|planning\s+to|thinking\s+(?:about|of)|might|want\s+to|decided\s+to|about\s+to)\s+(?:hurt|harm|injure|cut|burn)\s+myself|"
    r"(?:hurt|harm|injure)\s+myself\s+(?:tonight|today|soon)|"
    r"have\s+a\s+plan\s+to\s+(?:do\s+it|hurt\s+myself|harm\s+myself|die|end\s+it)\s*(?:tonight|today)?)\b",
    re.IGNORECASE,
)

# Clause Delimiters for locally scoped contextual analysis
CLAUSE_DELIMITERS = re.compile(
    r"(?:[.!?;:\n]+|\s+--\s+|\s+—\s+|,\s*(?:and|but|however|although|though|while|whereas|yet|except|besides)\s+|\s+(?:but|however|although|though|whereas|yet)\s+)",
    re.IGNORECASE,
)

# Locally scoped contextual patterns (negation, third-party, quotes, hypotheticals, routine meds)
NEGATION_PATTERNS = re.compile(
    r"\b(?:don'?t\s+want\s+to\s+(?:die|hurt\s+myself|harm\s+myself|kill\s+myself|commit\s+suicide|do\s+suicide|suicide|end\s+my\s+life)|"
    r"do\s+not\s+want\s+to\s+(?:die|hurt\s+myself|harm\s+myself|kill\s+myself|commit\s+suicide|do\s+suicide|suicide|end\s+my\s+life)|"
    r"never\s+(?:had|have)\s+suicidal\s+thoughts?|"
    r"never\s+wanted?\s+to\s+(?:die|hurt\s+myself|harm\s+myself|kill\s+myself|commit\s+suicide|do\s+suicide|suicide|end\s+my\s+life)|"
    r"not\s+suicidal|"
    r"not\s+(?:planning|going)\s+to\s+(?:die|kill\s+myself|end\s+my\s+life|hurt\s+myself|harm\s+myself|commit\s+suicide|do\s+suicide|suicide)|"
    r"(?:am\s+not|not)\s+going\s+to\s+(?:hurt|harm|kill)\s+myself|"
    r"(?:am\s+not|not)\s+going\s+to\s+(?:commit\s+suicide|do\s+suicide|suicide|end\s+my\s+life)|"
    r"would\s+never\s+(?:kill\s+myself|commit\s+suicide|do\s+suicide|suicide|end\s+my\s+life|hurt\s+myself|harm\s+myself)|"
    r"no\s+longer\s+(?:suicidal|want\s+to\s+die|want\s+to\s+kill\s+myself)|"
    r"said\s+i\s+do\s+not\s+want\s+to\s+(?:die|kill\s+myself|do\s+suicide|commit\s+suicide))\b",
    re.IGNORECASE,
)

THIRD_PARTY_PATTERNS = re.compile(
    r"\b(?:my\s+(?:friend|brother|sister|mom|mother|dad|father|cousin|roommate|partner|wife|husband|colleague|daughter|son)|"
    r"someone\s+i\s+know|a\s+friend|worried\s+about\s+(?:my|someone|a\s+friend))\s*"
    r"(?:said\s+(?:he|she|they)|told\s+me\s+(?:he|she|they)|is|was|feels?|has)?\s*"
    r"(?:wants?\s+to\s+die|feeling\s+suicidal|is\s+suicidal|has\s+suicidal\s+thoughts?|talking\s+about\s+suicide|threatening\s+suicide)\b|"
    r"\bworried\s+about\s+my\s+(?:friend|brother|sister|mom|dad)\b",
    re.IGNORECASE,
)

QUOTE_OR_ACADEMIC_PATTERNS = re.compile(
    r"\b(?:quoting(?:\s+someone)?|in\s+the\s+book|in\s+hamlet|character\s+says|shakespeare|movie\s+where|"
    r"he\s+said\s+['\"][^'\"]*['\"]|she\s+said\s+['\"][^'\"]*['\"]|"
    r"reading\s+about\s+suicide|lyrics\s+of\s+the\s+song)\b",
    re.IGNORECASE,
)

HYPOTHETICAL_PATTERNS = re.compile(
    r"\b(?:hypothetically(?:\s+speaking)?|"
    r"what\s+if\s+someone\s+(?:wants?\s+to\s+die|feels?\s+suicidal)|"
    r"if\s+a\s+person\s+(?:wanted?\s+to\s+die|commits?\s+suicide))\b",
    re.IGNORECASE,
)

ORDINARY_MEDICATION_PATTERNS = re.compile(
    r"\b(?:prescription\s+(?:for\s+my|medication)|"
    r"(?:blood\s+pressure|cholesterol|heart\s+condition|diabetes|thyroid|insulin|antibiotics|asthma|allergy)\s+medication|"
    r"take\s+my\s+(?:morning|daily|evening)\s+(?:pills?|medication)|"
    r"vitamins?\s+(?:and\s+supplements?|daily))\b",
    re.IGNORECASE,
)

# ============================================================================
# PII Detection & Scrubbing Regexes
# ============================================================================

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

URL_REGEX = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE,
)

# Identification Numbers: US Social Security Number (SSN: 3-2-4)
SSN_REGEX = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b"
)

# Identification Numbers: Indian Aadhaar (12 digits, separated by spaces or hyphens)
AADHAAR_REGEX = re.compile(
    r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"
)

# Telephone formats (standard 10+ digits, international prefixes, area codes)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b|\b[6-9]\d{9}\b"
)


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Safely normalizes Unicode and whitespace without altering clinical meaning.

    - Applies NFKC Unicode normalization (standardizing ligatures and full-width forms).
    - Strips non-printable control characters while preserving standard whitespace.
    - Collapses consecutive whitespace and limits multiple newlines.
    - Preserves all standard emojis and multi-lingual characters.
    """
    if not text:
        return ""

    # Step 1: Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # Step 2: Strip non-printable control characters (except newline, tab, carriage return)
    cleaned_chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("C"):
            if char in ("\n", "\r", "\t"):
                cleaned_chars.append(char)
            # Other control characters are dropped
        else:
            cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)

    # Step 3: Normalize excessive whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def evaluate_clauses_for_safety(text: str) -> Tuple[bool, Optional[EmergencyTriggerID], bool]:
    """
    Evaluates text across constituent clauses to prevent composite false negatives.

    A third-party, quoted, hypothetical, or negated phrase in one clause does NOT
    suppress an explicit emergency trigger or personal self-harm disclosure in
    another clause.

    Returns:
        (is_emergency, emergency_trigger_id, is_followup_needed)
    """
    if not text:
        return False, None, False

    clauses = [c.strip() for c in CLAUSE_DELIMITERS.split(text) if c.strip()]
    if not clauses:
        return False, None, False

    has_quote = bool(QUOTE_OR_ACADEMIC_PATTERNS.search(text))
    has_hypothetical = bool(HYPOTHETICAL_PATTERNS.search(text))

    unsuppressed_emergency_id: Optional[EmergencyTriggerID] = None
    unsuppressed_followup = False

    for c in clauses:
        is_neg = bool(NEGATION_PATTERNS.search(c))
        is_tp = bool(THIRD_PARTY_PATTERNS.search(c))
        is_med = bool(ORDINARY_MEDICATION_PATTERNS.search(c))
        is_hyp = bool(HYPOTHETICAL_PATTERNS.search(c))

        # Check emergency triggers in this clause
        c_emergency_id: Optional[EmergencyTriggerID] = None
        for trig_id, pat in EMERGENCY_TRIGGERS:
            if pat.search(c):
                c_emergency_id = trig_id
                break

        # Check follow-up indicators in this clause
        c_has_followup = bool(
            PASSIVE_DEATH_WISHES_REGEX.search(c)
            or SUICIDAL_IDEATION_REGEX.search(c)
            or SELF_HARM_DISCLOSURE_PATTERNS.search(c)
        )

        # Medication suppression: routine meds do not trigger lethal means unless overdose
        if is_med and c_emergency_id == "LETHAL_MEANS_DISCLOSURE" and "overdose" not in c.lower():
            c_emergency_id = None

        # Negation suppression: e.g. "I am not going to hurt myself"
        if is_neg:
            # Negation never suppresses acute self-harm in progress or overdose
            if c_emergency_id not in ("SELF_HARM_DIRECT_THREAT", "LETHAL_MEANS_DISCLOSURE") or (
                c_emergency_id == "LETHAL_MEANS_DISCLOSURE" and "overdose" not in c.lower()
            ):
                c_emergency_id = None
            c_has_followup = False

        # Third party suppression: e.g. "My friend said he wants to die"
        if is_tp:
            c_emergency_id = None
            c_has_followup = False

        # Hypothetical suppression in clause
        if is_hyp:
            c_emergency_id = None
            c_has_followup = False

        if c_emergency_id is not None:
            unsuppressed_emergency_id = c_emergency_id
        if c_has_followup:
            unsuppressed_followup = True

    # Quoted text rule:
    # Quoted text must not automatically be treated as confirmed personal emergency disclosure.
    # However, if safety triggers (ideation, plan, means) are present in the quote,
    # route to follow-up clarification, not standard conversation!
    if has_quote:
        if unsuppressed_emergency_id or unsuppressed_followup:
            return False, None, True
        return False, None, False

    # Pure hypotheticals with no unsuppressed personal disclosure do not trigger safety route
    if has_hypothetical and not unsuppressed_emergency_id:
        return False, None, False

    if unsuppressed_emergency_id is not None:
        return True, unsuppressed_emergency_id, False

    return False, None, unsuppressed_followup


def prepare_text_for_safety_scan(text: str, replace_zero_width_with_space: bool = True) -> str:
    """
    Normalizes Unicode and de-obfuscates whitespace/control characters before safety scanning.

    - Applies NFKC normalization to convert full-width / compatibility forms (e.g. 'ｉ ｗｉｌｌ ｋｉｌｌ ｍｙｓｅｌｆ') to standard ASCII.
    - Handles zero-width spaces and invisible format characters (\\u200b, \\u200c, \\u200d, \\u200e, \\u200f, \\ufeff, \\x00)
      either by converting to space (preserving word boundaries) or stripping them (reconstructing broken words).
    - Bounds text to SAFETY_SCAN_CEILING characters to prevent resource exhaustion attacks.
    - Collapses excessive whitespace.
    """
    if not text:
        return ""
    bounded = text[:SAFETY_SCAN_CEILING]
    normalized = unicodedata.normalize("NFKC", bounded)
    rep = " " if replace_zero_width_with_space else ""
    cleaned = re.sub(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\x00]+", rep, normalized)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


def pre_truncation_safety_scan(raw_text: str) -> Tuple[Optional[EmergencyTriggerID], bool, bool]:
    """
    Evaluates raw user input across its scannable length using clause-aware analysis.

    Distinguishes explicit immediate danger from non-imminent ideation / passive death wishes / self-harm disclosures,
    while preventing third-party or negated clauses from suppressing unrelated safety disclosures.
    Applies de-obfuscation against zero-width formatting and full-width Unicode evasion.

    Returns:
        Tuple of:
        - emergency_trigger_id: Optional[EmergencyTriggerID] (set ONLY for explicit immediate danger)
        - is_passive_death_wish: bool (True if passive wishes matched without immediate danger)
        - is_suicidal_ideation: bool (True if suicidal ideation/self-harm matched without immediate danger)
    """
    if not raw_text:
        return None, False, False

    # Primary pass: zero-width characters treated as word delimiters
    scannable_spaced = prepare_text_for_safety_scan(raw_text, replace_zero_width_with_space=True)
    is_emergency, trigger_id, is_followup = evaluate_clauses_for_safety(scannable_spaced)

    # Secondary de-obfuscation pass if needed: zero-width characters stripped inside words
    if not is_emergency and not is_followup and re.search(r"[\u200b\u200c\u200d\u200e\u200f\ufeff\x00]", raw_text[:SAFETY_SCAN_CEILING]):
        scannable_stripped = prepare_text_for_safety_scan(raw_text, replace_zero_width_with_space=False)
        is_emergency, trigger_id, is_followup = evaluate_clauses_for_safety(scannable_stripped)

    if is_emergency:
        return trigger_id, False, False

    if is_followup:
        is_passive = (
            bool(PASSIVE_DEATH_WISHES_REGEX.search(scannable_spaced))
            and not bool(SUICIDAL_IDEATION_REGEX.search(scannable_spaced) or SELF_HARM_DISCLOSURE_PATTERNS.search(scannable_spaced))
        )
        is_ideation = not is_passive
        return None, is_passive, is_ideation

    return None, False, False


def _is_emergency_number(digits_only: str) -> bool:
    """Returns True if the normalized digit string matches a known emergency number or short code."""
    return digits_only in EMERGENCY_WHITELIST


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """
    Detects and masks structured PII patterns while preserving emergency hotlines.

    Masks:
    - Email addresses -> [REDACTED_EMAIL]
    - URLs -> [REDACTED_URL]
    - US SSN -> [REDACTED_ID]
    - Indian Aadhaar -> [REDACTED_ID]
    - Phone numbers -> [REDACTED_PHONE] (except verified emergency numbers)

    Returns:
        Tuple of (sanitized_string, list_of_redaction_type_codes)
    """
    if not text:
        return "", []

    redactions: List[str] = []

    # 1. Mask URLs
    if URL_REGEX.search(text):
        text = URL_REGEX.sub("[REDACTED_URL]", text)
        redactions.append("URL_MASKED")

    # 2. Mask Emails
    if EMAIL_REGEX.search(text):
        text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
        redactions.append("EMAIL_MASKED")

    # 3. Mask National Identification Numbers (SSN & Aadhaar)
    if SSN_REGEX.search(text):
        text = SSN_REGEX.sub("[REDACTED_ID]", text)
        redactions.append("SSN_MASKED")

    if AADHAAR_REGEX.search(text):
        text = AADHAAR_REGEX.sub("[REDACTED_ID]", text)
        redactions.append("AADHAAR_MASKED")

    # 4. Mask Phone Numbers with Emergency Whitelist Protection
    def _phone_sub(match: re.Match[str]) -> str:
        matched_str = match.group(0)
        digits = re.sub(r"\D", "", matched_str)

        # Do not mask if the digits correspond to an emergency lifeline or short code
        if _is_emergency_number(digits):
            return matched_str

        # If digit count is small and not a phone pattern (e.g. quantities "50", years "2024"), preserve
        if len(digits) < 7:
            return matched_str

        redactions.append("PHONE_MASKED")
        return "[REDACTED_PHONE]"

    text = PHONE_REGEX.sub(_phone_sub, text)

    return text, list(set(redactions))


def validate_sanitized_messages(messages: Any) -> Tuple[bool, List[SanitizedMessage]]:
    """
    Validates that existing conversation messages strictly adhere to the SanitizedMessage contract.

    Contract:
    1. Must be a list.
    2. Each item must be a dictionary.
    3. Exactly two keys allowed: 'role' and 'content' (arbitrary or unsafe extra keys are prohibited).
    4. 'role' must be one of 'user', 'assistant', 'system'.
    5. 'content' must be a str.

    Returns:
        (True, validated_messages_list) if all items conform.
        (False, []) if validation fails.
    """
    if not isinstance(messages, list):
        return False, []

    validated: List[SanitizedMessage] = []
    for item in messages:
        if not isinstance(item, dict):
            return False, []
        if set(item.keys()) != {"role", "content"}:
            return False, []
        role = item.get("role")
        if role not in ("user", "assistant", "system"):
            return False, []
        content = item.get("content")
        if not isinstance(content, str):
            return False, []
        validated.append({"role": role, "content": content})

    return True, validated


# ============================================================================
# Main Node Function
# ============================================================================

def input_node(
    state: MindBridgeState,
    raw_user_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LangGraph Input Processing Node.

    Accepts raw user text ephemerally, performs pre-truncation safety scanning,
    validates conversation history integrity, normalizes text, scrubs structured PII,
    and returns state updates strictly conforming to MindBridgeState.

    Parameters:
        state: The current MindBridgeState dictionary.
        raw_user_text: The ephemeral, unmasked user utterance.
                       MUST NOT be stored in or read from persistent state.

    Returns:
        A dictionary of updates conforming strictly to MindBridgeState fields.
        Raw text, raw exception messages, or matched substrings are NEVER returned.
    """
    start_time = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    existing_trace: List[TraceEntry] = list(state.get("execution_trace", []))

    # ------------------------------------------------------------------------
    # 1. Existing Conversation History Validation
    # ------------------------------------------------------------------------
    existing_raw_messages = state.get("messages", [])
    history_valid, validated_existing_messages = validate_sanitized_messages(existing_raw_messages)

    if not history_valid:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry: TraceEntry = {
            "node": "input_node",
            "status": "error",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "INVALID_MESSAGE_HISTORY",
        }
        return {
            "sanitized_text": "",
            "messages": [],  # Untrusted history is quarantined and not copied
            "error_code": "STATE_TRANSITION_ERROR",
            "error_component": "input_node",
            "final_response": "A session state validation error occurred. Untrusted conversation history was quarantined.",
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # 2. Input Type & Null Validation
    # ------------------------------------------------------------------------
    if raw_user_text is None or not isinstance(raw_user_text, str):
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry = {
            "node": "input_node",
            "status": "error",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "INVALID_INPUT_TYPE",
        }
        return {
            "sanitized_text": "",
            "error_code": "INPUT_PARSING_FAILURE",
            "error_component": "input_node",
            "final_response": "I could not process that message format. Please share what is on your mind whenever you are ready.",
            "execution_trace": existing_trace + [trace_entry],
        }

    # ------------------------------------------------------------------------
    # 3. Empty / Whitespace-Only Validation
    # ------------------------------------------------------------------------
    stripped_input = raw_user_text.strip()
    if not stripped_input:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        trace_entry = {
            "node": "input_node",
            "status": "fallback",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "EMPTY_INPUT",
        }
        return {
            "sanitized_text": "",
            "error_code": "INPUT_PARSING_FAILURE",
            "error_component": "input_node",
            "final_response": "It looks like your message was empty. I am here with you and ready to support you whenever you are ready.",
            "execution_trace": existing_trace + [trace_entry],
        }

    try:
        # --------------------------------------------------------------------
        # 4. Pre-Truncation Emergency Safety Scan & Multi-Turn Context
        # --------------------------------------------------------------------
        # Evaluates conversational context, pending follow-ups, and active emergency states.
        session_id = state.get("session_id", "")
        from backend.safety.session_store import session_safety_store
        stored_safety_route = session_safety_store.get_session_safety_route(session_id) or state.get("overall_safety_route")

        from backend.safety.conversational_safety import evaluate_conversational_safety
        c_emergency, c_trigger_id, c_followup, c_followup_cleared = evaluate_conversational_safety(
            current_user_text=raw_user_text,
            history_messages=validated_existing_messages,
            stored_safety_route=stored_safety_route,
        )

        if c_emergency:
            is_emergency = True
            emergency_trigger_id = c_trigger_id
            is_followup_needed = False
        elif c_followup:
            is_emergency = False
            emergency_trigger_id = None
            is_followup_needed = True
        elif c_followup_cleared:
            is_emergency = False
            emergency_trigger_id = None
            is_followup_needed = False
        else:
            emergency_trigger_id, is_passive, is_ideation = pre_truncation_safety_scan(raw_user_text)
            is_emergency = emergency_trigger_id is not None
            is_followup_needed = not is_emergency and (is_passive or is_ideation)

        # --------------------------------------------------------------------
        # 5. Length Validation & Controlled Truncation
        # --------------------------------------------------------------------
        raw_length = len(raw_user_text)
        is_oversized = raw_length > MAX_INPUT_LENGTH
        is_hard_limit_exceeded = raw_length > HARD_MAX_INPUT_LENGTH

        # Truncate text for downstream processing if oversized, preserving safe boundary
        bounded_text = raw_user_text
        if is_oversized:
            bounded_text = raw_user_text[:MAX_INPUT_LENGTH]
            last_space = bounded_text.rfind(" ")
            if last_space > (MAX_INPUT_LENGTH - 100):
                bounded_text = bounded_text[:last_space]

        # --------------------------------------------------------------------
        # 6. Unicode & Whitespace Normalization
        # --------------------------------------------------------------------
        normalized_text = normalize_text(bounded_text)

        # --------------------------------------------------------------------
        # 7. Structured PII Masking
        # --------------------------------------------------------------------
        sanitized_text, redaction_codes = mask_pii(normalized_text)

        # --------------------------------------------------------------------
        # 8. Construct Sanitized Conversation Messages
        # --------------------------------------------------------------------
        new_message: SanitizedMessage = {
            "role": "user",
            "content": sanitized_text,
        }
        updated_messages = validated_existing_messages + [new_message]

        # --------------------------------------------------------------------
        # 9. Telemetry & Controlled State Assembly
        # --------------------------------------------------------------------
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Assign controlled metadata code (never raw input or user strings)
        if is_emergency:
            meta_code = "EMERGENCY_DETECTED"
        elif is_followup_needed:
            meta_code = "SAFETY_FOLLOWUP_FLAGGED"
        elif redaction_codes:
            meta_code = "PII_MASKED"
        elif is_hard_limit_exceeded:
            meta_code = "HARD_LIMIT_TRUNCATED"
        elif is_oversized:
            meta_code = "OVERSIZED_TRUNCATED"
        else:
            meta_code = "NORMAL_INPUT"

        trace_entry = {
            "node": "input_node",
            "status": "success",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": meta_code,
        }

        # Build state updates adhering strictly to MindBridgeState
        updates: Dict[str, Any] = {
            "sanitized_text": sanitized_text,
            "messages": updated_messages,
            "execution_trace": existing_trace + [trace_entry],
        }

        # Explicit immediate-danger disclosures activate emergency_bypass
        if is_emergency:
            updates["deterministic_danger_flag"] = True
            updates["emergency_trigger_id"] = emergency_trigger_id
            updates["overall_safety_route"] = "emergency_bypass"

        # Non-imminent ideation or passive death wishes initiate structured follow-up
        elif is_followup_needed:
            # Preliminary screening signal: initiates structured follow-up without
            # claiming confirmed endorsement or assigning clinical dimension values.
            updates["overall_safety_route"] = "pending_safety_follow_up"
            updates["phq9_followup_status"] = "in_progress"

        elif c_followup_cleared:
            updates["overall_safety_route"] = "standard_conversation"
            updates["phq9_followup_status"] = "completed"

        # If hard length limit was exceeded without emergency, record parsing notice and provide safe rejection response
        if is_hard_limit_exceeded and not is_emergency:
            updates["error_code"] = "INPUT_PARSING_FAILURE"
            updates["error_component"] = "input_node"
            if not is_followup_needed:
                updates["overall_safety_route"] = "ml_risk_unassessed"
                updates["final_response"] = (
                    "Your message exceeds the maximum allowed length (10,000 characters). "
                    "To ensure support is safe and accurate, please share a shorter message about what is on your mind."
                )
                updates["sanitized_text"] = ""

        return updates

    except Exception:
        # Fallback exception handling: strictly NO raw exception strings leaked
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Execute backup emergency scan on raw input in memory
        backup_trigger, _, _ = pre_truncation_safety_scan(raw_user_text)

        trace_entry = {
            "node": "input_node",
            "status": "error",
            "duration_ms": duration_ms,
            "timestamp_utc": timestamp_utc,
            "metadata_code": "EXCEPTION_HANDLED",
        }

        error_updates: Dict[str, Any] = {
            "sanitized_text": "",
            "error_code": "INPUT_PARSING_FAILURE",
            "error_component": "input_node",
            "execution_trace": existing_trace + [trace_entry],
        }

        pre_existing_emergency = (
            state.get("deterministic_danger_flag") is True
            or state.get("overall_safety_route") == "emergency_bypass"
        )
        if backup_trigger is not None:
            error_updates["deterministic_danger_flag"] = True
            error_updates["emergency_trigger_id"] = backup_trigger
            error_updates["overall_safety_route"] = "emergency_bypass"
        elif pre_existing_emergency:
            error_updates["deterministic_danger_flag"] = True
            error_updates["emergency_trigger_id"] = state.get("emergency_trigger_id") or "UNKNOWN_EMERGENCY_TRIGGER"
            error_updates["overall_safety_route"] = "emergency_bypass"
        else:
            error_updates["final_response"] = (
                "I encountered an unexpected issue processing your message. "
                "I am here with you and ready to listen whenever you are ready."
            )

        return error_updates
