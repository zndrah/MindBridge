"""
MindBridge 2.0 - Deterministic Query Understanding & Intent Mapping
===================================================================

Translates informal, colloquial user expressions into structured retrieval concepts
and focused, sanitized search queries without modifying the original user message
or changing its safety profile.

Authoritative Governance & Safety Rules:
1. PII & Privacy Integrity: Raw user messages are never stored, logged, or exposed.
2. Safety Precedence: Safety detection always precedes query understanding.
   Explicit self-harm or emergency disclosures are NEVER rewritten into ordinary queries.
3. Deterministic & Testable: Uses pure rule-based concept matching without external
   paid APIs or stochastic cloud LLM dependencies.
4. Non-Diagnostic Framing: Maps user utterances to psychoeducational topic concepts,
   never fabricating clinical diagnoses, psychiatric labels, or medical advice.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.orchestrator.input_node import EMERGENCY_TRIGGERS
from backend.orchestrator.state import DialogGoal, JurisdictionCode


@dataclass(frozen=True)
class QueryUnderstandingResult:
    """
    Immutable structured container representing the output of query understanding.
    """
    original_text: str
    identified_concepts: List[str]
    dialog_goal: DialogGoal
    retrieval_queries: List[str]
    is_crisis_related: bool = False
    is_out_of_domain: bool = False
    requires_restricted_access: bool = False


# ============================================================================
# Concept Matching Rules & Canonical Retrieval Queries
# ============================================================================

CONCEPT_PATTERNS: List[Dict[str, Any]] = [
    # 1. Low Motivation & Depressive Inertia
    {
        "id": "depressive_inertia",
        "concepts": ["low_motivation", "depressive_inertia", "behavioral_activation", "micro_commitments"],
        "goal": "cognitive_reframing",
        "patterns": [
            re.compile(r"\b(empty|numb|hopeless|pointless|meaningless|lethargic|lethargy)\b", re.IGNORECASE),
            re.compile(r"\b(no|zero|lack\s+of|lost\s+all)\s+(energy|motivation|drive|will)\b", re.IGNORECASE),
            re.compile(r"\b(stay\s+in|lying\s+in|get\s+out\s+of|stuck\s+in)\s+bed\b", re.IGNORECASE),
            re.compile(r"\b(bed\s+all\s+day|do\s+nothing\s+all\s+day|staying\s+in\s+bed)\b", re.IGNORECASE),
            re.compile(r"\bdepress(ed|ion|ive)?\s*(inertia)?\b", re.IGNORECASE),
            re.compile(r"\b(behavioral\s+activation|small\s+tasks?|micro\s*commitments?|small\s+steps?|action\s+precedes\s+motivation)\b", re.IGNORECASE),
        ],
        "query": "Behavioral activation and micro-steps for depressive inertia and low energy in bed",
    },

    # 2. Acute Panic & Somatic De-escalation
    {
        "id": "acute_panic",
        "concepts": ["acute_panic", "rapid_deescalation", "somatic_respiration", "parasympathetic_reset"],
        "goal": "somatic_grounding",
        "patterns": [
            re.compile(r"\b(panic\s+attack|panicking|panicked|feel(ing)?\s+panic|intense\s+anxiety|freaking\s+out)\b", re.IGNORECASE),
            re.compile(r"\b(heart\s+(?:is\s+)?(pounding|racing|beating\s+fast)|chest\s+(?:is\s+)?(tight|pain|pressure))\b", re.IGNORECASE),
            re.compile(r"\b(can('?t|not)\s+breathe|shortness\s+of\s+breath|hyperventilat(ing|ion))\b", re.IGNORECASE),
            re.compile(r"\b(calm\s+down\s+fast|overwhelmed\s+right\s+now|immediate\s+physical\s+de-?escalation|stop\s+panic)\b", re.IGNORECASE),
        ],
        "query": "Breathing techniques and rapid somatic de-escalation for acute panic attack",
    },

    # 3. Box Breathing & Tactical Breath
    {
        "id": "box_breathing",
        "concepts": ["box_breathing", "controlled_respiration", "parasympathetic_reset"],
        "goal": "somatic_grounding",
        "patterns": [
            re.compile(r"\b(box\s+breath(ing)?|4-4-4-4|4\s*4\s*4\s*4|square\s+breath(ing)?|tactical\s+breath(ing)?)\b", re.IGNORECASE),
        ],
        "query": "Box breathing 4-4-4-4 parasympathetic reset protocol",
    },

    # 4. Physiological Sigh & Cyclic Respiration
    {
        "id": "physiological_sigh",
        "concepts": ["physiological_sigh", "cyclic_sighing", "autonomic_regulation"],
        "goal": "somatic_grounding",
        "patterns": [
            re.compile(r"\b(physiological\s+sigh|cyclic\s+sigh(ing)?|double\s+inhale|cyclic\s+respiration)\b", re.IGNORECASE),
        ],
        "query": "The physiological sigh controlled cyclic respiration technique",
    },

    # 5. Sensory Grounding & 5-4-3-2-1 / PMR
    {
        "id": "grounding_sensory",
        "concepts": ["sensory_grounding", "5_4_3_2_1", "dissociation", "pmr_relaxation"],
        "goal": "somatic_grounding",
        "patterns": [
            re.compile(r"\b(5-?4-?3-?2-?1|54321|five\s+senses|sensory\s+(grounding|orientation))\b", re.IGNORECASE),
            re.compile(r"\b(dissociat(ing|ion|ed)?|unreal|detached|foggy|depersonaliz(ing|ation))\b", re.IGNORECASE),
            re.compile(r"\b(progressive\s+muscle\s+relaxation|pmr|muscle\s+tension|release\s+physical\s+tension)\b", re.IGNORECASE),
        ],
        "query": "5-4-3-2-1 sensory orientation and progressive muscle relaxation PMR grounding protocol",
    },

    # 6. Sleep Difficulties & CBT-I Stimulus Control
    {
        "id": "sleep_insomnia",
        "concepts": ["insomnia", "stimulus_control", "cbt_i", "bed_conditioning", "sleep_hygiene"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(insomnia|can('?t|not)\s+sleep|trouble\s+sleeping|unable\s+to\s+sleep)\b", re.IGNORECASE),
            re.compile(r"\b(toss(ing)?\s+and\s+turn(ing)?|lying\s+awake|wake\s+up\s+(at|in)\s+night|middle\s+of\s+the\s+night)\b", re.IGNORECASE),
            re.compile(r"\b(20\s*min(ute)?\s+rule|stimulus\s+control|cbt-?i|bed\s+(anxiety|association|conditioning))\b", re.IGNORECASE),
            re.compile(r"\b(circadian|sleep\s+hygiene|sleep\s+schedule|sleep\s+wake\s+cycles?)\b", re.IGNORECASE),
        ],
        "query": "Stimulus control therapy for insomnia CBT-I and circadian sleep hygiene",
    },

    # 7. Competitive Exam Stress & Parental Pressure (India)
    {
        "id": "exam_anxiety",
        "concepts": ["competitive_exam_stress", "academic_pressure", "familial_stakes", "exam_anxiety"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(jee|neet|upsc|cat|gate|entrance\s+exam|competitive\s+exam)\b", re.IGNORECASE),
            re.compile(r"\b(mock\s+test|kota|coaching\s+class(es)?|failing\s+exam|engineering\s+finals)\b", re.IGNORECASE),
            re.compile(r"\b(parent(al|s)?\s+(pressure|savings|expectations)|family\s+disappointment|crushing\s+(academic\s+)?pressure)\b", re.IGNORECASE),
        ],
        "query": "Navigating competitive exam anxiety familial stakes and panic in India",
    },

    # 8. General Academic Stress & Exam Anxiety
    {
        "id": "academic_exam_stress",
        "concepts": ["academic_stress", "exam_anxiety", "study_planning", "coping_strategies"],
        "goal": "general",
        "patterns": [
            re.compile(
                r"\b(stress(?:ed)?|anxious|anxiety|panic|worr(?:y|ied)|dread(?:ing)?|fear|overwhelm(?:ed)?)\s+"
                r"(?:about|over|with|from|around|due\s+to|because\s+of)\s+(?:(?:my|the|these|our)\s+)?"
                r"(?:(?:university|college|school|academic|semester|term|upcoming|final)\s+)?"
                r"(?:exams?|finals?|tests?|midterms?|stud(?:ies|ying)|school|coursework|academics?)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:exam|test|finals?|midterms?)\s+(?:anxiety|stress|panic|pressure|dread|fear|overwhelm)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:academic|study|coursework|school)\s+(?:stress|anxiety|pressure|overwhelm|burnout|fatigue)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:overwhelmed|stressed)\s+(?:by|with)\s+(?:(?:my|the|these|our)\s+)?"
                r"(?:(?:university|college|school|academic|semester|term|upcoming)\s+)?"
                r"(?:stud(?:ies|ying)|coursework|academics?|school\s*work|exams?)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bfeeling\s+stressed\s+about\s+(?:(?:my|the|these|our)\s+)?"
                r"(?:(?:university|college|school|academic)\s+)?"
                r"(?:exams?|stud(?:ies|ying)|tests?|academics?)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:so|really|very|extremely|feeling)\s+(?:stressed|anxious|overwhelmed)\s+(?:about|over|for|with)\s+"
                r"(?:(?:my|the|these|our)\s+)?"
                r"(?:(?:university|college|school|academic)\s+)?"
                r"(?:exams?|finals?|tests?)\b",
                re.IGNORECASE,
            ),
        ],
        "query": "Managing academic stress and exam anxiety through practical study planning manageable breaks and realistic expectations",
    },

    # 8. Cognitive Restructuring & Catastrophizing (Thought Court)
    {
        "id": "catastrophizing",
        "concepts": ["catastrophizing", "thought_court", "cognitive_restructuring", "evidence_testing"],
        "goal": "cognitive_reframing",
        "patterns": [
            re.compile(r"\b(catastrophiz(ing|e)?|catastrophic|worst\s*case\s*scenario|worst\s+possible\s+scenario)\b", re.IGNORECASE),
            re.compile(r"\b(thought\s+court|cognitive\s+(distortion|reframing|restructuring))\b", re.IGNORECASE),
            re.compile(r"\b(all\s+or\s+nothing|black\s+and\s+white\s+thinking|mind\s+reading|fortune\s+telling)\b", re.IGNORECASE),
            re.compile(r"\b(challenging\s+thoughts?|negative\s+thought\s+loop|assuming\s+the\s+worst)\b", re.IGNORECASE),
        ],
        "query": "Cognitive restructuring for catastrophic loops and Thought Court evidence testing",
    },

    # 9. Worry Postponement & Racing Thoughts at Night
    {
        "id": "worry_log",
        "concepts": ["worry_postponement", "racing_thoughts", "nighttime_rumination", "cbt_worry_log"],
        "goal": "cognitive_reframing",
        "patterns": [
            re.compile(r"\b(racing\s+thoughts?|racing\s+mind|thoughts\s+(racing|spiraling|won'?t\s+stop))\b", re.IGNORECASE),
            re.compile(r"\b(worry\s+(log|postponement|journal)|write\s+down\s+worries|writing\s+down\s+worries)\b", re.IGNORECASE),
            re.compile(r"\b(ruminat(ing|ion)|overthinking\s+at\s+night|ruminating\s+all\s+night)\b", re.IGNORECASE),
        ],
        "query": "De-escalating racing thoughts at night with CBT worry postponement log",
    },

    # 10. Values Clarification & Micro-Commitments (ACT)
    {
        "id": "values_act",
        "concepts": ["values_clarification", "act_therapy", "micro_commitments"],
        "goal": "cognitive_reframing",
        "patterns": [
            re.compile(r"\b(act|acceptance\s+and\s+commitment|values\s+clarification|core\s+values)\b", re.IGNORECASE),
            re.compile(r"\b(what\s+matters\s+to\s+me|micro\s*commitments?|values\s+and\s+actions?)\b", re.IGNORECASE),
        ],
        "query": "Values clarification and micro-commitments in Acceptance and Commitment Therapy ACT",
    },

    # 11. Workplace Burnout & Cultural Family Boundaries (India)
    {
        "id": "tech_burnout_culture",
        "concepts": ["workplace_burnout", "off_clock_autonomy", "cultural_family_boundaries"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(tech\s+burnout|workplace\s+burnout|bengaluru|bangalore|it\s+corridors?|commute\s+recovery)\b", re.IGNORECASE),
            re.compile(r"\b(off-?clock\s+autonomy|toxic\s+workplace)\b", re.IGNORECASE),
            re.compile(r"\b(joint\s+family|family\s+boundaries|respect\s+without\s+self-?silencing)\b", re.IGNORECASE),
        ],
        "query": "Burnout in India's tech corridors off-clock autonomy and cultural family boundaries",
    },

    # 12. Loneliness & Social Isolation
    {
        "id": "loneliness_isolation",
        "concepts": ["loneliness", "social_isolation", "connection_building", "belonging"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(lonel(?:y|iness)|feel(?:ing)?\s+alone|isolated|isolation|no\s+friends|feel\s+disconnected|nobody\s+to\s+talk\s+to|left\s+out|alienat(?:ed|ion))\b", re.IGNORECASE),
            re.compile(r"\b(making\s+friends|social\s+disconnection|struggling\s+to\s+connect|longing\s+for\s+connection|feeling\s+so\s+alone)\b", re.IGNORECASE),
        ],
        "query": "Coping with loneliness and social isolation building meaningful connections",
    },

    # 13. Family Conflict & Parental Dynamics
    {
        "id": "family_conflict",
        "concepts": ["family_conflict", "parental_conflict", "communication_boundaries", "family_dynamics"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(family\s+(?:conflict|fight|arguments?|drama|issues?|problems?|tension)|fighting\s+with\s+(?:my\s+)?(?:family|parents|mom|mother|dad|father|sister|brother))\b", re.IGNORECASE),
            re.compile(r"\b(parents?\s+(?:don'?t\s+understand|criticiz(?:e|ing)|fighting|disagree(?:ing)?|yell(?:ing)?|arguing))\b", re.IGNORECASE),
            re.compile(r"\b(mother|father|parents?|in-laws?)\s+conflict\b", re.IGNORECASE),
        ],
        "query": "Navigating family conflict communication breakdowns and parental boundaries",
    },

    # 14. Relationship Problems & Interpersonal Conflict
    {
        "id": "relationship_conflict",
        "concepts": ["relationship_problems", "partner_conflict", "communication_issues", "trust_intimacy"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(relationship\s+(?:problems?|issues?|struggles?|conflict|troubles?)|partner\s+conflict)\b", re.IGNORECASE),
            re.compile(r"\b(fighting\s+with\s+(?:my\s+)?(?:partner|boyfriend|girlfriend|husband|wife|spouse))\b", re.IGNORECASE),
            re.compile(r"\b(communication\s+issues?\s+with\s+(?:my\s+)?(?:partner|spouse|boyfriend|girlfriend)|trust\s+issues?\s+in\s+relationship)\b", re.IGNORECASE),
            re.compile(r"\b(breakup|breaking\s+up|divorce|growing\s+apart|drifting\s+apart)\b", re.IGNORECASE),
        ],
        "query": "Resolving relationship problems partner communication and rebuilding trust",
    },

    # 15. General Depression-Related Concerns & Sadness
    {
        "id": "depression_concerns",
        "concepts": ["depression_concerns", "persistent_sadness", "emotional_heaviness", "depressive_symptoms"],
        "goal": "general",
        "patterns": [
            re.compile(r"\b(feel(?:ing)?\s+(?:depressed|so\s+sad|down|blue|gloomy)|persistent\s+sadness|heavy\s+heart|crying\s+all\s+the\s+time)\b", re.IGNORECASE),
            re.compile(r"\b(struggling\s+with\s+depression|feeling\s+hopeless\s+and\s+sad|deep\s+sadness|depression\s+symptoms?)\b", re.IGNORECASE),
        ],
        "query": "Managing depression persistent sadness emotional heaviness and supportive coping",
    },

    # 16. Self-Esteem & Insecurity
    {
        "id": "self_esteem",
        "concepts": ["self_esteem", "self_worth", "self_criticism", "feeling_inadequate"],
        "goal": "cognitive_reframing",
        "patterns": [
            re.compile(r"\b(self-?esteem|low\s+self-?esteem|self-?worth|feel(?:ing)?\s+(?:worthless|inadequate|inferior|not\s+good\s+enough))\b", re.IGNORECASE),
            re.compile(r"\b(hate\s+myself|self-?doubt|insecurity|imposter\s+syndrome|comparing\s+myself\s+to\s+others)\b", re.IGNORECASE),
            re.compile(r"\b(lack\s+of\s+confidence|building\s+confidence|feeling\s+like\s+a\s+failure)\b", re.IGNORECASE),
        ],
        "query": "Building self-esteem overcoming self-doubt and feeling inadequate",
    },

    # 17. Clinical Screening Rubrics (PHQ-9, GAD-7, PSS-10, NIMHANS)
    {
        "id": "screening_rubrics",
        "concepts": ["clinical_screening_rubric", "psychometric_scale"],
        "goal": "screening",
        "patterns": [
            re.compile(r"\b(phq-?9|gad-?7|pss-?10|nimhans)\b", re.IGNORECASE),
            re.compile(r"\b(depression\s+scale|anxiety\s+scale|stress\s+scale|triage\s+framework)\b", re.IGNORECASE),
            re.compile(r"\b(cutoff\s+scores?|severity\s+tiers?|scoring\s+items?|baseline\s+threshold\s+rubric)\b", re.IGNORECASE),
        ],
        "query": "Psychometric screening rubric overview scoring criteria and severity tiers",
        "restricted": True,
    },
]

# Explicit Out-of-Domain indicators (non-mental health)
OOD_PATTERNS = [
    re.compile(r"\b(recipe|bake|baking|cookies?|butter|cake|roast|soup)\b", re.IGNORECASE),
    re.compile(r"\b(python|lambda|dictionary|function|code|programming|algorithm)\b", re.IGNORECASE),
    re.compile(r"\b(roman\s+empire|julius\s+caesar|emperor|ancient\s+rome|history)\b", re.IGNORECASE),
    re.compile(r"\b(stock\s+price|market\s+capitalization|shares?|apple|microsoft|nasdaq)\b", re.IGNORECASE),
    re.compile(r"\b(combustion\s+engine|four\s+stroke|horse\s*power|transmission)\b", re.IGNORECASE),
    re.compile(r"\b(capital\s+(?:city|of)|population\s+of|australia\s+capital|geography)\b", re.IGNORECASE),
    re.compile(r"\b(oil\s+filter|spark\s+plugs?|honda\s+civic|mechanic)\b", re.IGNORECASE),
]


def understand_query(
    sanitized_text: str,
    overall_safety_route: Optional[str] = None,
    deterministic_danger_flag: bool = False,
) -> QueryUnderstandingResult:
    """
    Deterministic query understanding and planning transformation.

    Analyzes sanitized user text to extract mental-health concepts, determine
    conversational dialog goal, and synthesize targeted retrieval queries.

    CRITICAL SAFETY INVARIANT:
    If emergency indicators are active, or if emergency patterns are detected,
    query understanding REFUSES to rewrite or convert crisis disclosures into
    ordinary mental-health search queries. It returns an empty query list.
    """
    cleaned_text = (sanitized_text or "").strip()

    # Safety Gate 1: Check existing safety flags
    if overall_safety_route == "emergency_bypass" or deterministic_danger_flag is True:
        return QueryUnderstandingResult(
            original_text=cleaned_text,
            identified_concepts=[],
            dialog_goal="general",
            retrieval_queries=[],
            is_crisis_related=True,
            is_out_of_domain=False,
            requires_restricted_access=False,
        )

    # Safety Gate 2: Defense-in-depth check for emergency triggers in text
    for trigger_id, pattern in EMERGENCY_TRIGGERS:
        if pattern.search(cleaned_text):
            return QueryUnderstandingResult(
                original_text=cleaned_text,
                identified_concepts=[],
                dialog_goal="general",
                retrieval_queries=[],
                is_crisis_related=True,
                is_out_of_domain=False,
                requires_restricted_access=False,
            )

    # Empty or whitespace input
    if not cleaned_text:
        return QueryUnderstandingResult(
            original_text=cleaned_text,
            identified_concepts=[],
            dialog_goal="general",
            retrieval_queries=[],
            is_crisis_related=False,
            is_out_of_domain=True,
            requires_restricted_access=False,
        )

    matched_concepts: List[str] = []
    generated_queries: List[str] = []
    determined_goals: List[DialogGoal] = []
    requires_restricted = False

    # Evaluate concept pattern matchers
    for rule in CONCEPT_PATTERNS:
        match_found = False
        for pattern in rule["patterns"]:
            if pattern.search(cleaned_text):
                match_found = True
                break

        if match_found:
            for c in rule["concepts"]:
                if c not in matched_concepts:
                    matched_concepts.append(c)
            if rule["query"] not in generated_queries:
                generated_queries.append(rule["query"])
            determined_goals.append(rule["goal"])
            if rule.get("restricted", False):
                requires_restricted = True

    # If mental-health concepts matched:
    if matched_concepts:
        # Determine primary dialog goal: prioritize specific goals over general
        primary_goal: DialogGoal = "general"
        for goal_candidate in ("screening", "somatic_grounding", "cognitive_reframing", "general"):
            if goal_candidate in determined_goals:
                primary_goal = goal_candidate  # type: ignore
                break

        if requires_restricted:
            for q in list(generated_queries):
                if "screening rubric" in q.lower() or "psychometric" in q.lower() or "triage" in q.lower():
                    generated_queries.remove(q)
                    generated_queries.insert(0, q)
                    break

        return QueryUnderstandingResult(
            original_text=cleaned_text,
            identified_concepts=matched_concepts,
            dialog_goal=primary_goal,
            retrieval_queries=generated_queries[:3],
            is_crisis_related=False,
            is_out_of_domain=False,
            requires_restricted_access=requires_restricted,
        )

    # Check for obvious Out-of-Domain topics
    for ood_pat in OOD_PATTERNS:
        if ood_pat.search(cleaned_text):
            return QueryUnderstandingResult(
                original_text=cleaned_text,
                identified_concepts=[],
                dialog_goal="general",
                retrieval_queries=[],
                is_crisis_related=False,
                is_out_of_domain=True,
                requires_restricted_access=False,
            )

    # General Fallback: Uncategorized or colloquial supportive query
    # Passes sanitized text directly as a single query, or empty if too generic
    return QueryUnderstandingResult(
        original_text=cleaned_text,
        identified_concepts=["general_support"],
        dialog_goal="general",
        retrieval_queries=[cleaned_text],
        is_crisis_related=False,
        is_out_of_domain=False,
        requires_restricted_access=False,
    )
