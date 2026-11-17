"""
MindBridge 2.0 - Knowledge Retrieval Governance Registry
=========================================================

Authoritative governance definitions, classification tiers, and mandatory
guardrails for all documents in `data/knowledge_base/`.

Grounded in:
- docs/PHASE3_KNOWLEDGE_BASE_AUDIT.md §2, §3
- data/knowledge_base/manifest.json
- docs/ARCHITECTURE.md §5

Three retrieval classification tiers:
1. `approved` (13 documents): Eligible for open informational retrieval and general
   psychoeducational / somatic conversational grounding.
2. `restricted` (7 documents): Eligible for ingestion, but restricted from open-ended
   conversational retrieval without mandatory guardrails, non-diagnostic disclaimers,
   or agent-level supervision.
3. `excluded_pending_review` (1 document): Unverified alternative therapies lacking
   neurobiological consensus. STRICTLY BLOCKED from index ingestion.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from backend.orchestrator.state import JurisdictionCode

RetrievalTier = Literal["approved", "restricted", "excluded_pending_review"]

ClinicalSafetyLevel = Literal[
    "open_informational",
    "screening_gated",
    "phq9_safety_protocol",
    "boundary_supervision",
    "legal_informational",
    "somatic_caution",
    "theoretical_model",
    "blocked_from_retrieval"
]


@dataclass(frozen=True)
class DocumentGovernance:
    """Canonical governance record for an audited knowledge base document."""
    document_id: str
    relative_path: str
    title: str
    category: str
    jurisdiction: JurisdictionCode
    verification_status: Literal["independently_verified", "partially_verified", "unverified"]
    retrieval_tier: RetrievalTier
    clinical_safety_level: ClinicalSafetyLevel
    guardrails: List[str] = field(default_factory=list)
    disclaimer: str = ""
    rejection_reason: Optional[str] = None


# Canonical Document Governance Registry (21 documents audited in Phase 3)
DOCUMENT_GOVERNANCE_REGISTRY: Dict[str, DocumentGovernance] = {
    # ------------------------------------------------------------------------
    # Category: clinical_rubrics (4 documents, all RESTRICTED)
    # ------------------------------------------------------------------------
    "clinical_rubrics/01_gad7_anxiety_screening_overview.md": DocumentGovernance(
        document_id="clinical_rubrics:01_gad7_anxiety_screening_overview",
        relative_path="clinical_rubrics/01_gad7_anxiety_screening_overview.md",
        title="GAD-7 (Generalized Anxiety Disorder 7-Item Scale) Overview",
        category="clinical_rubrics",
        jurisdiction="GL",
        verification_status="independently_verified",
        retrieval_tier="restricted",
        clinical_safety_level="screening_gated",
        guardrails=[
            "MANDATORY: You are providing informational screening details. You cannot and must not evaluate user responses as a clinical diagnosis.",
            "Informational screening only; never diagnostic confirmation."
        ],
        disclaimer="The GAD-7 is an informational screening rubric, NOT a diagnostic instrument. Elevated scores warrant evaluation by a licensed healthcare professional."
    ),
    "clinical_rubrics/02_phq9_mood_screening_overview.md": DocumentGovernance(
        document_id="clinical_rubrics:02_phq9_mood_screening_overview",
        relative_path="clinical_rubrics/02_phq9_mood_screening_overview.md",
        title="PHQ-9 (Patient Health Questionnaire 9-Item Scale) Overview",
        category="clinical_rubrics",
        jurisdiction="GL",
        verification_status="independently_verified",
        retrieval_tier="restricted",
        clinical_safety_level="phq9_safety_protocol",
        guardrails=[
            "CRITICAL: The PHQ-9 is an informational screening rubric, NOT a diagnostic confirmation or standalone suicide-risk assessment.",
            "Item 9 endorsement requires immediate direct safety follow-up to stratify risk across the 7 clinical dimensions.",
            "Imminent peril (active ideation + intent/plan/means) triggers deterministic emergency bypass (hotlines.json); ambiguous/unclear risk prioritizes supportive human assistance."
        ],
        disclaimer="CRITICAL: The PHQ-9 is an informational psychometric screening rubric, NOT a diagnostic confirmation or standalone suicide-risk assessment."
    ),
    "clinical_rubrics/03_pss10_stress_scale_overview.md": DocumentGovernance(
        document_id="clinical_rubrics:03_pss10_stress_scale_overview",
        relative_path="clinical_rubrics/03_pss10_stress_scale_overview.md",
        title="PSS-10 (Perceived Stress Scale 10-Item) Overview",
        category="clinical_rubrics",
        jurisdiction="GL",
        verification_status="independently_verified",
        retrieval_tier="restricted",
        clinical_safety_level="screening_gated",
        guardrails=[
            "Restricted to informational queries to prevent misrepresenting stress as clinical pathology.",
            "Informational screening only; never diagnostic confirmation."
        ],
        disclaimer="The PSS-10 measures perceived non-specific stress over the preceding month and does not assess clinical depression or anxiety disorders."
    ),
    "clinical_rubrics/04_nimhans_triage_boundaries.md": DocumentGovernance(
        document_id="clinical_rubrics:04_nimhans_triage_boundaries",
        relative_path="clinical_rubrics/04_nimhans_triage_boundaries.md",
        title="NIMHANS Digital Triage Framework & Clinical Red Lines",
        category="clinical_rubrics",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="restricted",
        clinical_safety_level="boundary_supervision",
        guardrails=[
            "Reserved for agent supervisor prompts and fallback routing logic; excluded from general empathetic dialogue retrieval.",
            "Do not quote raw operational triage formulas directly to distressed users."
        ],
        disclaimer="Clinical red lines and triage boundaries established in collaboration with NIMHANS Bengaluru and MoHFW guidelines."
    ),

    # ------------------------------------------------------------------------
    # Category: cognitive_reframing (5 documents: 4 APPROVED, 1 RESTRICTED)
    # ------------------------------------------------------------------------
    "cognitive_reframing/01_thought_court_catastrophizing.md": DocumentGovernance(
        document_id="cognitive_reframing:01_thought_court_catastrophizing",
        relative_path="cognitive_reframing/01_thought_court_catastrophizing.md",
        title="Cognitive Restructuring for Catastrophic Loops (Thought Court)",
        category="cognitive_reframing",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Non-clinical cognitive restructuring educational tool for challenging catastrophic thoughts."
    ),
    "cognitive_reframing/02_cbt_worry_postponement.md": DocumentGovernance(
        document_id="cognitive_reframing:02_cbt_worry_postponement",
        relative_path="cognitive_reframing/02_cbt_worry_postponement.md",
        title="De-escalating Racing Thoughts at Night: The CBT Worry Log",
        category="cognitive_reframing",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Behavioral worry postponement exercise for nocturnal rumination and overthinking."
    ),
    "cognitive_reframing/03_behavioral_activation_inertia.md": DocumentGovernance(
        document_id="cognitive_reframing:03_behavioral_activation_inertia",
        relative_path="cognitive_reframing/03_behavioral_activation_inertia.md",
        title="Behavioral Activation for Depressive Inertia",
        category="cognitive_reframing",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Structured micro-tasking to interrupt avoidance and depressive inertia."
    ),
    "cognitive_reframing/04_polyvagal_social_engagement.md": DocumentGovernance(
        document_id="cognitive_reframing:04_polyvagal_social_engagement",
        relative_path="cognitive_reframing/04_polyvagal_social_engagement.md",
        title="The Social Engagement Framing: Theoretical Model for Emotional Grounding",
        category="cognitive_reframing",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="restricted",
        clinical_safety_level="theoretical_model",
        guardrails=[
            "Present concept strictly as a reflective psychoeducational model or metaphor, not established neuroanatomical fact."
        ],
        disclaimer="Metaphorical psychoeducational model for reflective somatic awareness."
    ),
    "cognitive_reframing/05_values_clarification_act.md": DocumentGovernance(
        document_id="cognitive_reframing:05_values_clarification_act",
        relative_path="cognitive_reframing/05_values_clarification_act.md",
        title="Values Clarification & Micro-Commitments",
        category="cognitive_reframing",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="ACT-informed values alignment framework for decision fatigue and burnout."
    ),

    # ------------------------------------------------------------------------
    # Category: legal_and_privacy (1 document, RESTRICTED)
    # ------------------------------------------------------------------------
    "legal_and_privacy/dpdp_and_mhca_statutory_reference.md": DocumentGovernance(
        document_id="legal_and_privacy:dpdp_and_mhca_statutory_reference",
        relative_path="legal_and_privacy/dpdp_and_mhca_statutory_reference.md",
        title="DPDP Act 2023 & MHCA 2017: Statutory Baseline, UI Claims Analysis & Engineering Governance",
        category="legal_and_privacy",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="restricted",
        clinical_safety_level="legal_informational",
        guardrails=[
            "Gated behind explicit legal/privacy user queries.",
            "Mandatory disclaimer: 'This is an architectural and regulatory summary, not formal legal counsel.'"
        ],
        disclaimer="Architectural summary of statutory standards under DPDP Act 2023 and MHCA 2017. Does not constitute legal advice."
    ),

    # ------------------------------------------------------------------------
    # Category: psychoeducation (6 documents, all APPROVED)
    # ------------------------------------------------------------------------
    "psychoeducation/01_exam_anxiety_resilience_india.md": DocumentGovernance(
        document_id="psychoeducation:01_exam_anxiety_resilience_india",
        relative_path="psychoeducation/01_exam_anxiety_resilience_india.md",
        title="Navigating Competitive Exam Anxiety: Familial Stakes & Panic",
        category="psychoeducation",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Culturally sensitive psychoeducation for Indian competitive exam pressures."
    ),
    "psychoeducation/02_tech_workplace_burnout.md": DocumentGovernance(
        document_id="psychoeducation:02_tech_workplace_burnout",
        relative_path="psychoeducation/02_tech_workplace_burnout.md",
        title="Burnout in India's Tech Corridors: Off-Clock Autonomy & Commute Recovery",
        category="psychoeducation",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Workplace boundaries and off-clock autonomy psychoeducation."
    ),
    "psychoeducation/03_cultural_family_boundaries.md": DocumentGovernance(
        document_id="psychoeducation:03_cultural_family_boundaries",
        relative_path="psychoeducation/03_cultural_family_boundaries.md",
        title="Cultural Dimensions of Family Boundaries: Respect Without Self-Silencing",
        category="psychoeducation",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Interpersonal communication guidance for collectivist familial structures."
    ),
    "psychoeducation/04_stimulus_control_insomnia_cbti.md": DocumentGovernance(
        document_id="psychoeducation:04_stimulus_control_insomnia_cbti",
        relative_path="psychoeducation/04_stimulus_control_insomnia_cbti",
        title="Stimulus Control Therapy for Insomnia (CBT-I)",
        category="psychoeducation",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Standard behavioral sleep conditioning protocol based on the Bootzin stimulus control model."
    ),
    "psychoeducation/05_circadian_sleep_hygiene.md": DocumentGovernance(
        document_id="psychoeducation:05_circadian_sleep_hygiene",
        relative_path="psychoeducation/05_circadian_sleep_hygiene.md",
        title="Circadian Sleep Hygiene Checklist",
        category="psychoeducation",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Behavioral and environmental sleep hygiene recommendations."
    ),
    "psychoeducation/06_academic_exam_stress_grounding.md": DocumentGovernance(
        document_id="psychoeducation:06_academic_exam_stress_grounding",
        relative_path="psychoeducation/06_academic_exam_stress_grounding.md",
        title="Managing Academic Stress and Exam Anxiety: Practical Study Planning & Coping",
        category="psychoeducation",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Non-clinical educational guidance for academic and exam stress management."
    ),

    # ------------------------------------------------------------------------
    # Category: somatic_regulation (6 documents: 4 APPROVED, 1 RESTRICTED, 1 EXCLUDED)
    # ------------------------------------------------------------------------
    "somatic_regulation/01_box_breathing_4_4_4_4.md": DocumentGovernance(
        document_id="somatic_regulation:01_box_breathing_4_4_4_4",
        relative_path="somatic_regulation/01_box_breathing_4_4_4_4.md",
        title="Box Breathing (4-4-4-4 Parasympathetic Reset)",
        category="somatic_regulation",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Non-clinical supportive exercise for acute stress de-escalation."
    ),
    "somatic_regulation/02_pranayama_4_7_8_vagus.md": DocumentGovernance(
        document_id="somatic_regulation:02_pranayama_4_7_8_vagus",
        relative_path="somatic_regulation/02_pranayama_4_7_8_vagus.md",
        title="4-7-8 Paced Breathing & Pranayama: Optional Somatic Coping Technique",
        category="somatic_regulation",
        jurisdiction="IN",
        verification_status="partially_verified",
        retrieval_tier="restricted",
        clinical_safety_level="somatic_caution",
        guardrails=[
            "Ensure user is safely seated; instruct them to discontinue if dizzy; note that the 7-second breath retention is optional."
        ],
        disclaimer="Seated exercise only. Discontinue if dizzy; breath retention is optional."
    ),
    "somatic_regulation/03_sensory_grounding_5_4_3_2_1.md": DocumentGovernance(
        document_id="somatic_regulation:03_sensory_grounding_5_4_3_2_1",
        relative_path="somatic_regulation/03_sensory_grounding_5_4_3_2_1.md",
        title="5-4-3-2-1 Sensory Orientation & Somatic Grounding",
        category="somatic_regulation",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Sensory de-escalation protocol for panic or emotional overwhelm."
    ),
    "somatic_regulation/04_progressive_muscle_relaxation.md": DocumentGovernance(
        document_id="somatic_regulation:04_progressive_muscle_relaxation",
        relative_path="somatic_regulation/04_progressive_muscle_relaxation.md",
        title="Progressive Muscle Relaxation (PMR) Protocol",
        category="somatic_regulation",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Systematic neuromuscular relaxation exercise; avoid tensing injured muscles."
    ),
    "somatic_regulation/05_physiological_sigh.md": DocumentGovernance(
        document_id="somatic_regulation:05_physiological_sigh",
        relative_path="somatic_regulation/05_physiological_sigh.md",
        title="The Physiological Sigh: Controlled Cyclic Respiration Technique",
        category="somatic_regulation",
        jurisdiction="GL",
        verification_status="partially_verified",
        retrieval_tier="approved",
        clinical_safety_level="open_informational",
        disclaimer="Cyclic sighing technique for rapid autonomic down-regulation."
    ),
    "somatic_regulation/06_vagus_nerve_stimulation_tapping.md": DocumentGovernance(
        document_id="somatic_regulation:06_vagus_nerve_stimulation_tapping",
        relative_path="somatic_regulation/06_vagus_nerve_stimulation_tapping.md",
        title="Bilateral Somatic Tapping & Tactile Grounding: Non-Clinical Practice",
        category="somatic_regulation",
        jurisdiction="GL",
        verification_status="unverified",
        retrieval_tier="excluded_pending_review",
        clinical_safety_level="blocked_from_retrieval",
        guardrails=[
            "BLOCKED: Lacks neurobiological consensus and independent clinical review. Must not be ingested into vector index."
        ],
        disclaimer="BLOCKED: Unverified alternative therapy claims regarding acupoint vagus stimulation.",
        rejection_reason="Unverified alternative medicine claims linking cutaneous tapping to vagus nerve stimulation and trauma recovery without neurobiological consensus or replicable clinical trials."
    ),
}


def get_document_governance(path_or_id: str) -> Optional[DocumentGovernance]:
    """
    Look up governance record by relative path or document_id.
    Normalizes forward/backward slashes and strips leading data/knowledge_base/.
    """
    cleaned = path_or_id.replace("\\", "/").strip()
    if cleaned.startswith("data/knowledge_base/"):
        cleaned = cleaned[len("data/knowledge_base/"):]
    elif cleaned.startswith("./data/knowledge_base/"):
        cleaned = cleaned[len("./data/knowledge_base/"):]

    # Direct match on relative path
    if cleaned in DOCUMENT_GOVERNANCE_REGISTRY:
        return DOCUMENT_GOVERNANCE_REGISTRY[cleaned]

    # Match by document_id
    for gov in DOCUMENT_GOVERNANCE_REGISTRY.values():
        if gov.document_id == cleaned:
            return gov

    return None


def is_document_eligible_for_ingestion(path_or_id: str) -> bool:
    """
    Determine if a document is permitted to be ingested into the FAISS index.
    Returns True for 'approved' and 'restricted' documents.
    Returns False for 'excluded_pending_review' or unknown documents.
    """
    gov = get_document_governance(path_or_id)
    if gov is None:
        return False
    return gov.retrieval_tier in ("approved", "restricted")


def get_all_governed_documents() -> Dict[str, DocumentGovernance]:
    """Return immutable view of all governed documents."""
    return dict(DOCUMENT_GOVERNANCE_REGISTRY)


def get_counts_by_tier() -> Dict[str, int]:
    """Return document count breakdown across retrieval tiers."""
    counts = {"approved": 0, "restricted": 0, "excluded_pending_review": 0}
    for gov in DOCUMENT_GOVERNANCE_REGISTRY.values():
        counts[gov.retrieval_tier] = counts.get(gov.retrieval_tier, 0) + 1
    return counts
