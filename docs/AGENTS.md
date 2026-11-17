# MindBridge 2.0 — Functional Node & Module Specifications

**Document Version:** 2.0.0  
**Status:** Approved Component Specification (Phase 4 Milestone)  
**Date:** 2026-11-02  
**Target Environment:** Local WSL2 Linux (16 Logical Cores, 7.57 GB RAM, CPU-Only Execution)  
**Workspace:** `/home/adarsh/projects/MindBridge`  

---

## 1. Architectural Framework: Modular Nodes within a Single State Graph

In MindBridge 2.0, the term "agent" does not denote an unconstrained, autonomous LLM entity. Instead, it refers to **deterministic, testable transformation nodes** integrated into a single LangGraph state machine.

Each node operates on the shared `MindBridgeState` container according to a strict interface contract:
```python
def node_name(state: MindBridgeState) -> Dict[str, Any]:
    """
    Pure transformation node: reads state immutably and returns
    a dictionary of state mutations to be merged by the orchestrator.
    """
    ...
```

The system strictly supports seven standardized jurisdiction codes:
```python
JurisdictionCode = Literal["IN", "US", "UK", "CA", "AU", "GL", "UNKNOWN"]
```
And six mutually exclusive operational safety states:
```python
SafetyState = Literal[
    "emergency_bypass",
    "pending_safety_follow_up",
    "supportive_deescalation",
    "standard_conversation",
    "elevated_classifier_signal",
    "ml_risk_unassessed"
]
```

---

## 2. Input Processing Node (`input_node`)

* **Subsystem:** Input Ingestion & Preprocessing
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/input_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/input_agent.py)

### 2.1 Responsibilities
1. **Pre-Truncation Emergency Scanning:** Scan the un-truncated raw input string in ephemeral memory using deterministic regex emergency rules. This guarantees that explicit crisis disclosures are never silently lost due to message length limits.
2. **Safety-Aware Long-Message Chunking:** If raw input exceeds sequence limits ($L > 384$ tokens or $> 2,000$ characters), partition the message using a sliding window (256-token window with 64-token overlap, i.e., 192-token stride). The complete message is evaluated for safety across all chunks before any truncation or rejection.
3. **PII Minimization & Scrubbing:** Apply in-memory regex scrubbers to detect and mask phone numbers, email addresses, 12-digit Indian Aadhaar numbers, 9-digit US SSNs, and URLs.
4. **Redaction Limitations:** Acknowledge that regex scrubbers reliably detect structured numeric/pattern tokens. Heuristic personal name redaction has known limitations (cannot guarantee 100% recall or precision without machine learning NER; potential false positives on common nouns and false negatives on uncommon names).
5. **Raw Text Lifecycle & Ephemeral Input-Boundary Isolation:** The raw user text is defined strictly as temporary input-boundary data held in ephemeral in-memory buffers during node execution. It is **never** added to or stored in `MindBridgeState`. Only sanitized text (`sanitized_text`) and sanitized conversation history (`messages`) enter graph state. The raw text is discarded upon node completion; graph checkpoints, state persistence, execution traces (`execution_trace`), and application logs strictly never persist unmasked raw text.
6. **Input Normalization & Intent Tagging:** Strip control characters, normalize whitespace, validate Unicode, and perform preliminary conversational intent classification.

### 2.2 Interface Specification
* **Inputs:** `raw_user_text: str` (ephemeral input-boundary parameter passed to `input_node`; never stored in `MindBridgeState`), `state["session_id"]` (str)
* **Outputs / State Mutations:**
  * `sanitized_text`: Cleaned, PII-masked user utterance.
  * `dialog_goal`: Initial conversational intent classification.
  * `execution_trace`: Appends input validation timestamp and sanitization flags (metadata only; never unmasked raw text).
* **Failure Behavior:** If input processing or Unicode normalization raises an exception, the node executes a fallback regex scan over the raw in-memory string. If explicit danger is matched, it transitions to `emergency_bypass`; otherwise, it emits a safe clarification prompt without invoking downstream models.

---

## 3. Safety Intelligence Node (`safety_node`)

* **Subsystem:** Dual-Engine Crisis Triage & Classifier Screening
* **Implementation Status:** **Tier 1 Implemented Weights** ([`models/crisis_detector/roberta/`](file:///home/adarsh/projects/MindBridge/models/crisis_detector/roberta/)) / **Tier 2 Proposed Node** ([`backend/agents/safety_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/safety_agent.py))

### 3.1 Responsibilities
1. **Deterministic Emergency Pattern Matching:** Scan sanitized text against hardcoded regular expressions for acute crisis keywords (explicit self-harm threats, statements of lethal intent, suicide notes).
2. **Immediate Danger Intercept:** Immediate-danger disclosures activate the emergency safety pathway **independently of RoBERTa confidence and without requiring all seven risk dimensions to be present**. The assessment avoids simplistic Boolean formulas.
3. **RoBERTa Crisis Classification ($L = 384$):** Tokenize text using Byte-level BPE ($L = 384$) and execute inference using the fine-tuned RoBERTa model (`model.safetensors`). For multi-chunk long messages, compute crisis probability across all sliding window chunks (256-token window with 64-token overlap, i.e., 192-token stride) evaluating the complete message before any truncation, and assign:
   $$p_{\text{crisis}} = \max_i p_i$$
4. **Triage State Determination:**
   * If explicit danger matched $\to$ set `overall_safety_route = "emergency_bypass"`.
   * If PHQ-9 Item 9 endorsed $\to$ set `overall_safety_route = "pending_safety_follow_up"`.
   * If $p_{\text{crisis}} \ge 0.5098$ without explicit danger $\to$ set `overall_safety_route = "elevated_classifier_signal"`. **An elevated RoBERTa signal alone must NEVER activate `emergency_node`.** It initiates structured safety follow-up (`pending_safety_follow_up`). Route to emergency guidance only when explicit immediate-danger disclosure or holistic safety assessment indicates immediate danger.
   * If $p_{\text{crisis}} < 0.5098$ and no danger $\to$ set `overall_safety_route = "standard_conversation"`.

### 3.2 Interface Specification
* **Inputs:** `state["sanitized_text"]` (str), `state["messages"]` (list)
* **Outputs / State Mutations:**
  * `deterministic_danger_flag`: bool (True if explicit regex pattern matched).
  * `emergency_trigger_matched`: Optional[str].
  * `roberta_prob`: Optional[float] (sigmoid probability of Crisis-Risk).
  * `roberta_crisis_flag`: bool (True if $p \ge 0.5098$).
  * `phq9_item9_endorsed`: bool.
  * `overall_safety_route`: SafetyState.
* **Failure Behavior:** If RoBERTa inference times out ($> 200\text{ms}$) or raises a runtime exception, **the system shall NOT automatically classify the user as high-risk or safe**. Deterministic regex rules continue scanning. If no danger is found, the node sets `roberta_prob = None` and `overall_safety_route = "ml_risk_unassessed"`, routing to a safe non-clinical fallback offering general help resources without alarming the user.

---

## 4. Structured Safety Follow-Up Node (`followup_node`)

* **Subsystem:** Multi-Turn Risk Assessment & Item 9 Follow-Up
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/safety_followup_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/safety_followup_agent.py)

### 4.1 Responsibilities
1. **Multi-Turn Structured Evaluation:** When PHQ-9 Item 9 is endorsed or an elevated classifier signal requires evaluation, conduct structured dialogue across the exact seven clinical dimensions:
   1. *Passive death wishes* (desire for relief or cessation without suicidal intent).
   2. *Active suicidal ideation* (explicit thoughts of taking one's life).
   3. *Intent* (stated commitment or determination to act).
   4. *Plan* (formulated method, location, or sequence of actions).
   5. *Means* (knowledge of or possession of lethal mechanisms).
   6. *Access* (physical availability and proximity to lethal means).
   7. *Timeframe* (immediacy or stated timeline).
2. **Contextual Evaluation:** Record protective factors and emotional agitation strictly as supplementary contextual considerations to inform conversational tone, without modifying the seven-dimensional framework.
3. **Multi-Turn State Tracking:** Set `phq9_followup_status = "in_progress"` and maintain `overall_safety_route = "pending_safety_follow_up"` across assessment turns until complete.
4. **Holistic Risk Routing:**
   * If evaluation reveals imminent peril (active ideation + intent/plan/means/access/imminent timeframe) $\to$ transition to `emergency_bypass` (`emergency_node`).
   * If evaluation reveals passive death wishes or ambiguous distress without imminent peril or compressed timeframe $\to$ transition to `supportive_deescalation` (`deescalation_node`).

### 4.2 Interface Specification
* **Inputs:** `state["sanitized_text"]`, `state["messages"]`, `state["phq9_followup_status"]`
* **Outputs / State Mutations:**
  * `phq9_followup_status`: Literal["in_progress", "completed", "interrupted"].
  * `phq9_risk_level`: Optional[Literal["imminent", "passive_unclear", "none", "unassessed"]].
  * `overall_safety_route`: Literal["emergency_bypass", "supportive_deescalation", "pending_safety_follow_up"].
* **Failure Behavior:** If the follow-up module encounters an unhandled exception, conversational interruption, or incomplete evaluation:
  * The orchestrator sets `phq9_followup_status = "interrupted"` and records `phq9_risk_level = "unassessed"`.
  * The system shall **never** silently mark the assessment complete or classify the user as passive-risk.
  * The node emits an empathetic continuation prompt offering the user the option to resume structured reflection when ready, alongside accessible verified human-support resources (e.g., standard crisis lifelines), without triggering disruptive emergency takeover screens.

---

## 5. Emergency Bypass Node (`emergency_node`)

* **Subsystem:** Deterministic Emergency Crisis Escalation
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/emergency_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/emergency_agent.py)

### 5.1 Responsibilities
1. **Bypass Open Generation:** Completely halt RAG vector retrieval and generative LLM invocation.
2. **Jurisdiction-Aware Contact Presentation:** Filter [`backend/safety/hotlines.json`](file:///home/adarsh/projects/MindBridge/backend/safety/hotlines.json) by `state["jurisdiction_filter"]`. If jurisdiction is known (`IN`, `US`, `UK`, `CA`, `AU`), output region-specific contacts. If `UNKNOWN`, direct the user to dial their local emergency dispatch service immediately and prompt for country clarification.
3. **Category Distinction & Regional Nuance:** Clearly distinguish national emergency dispatch numbers (e.g., `112`, `911`, `999`) from mental health crisis lifelines (e.g., `14416`, `988`), urgent health pathways (UK NHS `111`), and emotional listening services (UK Samaritans `116 123`).
   * *UK NHS 111 Regional Applicability:* UK NHS 111 Option 2 is a mental health triage pathway operating across NHS England and parts of Wales; applicability must be verified for the relevant region (Scotland routes mental health triage via NHS 24 on 111 with distinct workflows, and Northern Ireland maintains regional health and social care trusts).
   * *Non-Substitution of Global Resources:* Global resources (`GL`, e.g., Befrienders Worldwide) must **never** be presented as substitutes for local emergency dispatch contacts.
4. **User-Initiated Guidance Enforcement:** Present clear instructions for the user or their caregiver to initiate contact directly. Phrasing implying automated system dialing or emergency responder dispatch is strictly prohibited.

### 5.2 Interface Specification
* **Inputs:** `state["jurisdiction_filter"]`, `state["emergency_trigger_matched"]`
* **Outputs / State Mutations:**
  * `draft_response`: None.
  * `final_response`: Pre-rendered emergency guidance card.
  * `verification_passed`: True (pre-verified static template).
* **Failure Behavior:** If hotline lookup fails or state is corrupted, emit the hardcoded static catastrophic safety card from `backend/safety/fallbacks.py`.

---

## 6. Supportive De-Escalation Node (`deescalation_node`)

* **Subsystem:** Compassionate Grounding & Validation
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/deescalation_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/deescalation_agent.py)

### 6.1 Responsibilities
1. **Empathetic Emotional Validation:** Acknowledge and validate feelings of chronic exhaustion, passive death wishes, or emotional distress without clinical judgment.
2. **Non-Intrusive Human Support Delivery:** Offer verified, non-emergency crisis support contacts (e.g., India Tele-MANAS `14416`, US/Canada `988`, UK Samaritans `116 123`, Crisis Text Lines) as supportive resources without triggering disruptive emergency takeover screens.
3. **Safeguard Somatic Suitability:** Unconditional breathing instructions (e.g., *"Let's take a slow breath together"*) are strictly prohibited. Somatic grounding may be offered only after checking contextual suitability (absence of acute panic, hyperventilation, respiratory distress, trauma triggers).

### 6.2 Interface Specification
* **Inputs:** `state["sanitized_text"]`, `state["jurisdiction_filter"]`, `state["phq9_risk_level"]`
* **Outputs / State Mutations:**
  * `draft_response`: Candidate de-escalation text.
  * `final_response`: Validated supportive response.
  * `verification_passed`: True.
* **Failure Behavior:** Falls back to the pre-approved neutral supportive template from `backend/safety/fallbacks.py`: *"I am here with you. I am ready to support you with information or resources whenever you are ready."*

---

## 7. Planning & Routing Node (`planning_node`)

* **Subsystem:** Conversational Orchestration
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/planning_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/planning_agent.py)

### 7.1 Responsibilities
1. **Dialog Goal Arbitration:** Determine operational dialog goal:
   * `somatic_grounding` (wellness exercises, tension release).
   * `cognitive_reframing` (educational self-reflection on cognitive distortions; strictly psychoeducational, not psychotherapy).
   * `screening` (standardized rubric reflection: PHQ-9, GAD-7, PSS-10).
   * `general` (supportive mental health psychoeducation).
2. **Retrieval Query Formulation:** Generate 1–3 focused semantic search queries targeting the audited knowledge base.
3. **Jurisdiction Filter Resolution:** Map session metadata or user preference to one of seven supported jurisdiction codes: `IN`, `US`, `UK`, `CA`, `AU`, `GL`, or `UNKNOWN`. For `UNKNOWN`, prepare a location clarification prompt and location-independent queries.

### 7.2 Interface Specification
* **Inputs:** `state["sanitized_text"]`, `state["messages"]`, `state["overall_safety_route"]`
* **Outputs / State Mutations:**
  * `dialog_goal`: Optional[Literal["somatic_grounding", "cognitive_reframing", "screening", "general"]].
  * `retrieval_queries`: List[str].
  * `jurisdiction_filter`: JurisdictionCode.
* **Failure Behavior:** If planning cannot categorize intent, it defaults to `dialog_goal = "general"` and generates queries for core psychoeducation guidelines, inheriting the safeguard that neutral supportive phrasing is used.

---

## 8. Retrieval Node (`retrieval_node`)

* **Subsystem:** Dense Vector Knowledge Retrieval
* **Implementation Status:** **Tier 1 Audited Docs** ([`data/knowledge_base/`](file:///home/adarsh/projects/MindBridge/data/knowledge_base/)) / **Tier 2 Proposed Index** ([`backend/agents/retrieval_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/retrieval_agent.py))

### 8.1 Responsibilities
1. **Dense Query Embedding & $L_2$ Normalization:** Convert retrieval queries into 384-dimensional dense vectors using `BAAI/bge-small-en-v1.5` on CPU, applying unit $L_2$ normalization ($\mathbf{v}' = \mathbf{v}/\|\mathbf{v}\|_2$).
2. **FAISS Inner Product Search:** Query the local FAISS index (`IndexFlatIP`). On unit-normalized vectors, inner product directly computes cosine similarity:
   $$\langle \mathbf{u}', \mathbf{v}' \rangle = \cos\theta$$
3. **Similarity Threshold Filtering:** Retrieve top-$k$ ($k=3$) chunks meeting the baseline threshold $\cos\theta \ge 0.65$. (Equivalent $L_2$ distance threshold under unit normalization is $d \le \sqrt{2(1-0.65)} \approx 0.8367$).
4. **Pre-RAG Eligibility Enforcement:**
   * *Approved Documents (13 files):* Ingested into open retrieval context.
   * *Restricted Documents (7 files):* Ingested **only** when query intent matches clinical rubrics, appending mandatory non-diagnostic disclaimers.
   * *Excluded Documents (1 file):* Bilateral tapping (`06_vagus_nerve_stimulation_tapping.md`) is **strictly blocked** from the index.
5. **Jurisdiction Filtering:** Filter chunks matching `jurisdiction_filter` or marked `GL`. For `UNKNOWN`, retrieve only `GL` chunks.

### 8.2 Interface Specification
* **Inputs:** `state["retrieval_queries"]`, `state["jurisdiction_filter"]`
* **Outputs / State Mutations:**
  * `retrieved_chunks`: List of `KnowledgeChunk` dictionaries.
* **Failure Behavior:** If FAISS index is unavailable, times out ($> 100\text{ms}$), or returns no chunks above 0.65 similarity, the node logs a retrieval failure and injects static core psychoeducation guidelines from `backend/safety/fallbacks.py`, checking somatic suitability before presenting any grounding content.

---

## 9. Response Generation Node (`response_node`)

* **Subsystem:** Grounded Natural Language Generation
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/response_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/response_agent.py)

### 9.1 Responsibilities
1. **Grounded Synthesis:** Synthesize empathetic, clear responses grounded strictly in the retrieved knowledge chunks using the local instruction-tuned LLM.
2. **Enforce Non-Diagnostic Framing:** Apply system prompts prohibiting diagnostic labeling, medical advice, or pharmacological suggestions.
3. **Cognitive Reframing Boundaries:** Present cognitive reframing strictly as educational self-reflection worksheets, never as clinical CBT psychotherapy.
4. **Enforce User-Initiated Emergency Guidance:** When presenting emergency contacts, strictly provide verified details from `backend/safety/hotlines.json` with instructions for the user to dial directly. Phrasing implying automated system dialing or emergency responder dispatch is strictly prohibited.
5. **Disclaimer Injection:** Automatically append required non-diagnostic disclaimers when clinical screening rubrics are discussed.

### 9.2 Interface Specification
* **Inputs:** `state["sanitized_text"]`, `state["retrieved_chunks"]`, `state["dialog_goal"]`, `state["phq9_risk_level"]`, `state["jurisdiction_filter"]`
* **Outputs / State Mutations:**
  * `draft_response`: Synthesized candidate response text.
* **Failure Behavior:** If the generative LLM process times out (proposed hard generation cutoff timeout: $2.5\text{s}$ for concise turns $\le 30$–$35$ tokens; up to $20.0\text{s}$ generation budget for full 256-token synthesis; see Section 7.3 of `ARCHITECTURE.md` regarding the unresolved latency/throughput trade-off at 15 tokens/sec) or returns malformed text, the node falls back to a pre-approved neutral supportive template from `backend/safety/fallbacks.py`: *"I am here with you. I am ready to support you with information or resources whenever you are ready."* Unconditional breathing instructions are strictly prohibited.

---

## 10. Verification Node (`verification_node`)

* **Subsystem:** Post-Generation Safety Gate
* **Implementation Status:** **Tier 2 (Proposed Architecture)**
* **Component Location:** [`backend/agents/verification_agent.py`](file:///home/adarsh/projects/MindBridge/backend/agents/verification_agent.py)

### 10.1 Responsibilities
1. **Contact Fidelity Verification & Normalization:**
   * Scan draft response for telephone numbers, short codes, and URLs.
   * Normalize detected strings by stripping whitespace, punctuation, hyphens, and international dialing prefixes (`+91`, `+1`, `+44`).
   * Match against verified entries in `backend/safety/hotlines.json` (accounting for 3-digit dispatch codes, 5/6-digit SMS text codes, and toll-free lines).
   * Filter false positives (e.g., 4-digit calendar years, quantities, time durations like "24/7").
   * Any unrecognized or hallucinated telephone number triggers an immediate safety violation. Acknowledge that deterministic regex verification significantly reduces risk but cannot guarantee 100% detection of all novel or obfuscated contact patterns.
2. **Medical & Diagnostic Claim Suppression:** Check draft text against a lexicon of prohibited clinical phrasing (diagnostic assertions, prescription advice, medication dosing).
3. **Emergency Phrasing Audit:** Verify that no phrasing implies automated calling or dispatch.
4. **Context-Aware Fallback Substitution:** If all verification checks pass, set `final_response = draft_response`. If any check fails, suppress draft response. **Verification failure must preserve the actual safety state; it must NOT convert elevated classifier risk into emergency status.** Emergency contact guidance is emitted **only** when explicit immediate-danger disclosure or holistic safety assessment indicates immediate danger (`emergency_bypass`). For all other states (including `elevated_classifier_signal` and `pending_safety_follow_up`), substitute a safe supportive fallback that preserves the active safety state without escalating to emergency dispatch.

### 10.2 Interface Specification
* **Inputs:** `state["draft_response"]`, `state["overall_safety_route"]`, `state["jurisdiction_filter"]`
* **Outputs / State Mutations:**
  * `verification_passed`: bool.
  * `verification_flags`: List[str].
  * `final_response`: Validated response string ready for emission.
* **Failure Behavior:** If the verification node encounters an internal exception, it suppresses the draft response and emits a safe fallback appropriate to the active safety state: a vetted supportive non-emergency template for general conversations, or verified crisis resources ONLY if the conversation was already flagged for immediate emergency crisis (`emergency_bypass`). Verification failure preserves the active safety state and never converts elevated classifier risk into emergency status.
