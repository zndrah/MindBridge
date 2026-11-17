/**
 * MindBridge 2.0 - Authoritative Safety Route UI Mapping (`safetyRoutes.js`)
 * =========================================================================
 * Pure JavaScript utility defining the mapping from backend safety routes
 * to UI presentation states.
 *
 * Core Governance Invariants:
 * 1. ONLY emergency_bypass triggers emergency Safe Harbor UI.
 * 2. elevated_classifier_signal is NEVER an emergency state (screening alert only).
 * 3. pending_safety_follow_up is multi-turn clinical inquiry in the chat stream (NOT emergency dispatch).
 * 4. supportive_deescalation and ml_risk_unassessed are supportive chat modes.
 * 5. Unknown routes use a safe neutral fallback (standard_conversation).
 */

export const SAFETY_ROUTE_UI_MAP = {
  emergency_bypass: {
    isEmergency: true,
    mode: 'emergency_bypass',
    badgeText: 'Emergency Safe Harbor',
    badgeVariant: 'emergency',
    description: 'Immediate emergency protocol active with verified hotlines',
  },
  pending_safety_follow_up: {
    isEmergency: false,
    mode: 'pending_safety_follow_up',
    badgeText: 'Structured Follow-Up',
    badgeVariant: 'warning',
    description: 'Clinical safety follow-up in progress within conversation',
  },
  supportive_deescalation: {
    isEmergency: false,
    mode: 'supportive_deescalation',
    badgeText: 'Supportive De-escalation',
    badgeVariant: 'neutral',
    description: 'Empathetic de-escalation dialogue active',
  },
  elevated_classifier_signal: {
    isEmergency: false,
    mode: 'elevated_classifier_signal',
    badgeText: 'Sensitive Topic',
    badgeVariant: 'neutral',
    description: 'Careful pacing active',
  },
  ml_risk_unassessed: {
    isEmergency: false,
    mode: 'ml_risk_unassessed',
    badgeText: 'Safe Fallback',
    badgeVariant: 'neutral',
    description: 'Operating in safe fallback mode',
  },
  standard_conversation: {
    isEmergency: false,
    mode: 'standard_conversation',
    badgeText: null,
    badgeVariant: 'default',
    description: 'Standard conversation',
  },
};

export function getSafetyRouteUI(route) {
  if (route && SAFETY_ROUTE_UI_MAP[route]) {
    return SAFETY_ROUTE_UI_MAP[route];
  }
  // Safe neutral fallback, NOT an assumed emergency
  return SAFETY_ROUTE_UI_MAP.standard_conversation;
}

/**
 * Resolves safe SMS action data from a backend resource object.
 * Governance Invariants:
 * 1. Modality must explicitly include 'sms'.
 * 2. Never infer SMS capability or contact numbers from shortcode length or phone format.
 * 3. SMS actions are constructed ONLY when the directory record explicitly supplies validated sms_instructions.
 * 4. Does not invent contact methods, numbers, or keywords.
 */
export function getSafeSmsAction(resource) {
  if (!resource || !Array.isArray(resource.modality) || !resource.modality.includes('sms')) {
    return null;
  }

  const { sms_instructions } = resource;

  // Strict invariant: Do not infer SMS validity from shortcode length or phone presence.
  // Directives must be explicitly supplied in the directory record's sms_instructions.
  if (!sms_instructions || typeof sms_instructions !== 'string') {
    return null;
  }

  const trimmed = sms_instructions.trim();

  // Pattern A: "Text KEYWORD to NUMBER" (e.g., standard SMS shortcode activation with keyword)
  const matchKeyword = trimmed.match(/^Text\s+([A-Z0-9]+)\s+to\s+([0-9\s\-]+)$/i);
  if (matchKeyword) {
    const keyword = matchKeyword[1].trim();
    const rawNumber = matchKeyword[2].trim();
    const cleanNumber = rawNumber.replace(/[^0-9+]/g, '');
    if (cleanNumber) {
      return {
        type: 'single',
        href: `sms:${cleanNumber}?body=${encodeURIComponent(keyword)}`,
        label: `Text ${keyword} to ${rawNumber}`,
        number: rawNumber,
        keyword: keyword,
      };
    }
  }

  // Pattern B: "Text NUMBER" (e.g., direct text without keyword)
  const matchNumberOnly = trimmed.match(/^Text\s+([0-9\s\-+]+)$/i);
  if (matchNumberOnly) {
    const rawNumber = matchNumberOnly[1].trim();
    const cleanNumber = rawNumber.replace(/[^0-9+]/g, '');
    if (cleanNumber) {
      return {
        type: 'single',
        href: `sms:${cleanNumber}`,
        label: `Text ${rawNumber}`,
        number: rawNumber,
      };
    }
  }

  // Pattern C: Demographic or conditional instructions (e.g. Youth vs Adults)
  return {
    type: 'instruction_only',
    instructionsText: trimmed,
  };
}
