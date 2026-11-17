/**
 * MindBridge 2.0 - API Client Service (`api.js`)
 * ===============================================
 * Integrates directly with the FastAPI backend according to the exact API contract.
 * - POST /chat
 * - GET  /health
 * - GET  /ready
 */

const API_BASE_URL = ((typeof import.meta !== 'undefined' && import.meta?.env?.VITE_API_BASE_URL) || '').replace(/\/+$/, '');
const REQUEST_TIMEOUT_MS = 30000;

export class ApiError extends Error {
  constructor(message, status, errorCode = null, details = null, emergencyResources = []) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.errorCode = errorCode;
    this.details = details;
    this.emergencyResources = emergencyResources;
  }
}

/**
 * Executes a fetch request with timeout enforcement and error parsing.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
        ...(options.headers || {}),
      },
    });
    clearTimeout(id);
    return response;
  } catch (err) {
    clearTimeout(id);
    if (err.name === 'AbortError') {
      throw new ApiError(
        'Request timed out. The server took too long to respond.',
        408,
        'REQUEST_TIMEOUT'
      );
    }
    throw new ApiError(
      'Unable to connect to MindBridge server. Please verify the backend is running.',
      0,
      'NETWORK_ERROR'
    );
  }
}

/**
 * Check backend liveness probe
 */
export async function getHealth() {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 5000);
    if (!res.ok) {
      throw new ApiError('Service unhealthy', res.status);
    }
    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError('Health probe failed', 0, 'HEALTH_CHECK_FAILED');
  }
}

/**
 * Check backend readiness probe (models, faiss, graph)
 */
export async function getReady() {
  try {
    const res = await fetchWithTimeout(`${API_BASE_URL}/ready`, {}, 10000);
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      return {
        isReady: false,
        status: data?.status || 'not_ready',
        details: data?.details || {},
        models: data?.models || null,
      };
    }
    return {
      isReady: data?.status === 'ready',
      ...data,
    };
  } catch (err) {
    return {
      isReady: false,
      status: 'offline',
      error: err.message,
    };
  }
}

/**
 * Send chat message to /chat
 * Strict boundary: passes ONLY message, session_id, jurisdiction, and conversation_history.
 * No extraneous keys allowed!
 *
 * @param {Object} params
 * @param {string} params.message
 * @param {string} [params.sessionId]
 * @param {string} [params.jurisdiction]
 * @param {Array<{role: string, content: string}>} [params.conversationHistory]
 * @returns {Promise<ChatResponse>}
 */
export async function sendChatMessage({ message, sessionId, jurisdiction, conversationHistory }) {
  const trimmed = (message || '').trim();
  if (!trimmed) {
    throw new ApiError('Message cannot be empty.', 400, 'EMPTY_MESSAGE');
  }

  // Construct payload with ONLY whitelisted fields
  const payload = {
    message: trimmed,
  };

  if (sessionId && typeof sessionId === 'string') {
    // Session ID validation: alphanumeric + '-' + '_'
    const sanitizedSessionId = sessionId.replace(/[^a-zA-Z0-9_\-]/g, '').slice(0, 64);
    if (sanitizedSessionId) {
      payload.session_id = sanitizedSessionId;
    }
  }

  if (jurisdiction) {
    payload.jurisdiction = jurisdiction;
  }

  if (Array.isArray(conversationHistory) && conversationHistory.length > 0) {
    // Send only up to the last 20 valid turns as permitted by backend schema
    const formattedHistory = conversationHistory
      .slice(-20)
      .filter(turn => (turn.role === 'user' || turn.role === 'assistant') && turn.content)
      .map(turn => ({
        role: turn.role,
        content: String(turn.content).slice(0, 10000),
      }));
    if (formattedHistory.length > 0) {
      payload.conversation_history = formattedHistory;
    }
  }

  const requestHeaders = {
    'Content-Type': 'application/json',
  };
  if (jurisdiction) {
    requestHeaders['X-Jurisdiction'] = jurisdiction;
  }

  const res = await fetchWithTimeout(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: requestHeaders,
    body: JSON.stringify(payload),
  }, 45000);

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const errorMsg = data?.error || (Array.isArray(data?.details) ? data.details.join(', ') : 'Server error occurred.');
    const errorCode = data?.error_code || 'API_ERROR';
    const emergencyResources = Array.isArray(data?.emergency_resources) ? data.emergency_resources : [];
    throw new ApiError(errorMsg, res.status, errorCode, data?.details, emergencyResources);
  }

  return {
    ...data,
    emergency_resources: Array.isArray(data?.emergency_resources) ? data.emergency_resources : [],
    citations: Array.isArray(data?.citations) ? data.citations : [],
  };
}

/**
 * Fetch authoritative emergency directory resources for a jurisdiction.
 * Sourced strictly from backend/safety/hotlines.json via GET /safety/resources.
 *
 * @param {string} [jurisdiction='IN']
 * @returns {Promise<Array<Object>>}
 */
export async function getEmergencyResources(jurisdiction = 'IN') {
  try {
    const query = jurisdiction ? `?jurisdiction=${encodeURIComponent(jurisdiction)}` : '';
    const res = await fetchWithTimeout(`${API_BASE_URL}/safety/resources${query}`, {}, 10000);
    if (!res.ok) {
      return [];
    }
    const data = await res.json().catch(() => null);
    return Array.isArray(data) ? data : [];
  } catch (err) {
    return [];
  }
}

/**
 * Execute live AI/ML inference demonstration with empirical telemetry.
 * Sourced strictly from POST /aiml/inference.
 *
 * @param {Object} params
 * @param {string} params.message
 * @param {string} [params.jurisdiction='IN']
 * @returns {Promise<Object>}
 */
export async function runAimlInference({ message, jurisdiction = 'IN' }) {
  const trimmed = (message || '').trim();
  if (!trimmed) {
    throw new ApiError('Message cannot be empty.', 400, 'EMPTY_MESSAGE');
  }

  const payload = {
    message: trimmed,
    jurisdiction: jurisdiction || 'IN',
  };

  const res = await fetchWithTimeout(
    `${API_BASE_URL}/aiml/inference`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Jurisdiction': jurisdiction || 'IN',
      },
      body: JSON.stringify(payload),
    },
    45000
  );

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const errorMsg = data?.error || (Array.isArray(data?.details) ? data.details.join(', ') : 'Inference failed.');
    throw new ApiError(errorMsg, res.status, data?.error_code || 'API_ERROR');
  }

  return data;
}

export { API_BASE_URL };
