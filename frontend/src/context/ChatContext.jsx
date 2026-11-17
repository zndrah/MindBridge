import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { sendChatMessage, getReady, ApiError } from '../services/api';
import { SAFETY_ROUTE_UI_MAP, getSafetyRouteUI } from '../utils/safetyRoutes';

const ChatContext = createContext(null);

export function ChatProvider({ children }) {
  // Ephemeral in-memory message list (NEVER saved to localStorage or sessionStorage)
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(() => 'mb-' + Math.random().toString(36).substring(2, 10));
  const [jurisdiction, setJurisdiction] = useState('IN');
  const [sessionTone, setSessionTone] = useState('listening'); // 'listening' | 'structured' | 'exploration'
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ checked: false, isReady: true, details: null });
  const [activeSafetyEvent, setActiveSafetyEvent] = useState(null);
  const [generalError, setGeneralError] = useState(null);

  // Check backend readiness on load
  useEffect(() => {
    let isMounted = true;
    async function checkServer() {
      const readyInfo = await getReady();
      if (isMounted) {
        setBackendStatus({
          checked: true,
          isReady: readyInfo.isReady,
          details: readyInfo,
        });
      }
    }
    checkServer();
    return () => { isMounted = false; };
  }, []);

  // Start new clean reflection session
  const startNewSession = useCallback(() => {
    setMessages([]);
    setSessionId('mb-' + Math.random().toString(36).substring(2, 10));
    setActiveSafetyEvent(null);
    setGeneralError(null);
  }, []);

  // Clear conversation history according to session semantics
  const clearHistory = useCallback(() => {
    setMessages([]);
    setSessionId('mb-' + Math.random().toString(36).substring(2, 10));
    setActiveSafetyEvent(null);
    setGeneralError(null);
  }, []);

  // Dismiss safety overlay/view back to conversation
  const dismissSafetyEvent = useCallback(() => {
    setActiveSafetyEvent(null);
  }, []);

  // Send message through backend pipeline
  const sendMessage = useCallback(async (text) => {
    const trimmed = (text || '').trim();
    if (!trimmed || isLoading) return;

    setGeneralError(null);

    const userMessageId = 'usr_' + Date.now();
    const userTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Append user message immediately
    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: trimmed,
      timestamp: userTimestamp,
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Build conversation history from existing turns (sanitized to role & content only)
      const conversationHistory = messages.map(m => ({
        role: m.role,
        content: m.content,
      }));

      const response = await sendChatMessage({
        message: trimmed,
        sessionId,
        jurisdiction,
        conversationHistory,
      });

      const assistantMessageId = 'asst_' + Date.now();
      const assistantTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      const rawRoute = response.safety_route;
      const uiState = getSafetyRouteUI(rawRoute);

      // Preserve actual backend response string without overriding
      const safeContent = response.response?.trim() || (
        uiState.isEmergency
          ? "Immediate crisis support is required. Please reach out to the verified emergency services below."
          : "I am here with you. Please take your time."
      );

      const assistantMessage = {
        id: assistantMessageId,
        role: 'assistant',
        content: safeContent,
        timestamp: assistantTimestamp,
        safetyRoute: rawRoute,
        safetyUI: uiState,
        verificationStatus: response.verification_status,
        groundingStatus: response.grounding_status,
        citations: response.citations || [],
        emergencyResources: response.emergency_resources || [],
        errorCode: response.error_code,
      };

      setMessages(prev => [...prev, assistantMessage]);

      // ONLY emergency_bypass triggers the emergency Safe Harbor escalation view
      if (uiState.isEmergency) {
        setActiveSafetyEvent({
          route: rawRoute,
          response: safeContent,
          citations: response.citations || [],
          emergencyResources: response.emergency_resources || [],
          timestamp: assistantTimestamp,
          triggerMessage: trimmed,
        });
      }
    } catch (err) {
      const errorMessageId = 'err_' + Date.now();
      const errorTimestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      let errorText = 'Unable to reach the MindBridge server. Please ensure the backend is running.';
      let errorCode = 'NETWORK_ERROR';
      let errorEmergencyResources = [];

      if (err instanceof ApiError) {
        errorText = err.message;
        errorCode = err.errorCode || 'API_ERROR';
        errorEmergencyResources = Array.isArray(err.emergencyResources) ? err.emergencyResources : [];
      }

      setGeneralError({
        message: errorText,
        code: errorCode,
      });

      // User-friendly display text for HTTP 413 (Payload Too Large) vs network connection errors
      let displayContent = `We encountered a connection issue: ${errorText}. If you are in immediate distress, please call your local emergency services directly.`;
      if (err?.status === 413 || errorCode === 'PAYLOAD_TOO_LARGE') {
        displayContent = `Your message exceeds the maximum allowed request size (64 KB). To ensure responses are safe, responsive, and reliable, please share a shorter message. If you are in distress or need crisis support, please contact your local emergency services directly.`;
      }

      setMessages(prev => [
        ...prev,
        {
          id: errorMessageId,
          role: 'assistant',
          isError: true,
          content: displayContent,
          timestamp: errorTimestamp,
          errorCode,
          emergencyResources: errorEmergencyResources,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, messages, sessionId, jurisdiction]);

  return (
    <ChatContext.Provider
      value={{
        messages,
        sessionId,
        jurisdiction,
        setJurisdiction,
        sessionTone,
        setSessionTone,
        isLoading,
        backendStatus,
        activeSafetyEvent,
        dismissSafetyEvent,
        generalError,
        sendMessage,
        startNewSession,
        clearHistory,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
