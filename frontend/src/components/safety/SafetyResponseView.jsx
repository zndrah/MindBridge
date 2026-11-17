import React, { useState, useEffect } from 'react';
import { Heart, ShieldAlert, ArrowLeft, Info, PhoneCall, AlertTriangle, Building2 } from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { getEmergencyResources } from '../../services/api';
import { EmergencyResourceCard } from './EmergencyResourceCard';
import { MarkdownView } from '../common/MarkdownView';

export function SafetyResponseView({ onReturnToChat, customResponseText = null, triggerText = null, resources = null }) {
  const { jurisdiction, activeSafetyEvent } = useChat();
  const [directoryResources, setDirectoryResources] = useState([]);

  useEffect(() => {
    // If explicit resources or activeSafetyEvent emergency resources are already provided, skip fetching
    if ((resources && resources.length > 0) || (activeSafetyEvent?.emergencyResources && activeSafetyEvent.emergencyResources.length > 0)) {
      return;
    }
    let isMounted = true;
    getEmergencyResources(jurisdiction)
      .then((data) => {
        if (isMounted && Array.isArray(data)) {
          setDirectoryResources(data);
        }
      })
      .catch(() => {
        if (isMounted) {
          setDirectoryResources([]);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [jurisdiction, resources, activeSafetyEvent]);

  const responseText = customResponseText || activeSafetyEvent?.response;
  const userTrigger = triggerText || activeSafetyEvent?.triggerMessage;
  
  // Use strictly backend-returned emergency resources
  const backendResources = (resources && resources.length > 0)
    ? resources
    : (activeSafetyEvent?.emergencyResources && activeSafetyEvent.emergencyResources.length > 0)
    ? activeSafetyEvent.emergencyResources
    : directoryResources;

  return (
    <div className="w-full max-w-[1520px] mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 md:py-10 animate-fadeIn">
      {/* Discreet Context Ribbon */}
      <div className="flex items-center justify-between py-2 px-3.5 bg-surface-container-low rounded-xl mb-6 text-xs text-on-surface-variant">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-tertiary"></span>
          <span className="font-semibold uppercase tracking-wider text-[11px]">
            Safe Harbor Protocol Active
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span>Selected Region: <strong>{jurisdiction}</strong></span>
          {onReturnToChat && (
            <button
              onClick={onReturnToChat}
              className="ml-2 flex items-center gap-1 text-primary hover:underline font-medium"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Return to chat</span>
            </button>
          )}
        </div>
      </div>

      {/* User's recent message (if available) rendered with dignity */}
      {userTrigger && (
        <div className="flex flex-col items-end gap-1.5 mb-8">
          <div className="max-w-2xl bg-surface-container p-4 rounded-2xl rounded-tr-sm shadow-xs text-sm sm:text-base text-on-surface leading-relaxed">
            “{userTrigger}”
          </div>
          <span className="text-[11px] text-on-surface-variant pr-1">Just now</span>
        </div>
      )}

      {/* Main Safe Harbor Anchor Container */}
      <div className="relative w-full bg-surface-container-lowest rounded-3xl p-6 sm:p-8 md:p-10 shadow-ambient-md border border-surface-container overflow-hidden mb-10">
        {/* Soft, warm terracotta/earth ambient glow (replaces alarming neon reds) */}
        <div className="absolute -top-24 -right-24 w-72 h-72 rounded-full bg-tertiary-fixed opacity-30 blur-3xl pointer-events-none" />
        <div className="absolute -bottom-24 -left-20 w-72 h-72 rounded-full bg-primary-fixed opacity-20 blur-3xl pointer-events-none" />

        {/* Priority Header */}
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 mb-6 border-b border-surface-container">
          <div className="flex items-center gap-3.5">
            <div className="w-12 h-12 rounded-2xl bg-tertiary-container flex items-center justify-center text-on-tertiary-container shadow-xs shrink-0">
              <Heart className="w-6 h-6 fill-current" />
            </div>
            <div>
              <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full bg-surface-container text-tertiary text-xs font-semibold mb-1">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Immediate Safe Harbor Protocol</span>
              </div>
              <h1 className="font-headline-md text-2xl sm:text-3xl text-on-surface font-semibold">
                You do not have to carry this alone
              </h1>
            </div>
          </div>
          <div className="self-start md:self-auto bg-surface-container-low px-3.5 py-1.5 rounded-full text-xs text-on-surface-variant font-medium">
            Emergency Resources Directory
          </div>
        </div>

        {/* Empathetic Validation & Scope Boundary */}
        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-6 sm:gap-8 my-8">
          <div className="lg:col-span-8 bg-surface-container-low p-6 sm:p-7 rounded-2xl flex flex-col justify-between">
            <div>
              <p className="font-display-mobile text-xl sm:text-2xl md:text-[26px] text-primary font-serif leading-snug">
                “You are carrying a tremendous amount of weight right now, and there are people ready to listen and help you through this.”
              </p>
              <p className="font-body-md text-sm sm:text-base text-on-surface-variant mt-3 leading-relaxed">
                Please connect directly with trained human crisis counselors or local emergency dispatch who can support you through this moment.
              </p>
            </div>

            {/* MindBridge Boundary Clarification */}
            <div className="mt-5 p-4 rounded-xl bg-surface-container-high flex items-start gap-3">
              <Info className="w-5 h-5 text-tertiary shrink-0 mt-0.5" />
              <div className="text-xs sm:text-sm text-on-surface-variant leading-relaxed">
                <strong className="text-on-surface uppercase tracking-wide block mb-1">
                  MindBridge Safety Boundary
                </strong>
                MindBridge is an automated conversational tool. It cannot dispatch emergency responders, place phone calls on your behalf, or replace professional crisis intervention. Please use the direct avenues provided below.
              </div>
            </div>
          </div>

          {/* Serene Grounding Card */}
          <div className="lg:col-span-4 bg-primary-container text-on-primary p-6 sm:p-7 rounded-2xl flex flex-col justify-between min-h-[220px]">
            <div>
              <span className="font-label-sm text-[11px] text-primary-fixed uppercase tracking-wider font-semibold">
                Gentle Anchor
              </span>
              <p className="font-headline-sm text-lg sm:text-xl font-serif italic mt-2 leading-snug">
                “You only need to take this one single breath at a time.”
              </p>
            </div>
            <p className="text-xs sm:text-sm text-on-primary-container mt-5">
              Direct emergency contacts are provided below.
            </p>
          </div>
        </div>

        {/* Backend Emergency Guidance Response */}
        {responseText && (
          <div className="relative z-10 my-8 p-5 sm:p-6 rounded-2xl bg-surface-container border border-surface-container-high">
            <h3 className="text-xs sm:text-sm font-semibold text-secondary uppercase tracking-wider mb-3">
              Listed Emergency Guidance:
            </h3>
            <MarkdownView content={responseText} className="text-sm sm:text-base leading-relaxed" />
          </div>
        )}

        {/* Regional Helplines Grid (Rendered strictly from backend directory data) */}
        <div className="relative z-10 mt-8 mb-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-headline-sm text-lg sm:text-xl font-semibold text-on-surface flex items-center gap-2">
              <PhoneCall className="w-5 h-5 text-primary" />
              <span>Listed Emergency & Crisis Resources</span>
            </h2>
            <span className="text-xs text-on-surface-variant">Institutional Directory Reference</span>
          </div>

          {backendResources.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {backendResources.map((res, i) => (
                <EmergencyResourceCard key={res.resource_id || i} resource={res} />
              ))}
            </div>
          ) : (
            /* Safe Fallback when no structured resource objects were returned */
            <div className="p-6 rounded-2xl bg-surface-container-low border border-surface-container flex flex-col sm:flex-row items-center gap-4 text-xs sm:text-sm text-on-surface-variant">
              <Building2 className="w-8 h-8 text-secondary shrink-0" />
              <div className="space-y-1 text-center sm:text-left">
                <p className="font-semibold text-on-surface text-base">
                  Local Emergency Dispatch & Crisis Care
                </p>
                <p className="leading-relaxed">
                  If you are in immediate physical danger, please dial your local emergency services telephone number or go directly to the nearest hospital emergency department. MindBridge cannot contact emergency personnel for you.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Navigation Back */}
        <div className="relative z-10 mt-10 pt-6 border-t border-surface-container flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-on-surface-variant">
            Emergency contacts compiled from published directory records. Operational availability has not been independently tested.
          </p>
          {onReturnToChat && (
            <button
              onClick={onReturnToChat}
              className="px-5 py-2.5 rounded-xl bg-primary hover:bg-primary-container text-on-primary text-xs sm:text-sm font-semibold transition-colors flex items-center gap-2 shrink-0 shadow-xs"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>Back to Conversation</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
