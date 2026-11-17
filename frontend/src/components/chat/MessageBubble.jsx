import React from 'react';
import { ShieldCheck, ShieldAlert, AlertCircle, HeartHandshake, HelpCircle } from 'lucide-react';
import { MarkdownView } from '../common/MarkdownView';
import { CitationCard } from '../citations/CitationCard';
import { EmergencyResourceCard } from '../safety/EmergencyResourceCard';

export function MessageBubble({ message }) {
  const {
    role,
    content,
    timestamp,
    citations,
    safetyRoute,
    safetyUI,
    verificationStatus,
    emergencyResources = [],
    isError,
    errorCode,
  } = message;

  if (role === 'user') {
    return (
      <div className="flex justify-end w-full animate-fadeIn">
        <div className="max-w-[90%] sm:max-w-[80%] md:max-w-[75%] bg-primary-fixed-dim/70 text-on-surface px-4 sm:px-5 py-2.5 sm:py-3 rounded-2xl rounded-tr-xs shadow-2xs flex flex-col gap-1 border border-primary-fixed-dim/30 text-left">
          <div className="font-body-md text-sm sm:text-[15px] text-on-surface leading-relaxed whitespace-pre-wrap break-words text-left">
            {content}
          </div>
          <div className="flex items-center justify-end gap-1.5 mt-0.5 text-on-surface-variant/75 font-label-sm text-[10px] sm:text-[11px] select-none">
            <span className="font-medium">You</span>
            <span>•</span>
            <span>{timestamp}</span>
          </div>
        </div>
      </div>
    );
  }

  // Assistant Message
  const isEmergency = safetyRoute === 'emergency_bypass';
  const isFollowUp = safetyRoute === 'pending_safety_follow_up';
  const isDeescalation = safetyRoute === 'supportive_deescalation';

  return (
    <div className="flex justify-start w-full items-start text-left animate-fadeIn">
      <div
        className={`max-w-[95%] sm:max-w-[88%] md:max-w-[82%] px-4 sm:px-5 md:px-6 py-3 sm:py-4 rounded-2xl rounded-tl-xs shadow-2xs flex flex-col gap-2.5 border text-left self-start ${
          isError
            ? 'bg-error-container/20 border-error/30 text-on-surface'
            : isEmergency
            ? 'bg-surface-container-lowest border-tertiary/40 ring-1 ring-tertiary/20'
            : 'bg-surface-container-lowest border-surface-container/40 text-on-surface'
        }`}
      >
        {/* Companion Header Info Bar - clean, compact, unboxed */}
        <div className="flex items-center justify-between gap-2 flex-wrap text-xs select-none">
          <div className="flex items-center gap-1.5 text-primary">
            <span className="w-1.5 h-1.5 rounded-full bg-primary shrink-0" />
            <span className="font-semibold text-xs text-primary tracking-wide">
              MindBridge
            </span>
            {verificationStatus === 'verified' && (
              <span className="inline-flex items-center gap-1 text-secondary font-medium text-[10px] ml-1 px-1.5 py-0.5 rounded-md bg-secondary-container/40">
                <ShieldCheck className="w-3 h-3 text-secondary" />
                <span>Grounded</span>
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5 flex-wrap">
            {isEmergency && (
              <span className="inline-flex items-center gap-1 text-tertiary bg-tertiary-fixed/50 px-2 py-0.5 rounded-full font-semibold text-[10px] tracking-wide uppercase">
                <ShieldAlert className="w-3 h-3 text-tertiary" />
                <span>Safe Harbor Active</span>
              </span>
            )}
            {isFollowUp && (
              <span className="inline-flex items-center gap-1 text-secondary bg-secondary-fixed/50 px-2 py-0.5 rounded-full font-medium text-[10px]">
                <HelpCircle className="w-3 h-3 text-secondary" />
                <span>Safety Follow-Up</span>
              </span>
            )}
            {isDeescalation && (
              <span className="inline-flex items-center gap-1 text-secondary bg-surface-container px-2 py-0.5 rounded-full font-medium text-[10px]">
                <HeartHandshake className="w-3 h-3 text-secondary" />
                <span>Supportive Reflection</span>
              </span>
            )}
          </div>
        </div>

        {/* System Error Banner */}
        {isError && (
          <div className="flex items-center gap-2 text-error text-xs font-semibold p-2.5 rounded-xl bg-error-container/30 border border-error/30">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>System Notice ({errorCode || 'SERVICE_UNAVAILABLE'})</span>
          </div>
        )}

        {/* Formatted Markdown Content */}
        <div className="font-body-md text-sm sm:text-[15px] text-on-surface leading-relaxed text-left">
          <MarkdownView content={content} citations={citations} />
        </div>

        {/* Inline Emergency Resource Cards if provided by backend or error response */}
        {emergencyResources && emergencyResources.length > 0 && (
          <div className="mt-2 pt-3 border-t border-surface-container/40 grid grid-cols-1 sm:grid-cols-2 gap-3">
            {emergencyResources.map((res, i) => (
              <EmergencyResourceCard key={res.resource_id || i} resource={res} />
            ))}
          </div>
        )}

        {/* Evidence Citations */}
        {citations && citations.length > 0 && (
          <div className="mt-1">
            <CitationCard citations={citations} />
          </div>
        )}

        {/* Footer Timestamp */}
        <div className="flex items-center justify-end gap-1.5 pt-0.5 text-on-surface-variant/60 font-label-sm text-[10px] select-none">
          <span>{timestamp}</span>
        </div>
      </div>
    </div>
  );
}
