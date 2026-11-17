import React from 'react';
import { Phone, MessageSquare, ExternalLink, ShieldAlert, HeartHandshake, Building2 } from 'lucide-react';
import { getSafeSmsAction } from '../../utils/safetyRoutes';

export function EmergencyResourceCard({ resource }) {
  if (!resource) return null;

  const {
    resource_name,
    authority,
    telephone,
    alternate_telephone,
    dialing_instructions,
    sms_instructions,
    resource_type,
    modality = [],
    availability,
    toll_free,
    official_url,
    operational_notes,
  } = resource;

  const isDispatch = resource_type === 'emergency_dispatch' || resource_type === 'hospital_emergency';
  const isCounselling = resource_type === 'crisis_counselling';
  const isTextSupport = resource_type === 'text_support' || resource_type === 'crisis_text_service';

  // Voice actions: explicitly requires 'voice' in modality AND a valid telephone number
  const supportsVoice = Array.isArray(modality) && modality.includes('voice') && Boolean(telephone);
  const primaryTel = supportsVoice ? telephone.replace(/[^0-9+]/g, '') : null;
  const alternateTel = (supportsVoice && alternate_telephone) ? alternate_telephone.replace(/[^0-9+]/g, '') : null;

  // Safe SMS action resolver: requires keyword instructions if needed; never creates misleading links
  const safeSms = getSafeSmsAction(resource);

  return (
    <div
      className={`rounded-2xl p-5 sm:p-6 flex flex-col justify-between transition-all hover:shadow-ambient-md border ${
        isDispatch
          ? 'bg-error-container/20 border-error/40'
          : isCounselling
          ? 'bg-surface-container border-secondary/30'
          : 'bg-surface-container border-surface-container-high'
      }`}
    >
      <div>
        {/* Category Header Badge */}
        <div className="flex items-center justify-between mb-2.5">
          <span
            className={`px-2.5 py-0.5 font-label-sm text-[11px] font-semibold rounded-full flex items-center gap-1 ${
              isDispatch
                ? 'bg-error text-on-error'
                : isCounselling
                ? 'bg-secondary text-on-secondary'
                : 'bg-surface-container-high text-on-surface-variant'
            }`}
          >
            {isDispatch ? (
              <>
                <ShieldAlert className="w-3 h-3" />
                <span>Emergency Dispatch</span>
              </>
            ) : isCounselling ? (
              <>
                <HeartHandshake className="w-3 h-3" />
                <span>Crisis Counselling</span>
              </>
            ) : isTextSupport ? (
              <>
                <MessageSquare className="w-3 h-3" />
                <span>Text Support</span>
              </>
            ) : (
              <span>Support Resource</span>
            )}
          </span>

          {/* Availability: clearly distinguishes directory-provided information from confirmed availability */}
          {availability && (
            <span
              className="text-[11px] text-on-surface-variant font-medium bg-surface-container-low px-2 py-0.5 rounded-full border border-surface-container"
              title="Directory-listed schedule; live operational availability not independently verified"
            >
              Listed: {availability}
            </span>
          )}
        </div>

        {/* Resource Name & Authority */}
        <h3 className="font-headline-sm text-base text-on-surface font-semibold mb-1">
          {resource_name}
        </h3>
        {authority && (
          <p className="text-[11px] text-secondary font-medium mb-1.5 leading-snug">
            {authority}
          </p>
        )}

        {/* Dialing Instructions (e.g., department option selection) */}
        {dialing_instructions && (
          <div className="p-2 rounded-xl bg-surface-container-high text-[11px] font-medium text-on-surface mb-2 border border-surface-container">
            <span className="text-secondary font-semibold block mb-0.5">Dialing Instructions:</span>
            {dialing_instructions}
          </div>
        )}

        {/* Operational Notes */}
        {operational_notes && (
          <p className="font-body-sm text-xs text-on-surface-variant mb-3 leading-relaxed">
            {operational_notes}
          </p>
        )}

        {toll_free && (
          <span className="inline-block text-[10px] uppercase tracking-wider font-semibold text-primary bg-primary-fixed/50 px-2 py-0.5 rounded mb-3">
            Toll-Free Service
          </span>
        )}
      </div>

      {/* Action Triggers (Strictly capability-governed; zero invented methods) */}
      <div className="space-y-2 pt-2 border-t border-surface-container/60">
        {/* Voice Calling Triggers */}
        {supportsVoice && primaryTel && (
          <a
            href={`tel:${primaryTel}`}
            className={`w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-xl font-label-md text-xs font-semibold transition-colors shadow-xs ${
              isDispatch
                ? 'bg-error text-on-error hover:bg-error/90'
                : 'bg-primary text-on-primary hover:bg-primary-container'
            }`}
          >
            <Phone className="w-3.5 h-3.5" />
            <span>Call {telephone}</span>
          </a>
        )}

        {supportsVoice && alternateTel && (
          <a
            href={`tel:${alternateTel}`}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-surface-container-high text-on-surface font-label-md text-xs font-semibold hover:bg-surface-container-highest transition-colors"
          >
            <Phone className="w-3 h-3 text-secondary" />
            <span>Alt: {alternate_telephone}</span>
          </a>
        )}

        {/* Safe SMS Triggers */}
        {safeSms && safeSms.type === 'single' && (
          <a
            href={safeSms.href}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-secondary text-on-secondary font-label-md text-xs font-semibold hover:bg-secondary/90 transition-colors"
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>{safeSms.label}</span>
          </a>
        )}

        {safeSms && safeSms.type === 'multi' && (
          <div className="space-y-1.5">
            {safeSms.options.map((opt, idx) => (
              <a
                key={idx}
                href={opt.href}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-secondary text-on-secondary font-label-md text-xs font-semibold hover:bg-secondary/90 transition-colors"
              >
                <MessageSquare className="w-3.5 h-3.5" />
                <span>{opt.label}</span>
              </a>
            ))}
          </div>
        )}

        {safeSms && safeSms.type === 'instruction_only' && (
          <div className="p-2 rounded-xl bg-surface-container-high text-xs text-on-surface text-center">
            <span className="font-semibold block mb-0.5">SMS Instructions:</span>
            <span>{safeSms.instructionsText}</span>
          </div>
        )}

        {/* Directory Website Link */}
        {official_url && (
          <a
            href={official_url}
            target="_blank"
            rel="noopener noreferrer"
            className="w-full flex items-center justify-center gap-1.5 py-1 text-xs text-secondary hover:text-primary transition-colors text-center font-medium"
          >
            <span>Directory Website</span>
            <ExternalLink className="w-3 h-3" />
          </a>
        )}

        {/* Operational Verification Disclaimers */}
        <p className="text-[10px] text-on-surface-variant/70 italic text-center pt-1">
          Live operational availability not independently verified.
        </p>
      </div>
    </div>
  );
}
