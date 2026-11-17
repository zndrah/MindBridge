import React from 'react';
import { Info } from 'lucide-react';

export function EmergencyHeaderBanner({ onTriggerEmergency }) {
  return (
    <div className="bg-surface-container-low border-b border-surface-container-high px-4 md:px-6 py-1.5 text-center select-none shrink-0">
      <div className="max-w-[1600px] mx-auto flex items-center justify-center gap-2 flex-wrap">
        <Info className="w-3.5 h-3.5 text-secondary shrink-0" />
        <p className="font-label-sm text-xs text-on-surface-variant flex items-center flex-wrap justify-center gap-1.5">
          <span>Non-clinical emotional first response</span>
          <span>•</span>
          <span>If in acute crisis or physical danger, contact emergency services directly</span>
        </p>
        <button
          onClick={onTriggerEmergency}
          className="text-xs text-primary font-semibold underline hover:text-primary-container transition-colors ml-1"
        >
          View Safety Protocols
        </button>
      </div>
    </div>
  );
}
