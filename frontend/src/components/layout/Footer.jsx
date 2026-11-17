import React from 'react';
import { ShieldCheck, Heart, AlertTriangle } from 'lucide-react';

export function Footer({ onNavigatePage }) {
  return (
    <footer className="w-full bg-surface-container-low border-t border-surface-container mt-space-md md:mt-space-lg">
      <div className="w-full max-w-[1560px] mx-auto px-gutter-mobile md:px-gutter py-space-md md:py-space-lg flex flex-col md:flex-row items-center justify-between gap-space-md text-on-surface-variant">
        <div className="flex flex-col sm:flex-row items-center gap-2 text-center sm:text-left">
          <span className="font-headline-sm text-headline-sm text-primary font-semibold">MindBridge 2.0</span>
          <span className="hidden sm:inline text-outline">•</span>
          <span className="font-body-sm text-body-sm text-on-surface-variant">
            Empathetic, evidence-grounded non-clinical emotional first response.
          </span>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-medium">
          <button
            onClick={() => onNavigatePage && onNavigatePage('aiml')}
            className="hover:text-primary transition-colors underline decoration-dotted font-semibold text-primary"
          >
            AIML Performance Dashboard
          </button>
          <button
            onClick={() => onNavigatePage && onNavigatePage('safety')}
            className="hover:text-primary transition-colors underline decoration-dotted"
          >
            Emergency Resources
          </button>
          <button
            onClick={() => onNavigatePage && onNavigatePage('safety')}
            className="hover:text-primary transition-colors underline decoration-dotted"
          >
            Scope & Boundaries
          </button>
          <span className="text-outline/40">|</span>
          <span className="text-[11px] text-on-surface-variant/80">
            Sessions are strictly ephemeral in memory
          </span>
        </div>
      </div>

      <div className="bg-surface-container-high py-2 px-gutter-mobile text-center text-[11px] text-on-surface-variant border-t border-surface-container">
        <p className="max-w-[900px] mx-auto">
          <strong>Important Clinical Boundary:</strong> MindBridge is an automated conversational tool, not a medical or psychological healthcare provider. If you are experiencing acute thoughts of suicide or physical danger, please contact your national emergency services or visit an emergency room immediately.
        </p>
      </div>
    </footer>
  );
}
