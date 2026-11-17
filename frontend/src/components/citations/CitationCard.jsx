import React, { useState } from 'react';
import { ChevronDown, ChevronUp, ShieldCheck } from 'lucide-react';

/**
 * Checks if a string is a raw technical chunk identifier
 */
function isTechnicalId(str) {
  if (!str) return true;
  return /^(counselchat_chunk_|chunk_|[a-zA-Z0-9_\-]+__chunk|[a-zA-Z0-9_\-]+:\d+)/i.test(str.trim());
}

/**
 * Resolves human-friendly title for citation card display without exposing technical chunk IDs
 */
function getDisplayTitle(cite) {
  if (cite.title && !isTechnicalId(cite.title)) {
    return cite.title;
  }
  const isCounselChat =
    (cite.document_id && cite.document_id.toLowerCase().includes('counselchat')) ||
    (cite.chunk_id && cite.chunk_id.toLowerCase().includes('counselchat')) ||
    cite.category === 'peer_counseling_example';

  if (isCounselChat) {
    return 'Community Counseling Reference';
  }
  return 'Verified Evidence Reference';
}

export function CitationCard({ citations }) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!citations || citations.length === 0) return null;

  return (
    <div className="mt-2.5 pt-1.5 border-t border-surface-container/40">
      <button
        onClick={() => setIsExpanded(prev => !prev)}
        className="inline-flex items-center justify-between w-full text-left px-2.5 py-1.5 rounded-xl bg-surface-container-low hover:bg-surface-container text-on-surface-variant transition-colors text-xs border border-surface-container/30"
        aria-expanded={isExpanded}
      >
        <span className="inline-flex items-center gap-1.5 font-medium text-secondary">
          <ShieldCheck className="w-3.5 h-3.5 text-secondary" />
          <span>Evidence Provenance ({citations.length} verified {citations.length === 1 ? 'source' : 'sources'})</span>
        </span>
        {isExpanded ? (
          <ChevronUp className="w-3.5 h-3.5 text-outline" />
        ) : (
          <ChevronDown className="w-3.5 h-3.5 text-outline" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-2 p-3 rounded-xl bg-surface-container-lowest border border-surface-container/40 text-xs shadow-2xs">
          <p className="font-medium text-on-surface text-[11px] uppercase tracking-wider text-secondary">
            Grounded Knowledge References:
          </p>
          <div className="divide-y divide-surface-container/40">
            {citations.map((cite, idx) => (
              <div key={idx} className="py-1.5 first:pt-0 last:pb-0 flex flex-col gap-0.5">
                <span className="font-semibold text-on-surface">{getDisplayTitle(cite)}</span>
                <div className="flex items-center gap-2 text-[11px] text-on-surface-variant flex-wrap">
                  <span className="capitalize">Domain: {cite.category ? cite.category.replace(/_/g, ' ') : 'Clinical Coping'}</span>
                  <span>•</span>
                  <span>Doc ID: {cite.document_id}</span>
                  {cite.jurisdiction && (
                    <>
                      <span>•</span>
                      <span>Region: {cite.jurisdiction}</span>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
