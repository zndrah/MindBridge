import React from 'react';

const PROMPTS = [
  'I am feeling overwhelmed by work today',
  'Can we try a 2-minute somatic grounding exercise?',
  'I cannot stop overthinking about tomorrow',
  'Help me untangle why I feel so anxious right now',
];

export function SuggestedPrompts({ onSelectPrompt }) {
  return (
    <div className="pt-2">
      <p className="font-label-sm text-[11px] text-on-surface-variant uppercase tracking-wider mb-2">
        Suggested Entry Points
      </p>
      <div className="flex flex-wrap gap-2">
        {PROMPTS.map((prompt, index) => (
          <button
            key={index}
            onClick={() => onSelectPrompt(prompt)}
            className="px-3.5 py-1.5 rounded-full bg-surface-container-lowest hover:bg-surface-container-high text-on-surface text-xs font-medium shadow-xs transition-all text-left border border-surface-container"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}
