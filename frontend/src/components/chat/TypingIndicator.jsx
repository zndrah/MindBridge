import React from 'react';

export function TypingIndicator() {
  return (
    <div className="flex flex-col items-start gap-1 self-start animate-fadeIn">
      <div className="flex items-center gap-1.5 text-on-surface-variant font-label-sm text-xs pl-2">
        <span className="font-semibold text-primary">MindBridge Companion</span>
      </div>
      <div className="bg-surface-container-low px-4 py-2.5 rounded-2xl shadow-xs flex items-center gap-2.5 border border-surface-container">
        <div className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce"></span>
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.2s]"></span>
          <span className="w-2 h-2 rounded-full bg-primary animate-bounce [animation-delay:0.4s]"></span>
        </div>
        <span className="font-body-sm text-xs text-on-surface-variant font-medium">
          Grounded response generating...
        </span>
      </div>
    </div>
  );
}
