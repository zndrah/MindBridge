import React, { useRef, useEffect } from 'react';
import { Sprout, Lock, SlidersHorizontal, HeartHandshake } from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { MessageBubble } from './MessageBubble';
import { TypingIndicator } from './TypingIndicator';
import { SuggestedPrompts } from './SuggestedPrompts';

export function ChatWindow({ onSelectPrompt, onToggleMobileSidebar }) {
  const { messages, isLoading, sessionTone } = useChat();
  const scrollContainerRef = useRef(null);
  const streamEndRef = useRef(null);
  const isNearBottomRef = useRef(true);

  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    isNearBottomRef.current = scrollHeight - scrollTop - clientHeight < 150;
  };

  // Auto-scroll to bottom smoothly when new messages arrive if user is near bottom
  useEffect(() => {
    if (isNearBottomRef.current && streamEndRef.current) {
      streamEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const toneLabels = {
    listening: 'Gentle Listening',
    structured: 'Structured Coping',
    exploration: 'Thought Exploration',
  };

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-surface-dim/30 overflow-hidden relative">
      {/* Chat Area Subheader / Context Bar - Spacious Alignment */}
      <div className="shrink-0 h-13 sm:h-14 px-3 sm:px-6 md:px-8 lg:px-10 bg-surface-container-lowest/90 backdrop-blur-md flex items-center border-b border-surface-container z-20">
        <div className="w-full max-w-5xl xl:max-w-6xl 2xl:max-w-7xl mx-auto flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-8 h-8 rounded-xl bg-primary-fixed text-primary flex items-center justify-center shrink-0 shadow-xs">
              <HeartHandshake className="w-4 h-4" />
            </div>
            <div className="flex flex-col min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="font-headline-sm text-sm sm:text-[15px] font-semibold text-on-surface truncate">
                  MindBridge First-Response Assistant
                </h1>
                <span className="hidden sm:inline-block px-2 py-0.5 rounded-full bg-secondary-container/80 text-on-secondary-container font-label-sm text-[10px] font-semibold uppercase tracking-wider shrink-0">
                  Safe Harbor
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-on-surface-variant font-label-sm text-[11px] truncate">
                <span className="text-secondary font-medium">
                  {toneLabels[sessionTone] || 'Gentle Listening'}
                </span>
                <span>•</span>
                <span className="truncate">Non-clinical emotional companion</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Mobile Sidebar Toggle Button */}
            {onToggleMobileSidebar && (
              <button
                onClick={onToggleMobileSidebar}
                className="lg:hidden flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-surface-container hover:bg-surface-container-high text-on-surface font-label-sm text-xs font-medium transition-colors"
                title="Session controls & tone"
                aria-label="Open session controls"
              >
                <SlidersHorizontal className="w-3.5 h-3.5 text-secondary" />
                <span className="hidden xs:inline">Session Tone</span>
              </button>
            )}

            <span className="hidden md:flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface-container text-on-surface-variant font-label-sm text-[11px]">
              <Lock className="w-3 h-3 text-secondary" />
              <span>Private Session</span>
            </span>
          </div>
        </div>
      </div>

      {/* Message Stream Container - Spacious & Uncramped */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-3 sm:px-6 md:px-8 lg:px-10 pt-4 sm:pt-6 pb-20 sm:pb-24 min-h-0 scroll-smooth"
        id="chatMessageStream"
      >
        <div className="w-full max-w-5xl xl:max-w-6xl 2xl:max-w-7xl mx-auto flex flex-col gap-4 sm:gap-5">
          {/* Welcome Empty State */}
          {messages.length === 0 && (
            <div className="bg-surface-container-lowest p-5 sm:p-6 rounded-2xl flex flex-col gap-4 border border-surface-container/60 shadow-xs animate-fadeIn">
              <div className="flex items-start gap-3.5">
                <div className="w-10 h-10 rounded-2xl bg-primary-fixed text-primary flex items-center justify-center shrink-0">
                  <Sprout className="w-5 h-5" />
                </div>
                <div className="flex flex-col gap-1">
                  <h2 className="font-headline-md text-lg sm:text-xl text-on-surface font-semibold">
                    Take your time. What is on your mind today?
                  </h2>
                  <p className="font-body-md text-sm sm:text-[15px] text-on-surface-variant leading-relaxed">
                    I am here to help you slow down, explore feelings, and find calming perspective. Everything spoken in this window stays entirely within your local browser session and evaporates upon closing.
                  </p>
                </div>
              </div>

              {/* Suggested Prompts */}
              <SuggestedPrompts onSelectPrompt={onSelectPrompt} />
            </div>
          )}

          {/* Render Conversation Turn Bubbles */}
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          {/* Loading Indicator */}
          {isLoading && <TypingIndicator />}

          {/* Anchor spacer ensuring full scroll clearance above fixed composer */}
          <div ref={streamEndRef} className="h-8 shrink-0" aria-hidden="true" />
        </div>
      </div>
    </div>
  );
}
