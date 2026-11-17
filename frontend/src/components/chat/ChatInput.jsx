import React, { useState, useRef, useEffect } from 'react';
import { Send, Wind } from 'lucide-react';
import { useChat } from '../../context/ChatContext';

const MAX_CHAR_LIMIT = 10000;

export function ChatInput({ onSendMessage, isLoading, onOpenGrounding }) {
  const { sessionId } = useChat();
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef(null);
  const prevSessionIdRef = useRef(sessionId);
  const prevLoadingRef = useRef(isLoading);

  const charCount = inputText.length;
  const isOverLimit = charCount > MAX_CHAR_LIMIT;
  const canSend = inputText.trim().length > 0 && !isOverLimit && !isLoading;

  // 1. Autofocus immediately on mount so cursor is ready for typing or Ctrl+V
  useEffect(() => {
    const timer = setTimeout(() => {
      if (textareaRef.current && !textareaRef.current.disabled) {
        textareaRef.current.focus();
      }
    }, 50);
    return () => clearTimeout(timer);
  }, []);

  // 2. Focus input on New Reflection / session change
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      prevSessionIdRef.current = sessionId;
      setInputText('');
      const timer = setTimeout(() => {
        if (textareaRef.current && !textareaRef.current.disabled) {
          textareaRef.current.focus();
        }
      }, 50);
      return () => clearTimeout(timer);
    }
  }, [sessionId]);

  // 3. Restore focus when an assistant response finishes loading
  useEffect(() => {
    if (prevLoadingRef.current && !isLoading) {
      const timer = setTimeout(() => {
        const active = document.activeElement;
        const isInteractingElsewhere =
          active &&
          active !== document.body &&
          active !== textareaRef.current &&
          active.tagName !== 'BUTTON' &&
          active.tagName !== 'SELECT';
        if (!isInteractingElsewhere && textareaRef.current && !textareaRef.current.disabled) {
          textareaRef.current.focus();
        }
      }, 50);
      return () => clearTimeout(timer);
    }
    prevLoadingRef.current = isLoading;
  }, [isLoading]);

  // Auto-resize textarea dynamically from compact single-line (~24px) up to 130px max
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    textarea.style.height = 'auto';

    if (!inputText) {
      textarea.style.height = '24px';
    } else {
      const scrollHeight = textarea.scrollHeight;
      const targetHeight = Math.min(Math.max(scrollHeight, 24), 130);
      textarea.style.height = `${targetHeight}px`;
    }
  }, [inputText]);

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!canSend) return;

    onSendMessage(inputText.trim());
    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="shrink-0 bg-surface-container-lowest/95 backdrop-blur-md border-t border-surface-container px-3 sm:px-6 md:px-8 lg:px-10 py-2 sm:py-2.5 shadow-[0_-2px_12px_rgba(0,0,0,0.02)] z-20">
      <form onSubmit={handleSubmit} className="flex flex-col gap-1 w-full max-w-5xl xl:max-w-6xl 2xl:max-w-7xl mx-auto">
        {/* Compact WhatsApp-style Composer Bar */}
        <div className="bg-surface-container-low hover:bg-surface-container rounded-2xl p-1.5 sm:p-2 shadow-2xs flex items-end gap-1.5 sm:gap-2 border border-surface-container focus-within:border-primary/40 focus-within:ring-2 focus-within:ring-primary/15 focus-within:bg-surface-container-lowest transition-all min-h-[48px] sm:min-h-[52px]">
          {/* Subtle Grounding Action Icon Button */}
          {onOpenGrounding && (
            <button
              type="button"
              onClick={onOpenGrounding}
              className="w-8 h-8 rounded-xl bg-transparent hover:bg-surface-container-high text-secondary hover:text-primary flex items-center justify-center shrink-0 mb-0.5 transition-colors"
              title="Open Grounding Sanctuary"
              aria-label="Grounding tools"
            >
              <Wind className="w-4 h-4 sm:w-[18px] sm:h-[18px]" />
            </button>
          )}

          {/* Flexible Auto-expanding Textarea */}
          <div className="flex-1 min-w-0 flex items-center py-0.5">
            <textarea
              ref={textareaRef}
              autoFocus={true}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              placeholder="Share whatever you're experiencing right now... take your time."
              rows={1}
              className="w-full bg-transparent resize-none outline-none font-body-md text-sm sm:text-[15px] text-on-surface placeholder:text-on-surface-variant/60 leading-normal max-h-[130px] overflow-y-auto px-1 py-0.5 block disabled:opacity-60"
              aria-label="Message input"
            />
          </div>

          {/* Character counter (only visible when approaching limit) */}
          {charCount > 1000 && (
            <span
              className={`text-[10px] font-mono shrink-0 mb-2 px-1 select-none ${
                isOverLimit ? 'text-error font-semibold' : 'text-on-surface-variant/70'
              }`}
            >
              {charCount > 9000 ? `${charCount}/${MAX_CHAR_LIMIT}` : `${charCount}`}
            </span>
          )}

          {/* Send Action Button */}
          <button
            type="submit"
            disabled={!canSend}
            className={`h-8 sm:h-9 px-3 rounded-xl font-label-md text-xs font-semibold flex items-center justify-center gap-1.5 shrink-0 mb-0.5 shadow-xs transition-all ${
              canSend
                ? 'bg-primary hover:bg-primary-container text-on-primary active:scale-95 cursor-pointer'
                : 'bg-surface-container text-on-surface-variant/40 cursor-not-allowed opacity-60'
            }`}
            aria-label="Send message"
            title="Send (Enter)"
          >
            <span className="hidden sm:inline">Send</span>
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Micro Boundary Notice & Keyboard Hint */}
        <div className="flex items-center justify-between gap-2 px-1 text-on-surface-variant/70 font-label-sm text-[10.5px] select-none">
          <span className="truncate">
            MindBridge provides emotional support, not clinical care. If in acute crisis, tap <span className="font-semibold text-primary">Need Immediate Help</span> above.
          </span>
          <span className="hidden md:inline shrink-0 text-on-surface-variant/50">
            Press <kbd className="px-1 py-0.5 rounded bg-surface-container-high text-on-surface font-mono text-[9.5px]">Enter</kbd> to send • <kbd className="px-1 py-0.5 rounded bg-surface-container-high text-on-surface font-mono text-[9.5px]">Shift+Enter</kbd> for new line
          </span>
        </div>
      </form>
    </div>
  );
}
