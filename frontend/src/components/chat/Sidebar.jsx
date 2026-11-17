import React from 'react';
import { PlusCircle, Trash2, Shield, Sliders, Sparkles, ListChecks, HelpCircle, Volume2, VolumeX, Trees, X } from 'lucide-react';
import { useChat } from '../../context/ChatContext';
import { BreathingOrb } from '../grounding/BreathingOrb';

export function Sidebar({ onOpenAudioModal, audioHook, onClose }) {
  const {
    sessionId,
    startNewSession,
    clearHistory,
    sessionTone,
    setSessionTone,
  } = useChat();

  const { isPlaying, togglePlay } = audioHook;

  const handleStartNew = () => {
    startNewSession();
    if (onClose) onClose();
  };

  const tones = [
    {
      id: 'listening',
      label: 'Gentle Listening',
      desc: 'Soft reflection without rapid advice',
      icon: Sparkles,
      color: 'text-primary',
    },
    {
      id: 'structured',
      label: 'Structured Coping',
      desc: 'Step-by-step somatic micro-steps',
      icon: ListChecks,
      color: 'text-secondary',
    },
    {
      id: 'exploration',
      label: 'Thought Exploration',
      desc: 'Gentle inquiry to untangle loop thoughts',
      icon: HelpCircle,
      color: 'text-tertiary',
    },
  ];

  return (
    <aside className="w-full flex flex-col gap-4 p-space-md">
      {/* Mobile Close Button Header (only visible on mobile drawer) */}
      {onClose && (
        <div className="flex lg:hidden items-center justify-between pb-2 border-b border-surface-container">
          <span className="font-label-md text-xs font-semibold uppercase tracking-wider text-on-surface">
            Session Controls
          </span>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface-variant transition-colors"
            aria-label="Close session controls"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Primary Action & Status Card */}
      <div className="bg-surface-container-lowest p-4 rounded-2xl shadow-xs border border-surface-container flex flex-col gap-3">
        <button
          onClick={handleStartNew}
          className="w-full h-11 bg-primary hover:bg-primary-container text-on-primary font-label-md text-xs font-semibold rounded-xl flex items-center justify-center gap-2 transition-all shadow-xs active:scale-95 group"
          type="button"
        >
          <PlusCircle className="w-4 h-4 transition-transform group-hover:rotate-90" />
          <span>New Reflection</span>
        </button>

        <div className="flex items-center justify-between pt-1 px-1 text-on-surface-variant font-label-sm text-xs">
          <span className="flex items-center gap-1.5" title="Messages are held in temporary memory during the active session">
            <span className="w-2 h-2 rounded-full bg-secondary"></span>
            <span>Ephemeral Session</span>
          </span>
          <button
            onClick={clearHistory}
            className="hover:text-error transition-colors flex items-center gap-1 text-[11px]"
            title="Clear in-memory dialogue turns"
            type="button"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear History</span>
          </button>
        </div>
      </div>

      {/* Session Tone Selector */}
      <div className="bg-surface-container-lowest p-4 rounded-2xl shadow-xs border border-surface-container flex flex-col gap-2.5">
        <div className="flex items-center justify-between">
          <h2 className="font-label-lg text-xs font-semibold text-on-surface uppercase tracking-wider">
            Session Tone
          </h2>
          <Sliders className="w-3.5 h-3.5 text-secondary" />
        </div>
        <p className="font-body-sm text-[11px] text-on-surface-variant leading-snug">
          Adjust how the companion holds space with you:
        </p>

        <div className="flex flex-col gap-1.5 mt-1">
          {tones.map((tone) => {
            const isSelected = sessionTone === tone.id;
            const Icon = tone.icon;
            return (
              <button
                key={tone.id}
                onClick={() => setSessionTone(tone.id)}
                type="button"
                className={`w-full text-left p-2.5 rounded-xl flex items-start gap-2.5 transition-all border ${
                  isSelected
                    ? 'bg-secondary-container/30 border-primary/40 shadow-2xs'
                    : 'bg-surface-container-low/60 hover:bg-surface-container border-transparent'
                }`}
              >
                <div className={`p-1.5 rounded-full bg-surface-container-high ${tone.color} flex items-center justify-center mt-0.5 shrink-0`}>
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span className="font-label-md text-xs text-on-surface font-semibold">
                      {tone.label}
                    </span>
                    <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-primary' : 'bg-transparent'}`} />
                  </div>
                  <p className="font-body-sm text-[11px] text-on-surface-variant truncate">
                    {tone.desc}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Somatic Resonant Breathing Utility Widget */}
      <div className="bg-surface-container-lowest p-4 rounded-2xl shadow-xs border border-surface-container flex flex-col gap-2">
        <BreathingOrb patternKey="478" compact={true} />
      </div>

      {/* Ambient Audio Quick-Toggle Card */}
      <div className="bg-surface-container-lowest p-3.5 rounded-2xl shadow-xs border border-surface-container flex items-center justify-between">
        <button
          onClick={onOpenAudioModal}
          className="flex items-center gap-2.5 text-left group"
          type="button"
        >
          <div className="w-9 h-9 rounded-full bg-surface-container flex items-center justify-center text-secondary group-hover:bg-secondary-fixed transition-colors">
            <Trees className="w-4 h-4" />
          </div>
          <div className="flex flex-col">
            <span className="font-label-md text-xs text-on-surface font-semibold group-hover:text-primary transition-colors">
              Sound Sanctuary
            </span>
            <span className="font-body-sm text-[11px] text-on-surface-variant">
              {isPlaying ? 'Synthesizer Active' : 'Procedural Soundscapes'}
            </span>
          </div>
        </button>

        <button
          onClick={togglePlay}
          className="w-8 h-8 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface flex items-center justify-center transition-colors"
          title={isPlaying ? 'Mute Audio' : 'Play Ambient Audio'}
          type="button"
        >
          {isPlaying ? <Volume2 className="w-4 h-4 text-primary" /> : <VolumeX className="w-4 h-4 text-outline" />}
        </button>
      </div>

      {/* Serene Editorial Vignette */}
      <div className="relative rounded-2xl overflow-hidden shadow-xs bg-primary-container text-on-primary p-4 flex flex-col justify-end min-h-[110px]">
        <div className="relative z-10 space-y-1">
          <span className="font-label-sm text-[10px] uppercase tracking-wider text-primary-fixed font-semibold">
            Gentle Reminder
          </span>
          <p className="font-headline-sm text-xs sm:text-sm font-serif italic leading-snug">
            “You do not have to solve the entire storm in this single breath.”
          </p>
        </div>
      </div>
    </aside>
  );
}
