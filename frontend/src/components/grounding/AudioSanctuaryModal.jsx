import React from 'react';
import { Play, Pause, Volume2, CloudRain, Waves, Flame, Bell, X, Shield, Wind } from 'lucide-react';
import { BreathingOrb } from './BreathingOrb';

export function AudioSanctuaryModal({ isOpen, onClose, audioHook }) {
  if (!isOpen) return null;

  const { isPlaying, currentTrack, volume, setVolume, togglePlay, changeTrack } = audioHook;

  const soundscapes = [
    { id: 'rain', label: 'Deep Forest Rain', icon: CloudRain, subtitle: 'Filtered pink noise simulating soft pine rainfall' },
    { id: 'ocean', label: 'Pacific Ocean Swell', icon: Waves, subtitle: '12-second rolling tidal oscillation' },
    { id: 'hearth', label: 'Warm Evening Hearth', icon: Flame, subtitle: 'Low frequency resonance & crackling embers' },
    { id: 'bowl', label: 'Soft Singing Bowl', icon: Bell, subtitle: 'Harmonic sine wave acoustic resonance' },
  ];

  const activeSoundscape = soundscapes.find(s => s.id === currentTrack) || soundscapes[0];
  const IconComponent = activeSoundscape.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-inverse-surface/40 backdrop-blur-md animate-fadeIn">
      <div className="bg-surface-container-lowest text-on-surface w-full max-w-[840px] xl:max-w-[920px] rounded-2xl shadow-ambient-lg border border-surface-container overflow-hidden flex flex-col my-auto max-h-[92vh] overflow-y-auto">
        {/* Modal Header */}
        <div className="bg-surface-container-low px-5 py-4 flex items-center justify-between border-b border-surface-container">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-secondary-fixed flex items-center justify-center text-on-secondary-fixed">
              <IconComponent className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="font-headline-sm text-base text-primary font-semibold">
                  MindBridge Audio Sanctuary
                </h2>
                <span className="px-2 py-0.5 rounded-full bg-primary-fixed text-on-primary-fixed-variant text-[10px] font-semibold uppercase tracking-wider">
                  Optional Relaxation
                </span>
              </div>
              <p className="text-xs text-on-surface-variant">Non-clinical ambient relaxation • No therapeutic guarantees</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full hover:bg-surface-container-high text-on-surface-variant flex items-center justify-center transition-colors"
            title="Close modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6">
          {/* Soundscape Selector Pills */}
          <div className="flex flex-wrap items-center gap-2 border-b border-surface-container pb-4">
            {soundscapes.map(track => {
              const isActive = currentTrack === track.id;
              return (
                <button
                  key={track.id}
                  onClick={() => changeTrack(track.id)}
                  className={`px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all ${
                    isActive
                      ? 'bg-primary text-on-primary shadow-xs'
                      : 'bg-surface-container-low hover:bg-surface-container text-on-surface-variant'
                  }`}
                >
                  {track.label}
                </button>
              );
            })}
          </div>

          {/* Central Visualizer Card */}
          <div className="bg-surface-container-low p-6 rounded-2xl flex flex-col sm:flex-row items-center gap-6 border border-surface-container">
            {/* Visualizer Pulsing Core */}
            <div className="relative w-28 h-28 shrink-0 flex items-center justify-center">
              {isPlaying && (
                <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping duration-1000" />
              )}
              <div className="w-24 h-24 rounded-full bg-surface-container-highest flex items-center justify-center border-2 border-primary/30">
                <div className="w-16 h-16 rounded-full bg-secondary-container flex flex-col items-center justify-center text-center p-1">
                  <Wind className="w-4 h-4 text-secondary mb-0.5" />
                  <span className="text-[10px] text-on-secondary-fixed font-semibold leading-tight">
                    {isPlaying ? 'Playing' : 'Muted'}
                  </span>
                </div>
              </div>
            </div>

            {/* Info and Play Control */}
            <div className="flex-1 w-full min-w-0 flex flex-col gap-2 text-center sm:text-left">
              <div className="flex items-center justify-center sm:justify-between">
                <span className="text-xs text-secondary uppercase font-semibold tracking-wider flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-secondary animate-pulse' : 'bg-outline'}`} />
                  {isPlaying ? 'Synthesizer Active' : 'Sound Ready'}
                </span>
                <span className="text-xs text-on-surface-variant hidden sm:inline">
                  Procedural Acoustic Engine
                </span>
              </div>
              <h3 className="font-headline-md text-lg text-on-surface font-semibold">
                {activeSoundscape.label}
              </h3>
              <p className="text-xs text-on-surface-variant leading-relaxed">
                {activeSoundscape.subtitle}
              </p>

              {/* Master Play / Pause Button */}
              <div className="pt-2 flex items-center justify-center sm:justify-start gap-4">
                <button
                  onClick={togglePlay}
                  className="px-6 py-2.5 rounded-full bg-primary hover:bg-primary-container text-on-primary font-semibold text-xs flex items-center gap-2 shadow-xs transition-colors"
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  <span>{isPlaying ? 'Pause Ambient Sound' : 'Play Calming Sound'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Volume and Breathing Controls */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Volume Slider */}
            <div className="bg-surface-container p-4 rounded-xl flex flex-col justify-between gap-3">
              <div className="flex items-center justify-between text-xs font-medium">
                <span className="flex items-center gap-1.5 text-on-surface">
                  <Volume2 className="w-4 h-4 text-secondary" />
                  <span>Master Volume</span>
                </span>
                <span className="text-on-surface-variant font-mono">{Math.round(volume * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="w-full accent-primary h-1.5 bg-surface-container-high rounded cursor-pointer"
                aria-label="Sound volume"
              />
            </div>

            {/* Sync Breathing Orb */}
            <div className="bg-surface-container p-4 rounded-xl flex items-center justify-between">
              <BreathingOrb patternKey="478" compact={true} />
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="bg-surface-container-low px-5 py-3.5 border-t border-surface-container flex flex-col sm:flex-row items-center justify-between gap-2 text-xs">
          <p className="text-on-surface-variant flex items-center gap-1.5 text-center sm:text-left">
            <Shield className="w-3.5 h-3.5 text-secondary shrink-0" />
            <span>Synthesized in-browser soundscape. Runs locally without external audio dependencies.</span>
          </p>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-surface-container hover:bg-surface-container-high text-on-surface font-medium transition-colors"
          >
            Close Sanctuary
          </button>
        </div>
      </div>
    </div>
  );
}
