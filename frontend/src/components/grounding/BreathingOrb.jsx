import React from 'react';
import { Wind, Play, Pause, RotateCcw } from 'lucide-react';
import { useBreathing, BREATHING_PATTERNS } from '../../hooks/useBreathing';

export function BreathingOrb({ patternKey = '478', compact = false }) {
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false);

  React.useEffect(() => {
    if (typeof window !== 'undefined' && window.matchMedia) {
      const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
      setPrefersReducedMotion(mediaQuery.matches);
      const handler = (e) => setPrefersReducedMotion(e.matches);
      mediaQuery.addEventListener('change', handler);
      return () => mediaQuery.removeEventListener('change', handler);
    }
  }, []);

  const {
    isActive,
    pattern,
    currentPattern,
    currentPhase,
    secondsRemaining,
    progressPercent,
    toggle,
    reset,
    selectPattern,
  } = useBreathing(patternKey);

  // SVG parameters
  const size = compact ? 70 : 120;
  const strokeWidth = compact ? 4 : 6;
  const radius = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (progressPercent / 100) * circumference;

  // Scale animation for central orb based on phase (disabled under reduced motion)
  let orbScale = 'scale-100';
  if (isActive && !prefersReducedMotion) {
    if (currentPhase.action === 'inhale') {
      orbScale = 'scale-125 transition-transform duration-1000';
    } else if (currentPhase.action === 'hold') {
      orbScale = 'scale-125 transition-none';
    } else if (currentPhase.action === 'exhale') {
      orbScale = 'scale-90 transition-transform duration-1000';
    }
  }

  return (
    <div className={`flex flex-col gap-3 ${compact ? 'items-start' : 'items-center text-center'}`}>
      <div className="flex items-center gap-3">
        {/* SVG Circle Progress */}
        <div className="relative flex items-center justify-center shrink-0" style={{ width: size, height: size }}>
          <svg className="transform -rotate-90" width={size} height={size}>
            <circle
              className="text-surface-container-highest"
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke="currentColor"
              strokeWidth={strokeWidth}
            />
            <circle
              className="text-primary transition-all duration-100 ease-linear"
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke="currentColor"
              strokeWidth={strokeWidth}
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
            />
          </svg>

          {/* Central Pulsing Orb */}
          <div
            className={`absolute rounded-full bg-secondary-fixed text-on-secondary-fixed flex items-center justify-center shadow-xs ${orbScale}`}
            style={{ width: compact ? 30 : 50, height: compact ? 30 : 50 }}
          >
            <Wind className={compact ? 'w-3.5 h-3.5' : 'w-5 h-5'} />
          </div>
        </div>

        {/* Phase Details & Seconds */}
        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-2">
            <span className={`font-headline-sm font-semibold text-primary leading-tight ${compact ? 'text-sm' : 'text-base'}`}>
              {isActive ? currentPhase.name : 'Breathing Pacer'}
            </span>
            {isActive && (
              <span className="text-xs font-mono font-bold text-secondary bg-secondary-container px-1.5 py-0.5 rounded">
                {secondsRemaining}s
              </span>
            )}
          </div>
          <p className="text-xs text-on-surface-variant leading-tight mt-0.5">
            {currentPattern.name}
          </p>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center gap-2">
        <button
          onClick={toggle}
          className={`px-3 py-1 rounded-full text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs ${
            isActive
              ? 'bg-error-container text-on-error-container hover:bg-error hover:text-on-error'
              : 'bg-primary-fixed text-on-primary-fixed-variant hover:bg-primary-container hover:text-on-primary'
          }`}
        >
          {isActive ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
          <span>{isActive ? 'Pause' : 'Start Rhythm'}</span>
        </button>

        {isActive && (
          <button
            onClick={reset}
            className="p-1 rounded-full bg-surface-container hover:bg-surface-container-high text-on-surface-variant transition-colors"
            title="Reset"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        )}

        {/* Pattern switcher */}
        {!compact && (
          <select
            value={pattern}
            onChange={(e) => selectPattern(e.target.value)}
            className="bg-surface-container-low text-xs rounded-lg px-2 py-1 text-on-surface outline-none cursor-pointer border border-surface-container"
          >
            <option value="478">4-7-8 Somatic Rhythm</option>
            <option value="box">Box Breathing (4-4-4-4)</option>
          </select>
        )}
      </div>
    </div>
  );
}
