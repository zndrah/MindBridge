import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Breathing Patterns:
 * - 4-7-8 Somatic Rhythm: 4s Inhale, 7s Hold, 8s Exhale (19s total)
 * - Box Breathing: 4s Inhale, 4s Hold, 4s Exhale, 4s Hold (16s total)
 */
export const BREATHING_PATTERNS = {
  '478': {
    name: '4-7-8 Somatic Rhythm',
    subtitle: 'Optional self-paced extended exhale exercise (non-clinical)',
    phases: [
      { name: 'Soft Inhale', duration: 4, action: 'inhale' },
      { name: 'Gentle Stillness', duration: 7, action: 'hold' },
      { name: 'Extended Exhale', duration: 8, action: 'exhale' },
    ],
  },
  'box': {
    name: 'Box Breathing (4-4-4-4)',
    subtitle: 'Optional four-stage rhythmic breathing exercise (non-clinical)',
    phases: [
      { name: 'Inhale', duration: 4, action: 'inhale' },
      { name: 'Hold Full', duration: 4, action: 'hold' },
      { name: 'Exhale', duration: 4, action: 'exhale' },
      { name: 'Hold Empty', duration: 4, action: 'hold' },
    ],
  },
};

export function useBreathing(patternKey = '478') {
  const [isActive, setIsActive] = useState(false);
  const [pattern, setPattern] = useState(patternKey);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [secondsRemaining, setSecondsRemaining] = useState(BREATHING_PATTERNS[patternKey].phases[0].duration);
  const [progressPercent, setProgressPercent] = useState(0);

  const timerRef = useRef(null);

  const currentPattern = BREATHING_PATTERNS[pattern] || BREATHING_PATTERNS['478'];
  const currentPhase = currentPattern.phases[phaseIndex] || currentPattern.phases[0];

  const reset = useCallback(() => {
    setIsActive(false);
    setPhaseIndex(0);
    setSecondsRemaining(currentPattern.phases[0].duration);
    setProgressPercent(0);
    if (timerRef.current) clearInterval(timerRef.current);
  }, [currentPattern]);

  const toggle = useCallback(() => {
    setIsActive(prev => !prev);
  }, []);

  useEffect(() => {
    if (!isActive) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    const intervalMs = 100;
    const totalDuration = currentPhase.duration;
    let elapsedMs = 0;

    timerRef.current = setInterval(() => {
      elapsedMs += intervalMs;
      const remainingSec = Math.max(0, Math.ceil(totalDuration - elapsedMs / 1000));
      const progress = Math.min(100, (elapsedMs / (totalDuration * 1000)) * 100);

      setSecondsRemaining(remainingSec);
      setProgressPercent(progress);

      if (elapsedMs >= totalDuration * 1000) {
        // Transition to next phase
        setPhaseIndex(prev => (prev + 1) % currentPattern.phases.length);
        elapsedMs = 0;
      }
    }, intervalMs);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isActive, phaseIndex, currentPattern]);

  const selectPattern = useCallback((newPatternKey) => {
    if (BREATHING_PATTERNS[newPatternKey]) {
      setPattern(newPatternKey);
      setPhaseIndex(0);
      setSecondsRemaining(BREATHING_PATTERNS[newPatternKey].phases[0].duration);
      setProgressPercent(0);
    }
  }, []);

  return {
    isActive,
    pattern,
    currentPattern,
    currentPhase,
    phaseIndex,
    secondsRemaining,
    progressPercent,
    toggle,
    reset,
    selectPattern,
  };
}
