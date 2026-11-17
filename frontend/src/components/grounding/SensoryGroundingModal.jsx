import React, { useState } from 'react';
import { Eye, Hand, Volume2, Sparkles, Coffee, Check, ArrowRight, RotateCcw, X } from 'lucide-react';

const STEPS = [
  {
    step: 5,
    title: '5 Things You Can See',
    prompt: 'Take a slow look around your immediate space. Notice 5 distinct objects, colors, or subtle patterns of light.',
    icon: Eye,
    color: 'bg-primary-fixed text-on-primary-fixed',
    examples: ['A shadow on the wall', 'A grain in the wood', 'A piece of fabric', 'A spot of sunlight', 'A cup on the desk'],
  },
  {
    step: 4,
    title: '4 Things You Can Touch',
    prompt: 'Bring your awareness to physical contact. Notice 4 tactile sensations right now.',
    icon: Hand,
    color: 'bg-secondary-fixed text-on-secondary-fixed',
    examples: ['Your feet firmly resting on the floor', 'The texture of your sleeve', 'The temperature of the room', 'The back of the chair supporting you'],
  },
  {
    step: 3,
    title: '3 Things You Can Hear',
    prompt: 'Soften your gaze and listen attentively. What 3 sounds can your ears detect?',
    icon: Volume2,
    color: 'bg-tertiary-fixed text-on-tertiary-fixed',
    examples: ['A distant vehicle or hum', 'The sound of your breath', 'Leaves rustling or air conditioning'],
  },
  {
    step: 2,
    title: '2 Things You Can Smell',
    prompt: 'Take a gentle nasal breath. Can you notice 2 scents in the air, or recall 2 favorite comforting fragrances?',
    icon: Sparkles,
    color: 'bg-primary-container text-on-primary',
    examples: ['Fresh air or rain', 'A hint of coffee or tea', 'A comforting soap scent', 'The room’s clean atmosphere'],
  },
  {
    step: 1,
    title: '1 Thing You Can Taste',
    prompt: 'Notice the sensation inside your mouth, swallow gently, or take a refreshing sip of water.',
    icon: Coffee,
    color: 'bg-secondary text-on-secondary',
    examples: ['A lingering taste of mint or tea', 'A sip of cool water', 'Just the natural neutral stillness'],
  },
];

export function SensoryGroundingModal({ isOpen, onClose }) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isCompleted, setIsCompleted] = useState(false);

  if (!isOpen) return null;

  const currentStep = STEPS[currentStepIndex];
  const IconComponent = currentStep.icon;

  const handleNext = () => {
    if (currentStepIndex < STEPS.length - 1) {
      setCurrentStepIndex(prev => prev + 1);
    } else {
      setIsCompleted(true);
    }
  };

  const handleReset = () => {
    setCurrentStepIndex(0);
    setIsCompleted(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-inverse-surface/40 backdrop-blur-sm animate-fadeIn">
      <div className="bg-surface-container-lowest text-on-surface w-full max-w-[640px] rounded-2xl shadow-ambient-lg border border-surface-container overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="bg-surface-container-low px-5 py-4 flex items-center justify-between border-b border-surface-container">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full bg-secondary-fixed flex items-center justify-center text-on-secondary-fixed">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h2 className="font-headline-sm text-base text-primary font-semibold">
                5-4-3-2-1 Sensory Grounding
              </h2>
              <p className="text-xs text-on-surface-variant">Optional sensory awareness exercise • Non-clinical</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full hover:bg-surface-container-high text-on-surface-variant flex items-center justify-center transition-colors"
            aria-label="Close modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6">
          {!isCompleted ? (
            <div className="space-y-6">
              {/* Progress Bar */}
              <div className="w-full bg-surface-container-high h-1.5 rounded-full overflow-hidden">
                <div
                  className="bg-primary h-full transition-all duration-300"
                  style={{ width: `${((currentStepIndex + 1) / STEPS.length) * 100}%` }}
                />
              </div>

              {/* Step Card */}
              <div className="flex items-start gap-4">
                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 shadow-xs ${currentStep.color}`}>
                  <IconComponent className="w-6 h-6" />
                </div>
                <div className="space-y-1">
                  <span className="text-xs font-semibold text-secondary uppercase tracking-wider">
                    Step {currentStepIndex + 1} of 5
                  </span>
                  <h3 className="font-headline-md text-xl text-on-surface font-semibold">
                    {currentStep.title}
                  </h3>
                  <p className="text-sm text-on-surface-variant leading-relaxed pt-1">
                    {currentStep.prompt}
                  </p>
                </div>
              </div>

              {/* Gentle Prompt Guidance */}
              <div className="bg-surface-container-low p-4 rounded-xl border border-surface-container text-xs text-on-surface-variant space-y-2">
                <span className="font-semibold text-on-surface text-[11px] uppercase tracking-wider block">
                  Examples to notice:
                </span>
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 list-disc pl-4 text-on-surface">
                  {currentStep.examples.map((ex, idx) => (
                    <li key={idx}>{ex}</li>
                  ))}
                </ul>
              </div>

              {/* Actions */}
              <div className="flex items-center justify-between pt-2">
                <button
                  onClick={handleReset}
                  className="text-xs text-on-surface-variant hover:text-on-surface flex items-center gap-1 font-medium"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Start over</span>
                </button>
                <button
                  onClick={handleNext}
                  className="px-5 py-2.5 rounded-xl bg-primary hover:bg-primary-container text-on-primary text-xs font-semibold flex items-center gap-2 shadow-xs transition-colors"
                >
                  <span>{currentStepIndex === STEPS.length - 1 ? 'Complete Exercise' : 'I Notice These'}</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ) : (
            <div className="text-center py-6 space-y-4">
              <div className="w-16 h-16 rounded-full bg-secondary-container text-on-secondary-container mx-auto flex items-center justify-center">
                <Check className="w-8 h-8" />
              </div>
              <div>
                <h3 className="font-headline-md text-xl font-semibold text-primary">
                  You are anchored in this moment
                </h3>
                <p className="text-sm text-on-surface-variant max-w-sm mx-auto mt-2 leading-relaxed">
                  Take one last gentle breath and notice how your body and surroundings feel right now.
                </p>
              </div>
              <div className="pt-4 flex items-center justify-center gap-3">
                <button
                  onClick={handleReset}
                  className="px-4 py-2 rounded-xl bg-surface-container-high hover:bg-surface-container-highest text-on-surface text-xs font-semibold transition-colors"
                >
                  Repeat Exercise
                </button>
                <button
                  onClick={onClose}
                  className="px-5 py-2 rounded-xl bg-primary hover:bg-primary-container text-on-primary text-xs font-semibold transition-colors"
                >
                  Return to Sanctuary
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
