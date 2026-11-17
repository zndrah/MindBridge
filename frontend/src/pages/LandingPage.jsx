import React from 'react';
import { ArrowRight, ShieldCheck, BookOpen, Wind } from 'lucide-react';

export function LandingPage({ onStartChat, onOpenSafety, onOpenGrounding }) {
  return (
    <div className="flex flex-col w-full animate-fadeIn">
      {/* Key Trust / Boundary Callout Ribbon */}
      <section className="w-full px-4 sm:px-6 lg:px-8 py-3 bg-surface-container-low border-b border-surface-container">
        <div className="w-full max-w-[1560px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3 text-center sm:text-left">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="w-4 h-4 text-secondary shrink-0" />
            <p className="font-body-sm text-xs text-on-surface-variant leading-normal">
              <strong className="font-semibold text-on-surface uppercase tracking-wide">
                First-Response Companion •{' '}
              </strong>
              MindBridge provides non-clinical emotional first response, not psychiatric therapy or medical diagnosis. Sessions are strictly ephemeral in memory.
            </p>
          </div>
          <button
            onClick={onOpenSafety}
            className="shrink-0 text-xs text-primary hover:text-primary-container font-semibold transition-colors inline-flex items-center gap-1.5"
          >
            <span>Read ethics pledge</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </section>

      {/* Editorial Warm Hero */}
      <section className="w-full px-4 sm:px-6 lg:px-8 pt-8 md:pt-14 lg:pt-16 pb-12 md:pb-16 lg:pb-20">
        <div className="w-full max-w-[1560px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10 xl:gap-14 items-center">
          {/* Left Column: Editorial Typography & Tone */}
          <div className="lg:col-span-7 flex flex-col items-start gap-6">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-surface-container text-secondary font-label-md text-xs shadow-xs border border-surface-container-high">
              <span className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
              <span>Safe Harbor • Version 2.0</span>
            </div>

            <h1 className="font-display text-3xl sm:text-4xl md:text-5xl lg:text-[50px] lg:leading-[1.18] text-primary tracking-tight font-serif text-balance">
              A grounded first-response for emotional moments.
            </h1>

            <p className="font-body-lg text-base sm:text-lg text-on-surface-variant max-w-2xl leading-relaxed">
              Empathetic, non-clinical AI support grounded in evidence-backed coping practices when you need to pause, breathe, or reflect. Here to companion your stillness without judgment.
            </p>

            {/* CTA Cluster */}
            <div className="flex flex-wrap items-center gap-3.5 pt-2 w-full sm:w-auto">
              <button
                onClick={onStartChat}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-primary text-on-primary font-semibold text-sm shadow-ambient-sm hover:bg-primary-container transition-all group"
              >
                <span>Start a Reflection</span>
                <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
              </button>

              <button
                onClick={onOpenGrounding}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-full bg-surface-container-high text-secondary font-semibold text-sm hover:bg-surface-container-highest transition-colors"
              >
                <Wind className="w-4 h-4" />
                <span>Explore Grounding Tools</span>
              </button>
            </div>

            {/* Trust Indicators */}
            <div className="flex items-center gap-6 sm:gap-8 pt-6 border-t border-surface-container w-full">
              <div className="flex flex-col">
                <span className="font-headline-sm text-xl sm:text-2xl font-semibold text-primary font-serif">0 ms</span>
                <span className="font-label-sm text-[11px] sm:text-xs text-on-surface-variant">Persistent Session Storage</span>
              </div>
              <div className="w-px h-8 bg-surface-container-highest" />
              <div className="flex flex-col">
                <span className="font-headline-sm text-xl sm:text-2xl font-semibold text-primary font-serif">Audited</span>
                <span className="font-label-sm text-[11px] sm:text-xs text-on-surface-variant">Peer-Reviewed Coping Roots</span>
              </div>
              <div className="w-px h-8 bg-surface-container-highest" />
              <div className="flex flex-col">
                <span className="font-headline-sm text-xl sm:text-2xl font-semibold text-tertiary font-serif">Directory</span>
                <span className="font-label-sm text-[11px] sm:text-xs text-on-surface-variant">Resources Maintained</span>
              </div>
            </div>
          </div>

          {/* Right Column: Serene Editorial Visual Card */}
          <div className="lg:col-span-5 relative w-full">
            <div className="relative rounded-3xl overflow-hidden bg-primary-container text-on-primary shadow-ambient-lg p-6 sm:p-8 flex flex-col justify-between min-h-[300px] border border-surface-container">
              <div className="space-y-2.5">
                <span className="font-label-sm text-[11px] uppercase tracking-wider text-primary-fixed font-semibold">
                  Mindful Anchor
                </span>
                <p className="font-headline-md text-2xl font-serif italic leading-snug">
                  “You do not need to solve the entire storm today. Just stay anchored right now.”
                </p>
              </div>

              <div className="space-y-3.5 pt-6">
                <div className="p-4 rounded-2xl bg-surface-container-lowest/15 backdrop-blur-sm border border-white/10 flex items-center gap-3.5">
                  <div className="w-10 h-10 rounded-full bg-secondary-fixed text-on-secondary-fixed flex items-center justify-center shrink-0">
                    <Wind className="w-5 h-5" />
                  </div>
                  <div>
                    <p className="font-semibold text-xs sm:text-sm text-white">4-7-8 Somatic Rhythm Pacer</p>
                    <p className="text-[11px] sm:text-xs text-on-primary-container">4s Inhale • 7s Stillness • 8s Exhale</p>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-on-primary-container pt-1">
                  <span>Non-Clinical Coping Protocol</span>
                  <span className="font-semibold text-white">Safe Harbor Ready</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Thoughtful Frameworks Highlights */}
      <section className="w-full px-4 sm:px-6 lg:px-8 py-12 md:py-16 lg:py-20 bg-surface-container-low border-t border-surface-container">
        <div className="w-full max-w-[1560px] mx-auto flex flex-col gap-8 sm:gap-10">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
            <div className="max-w-2xl">
              <span className="font-label-md text-xs text-secondary uppercase tracking-widest font-semibold">
                Thoughtful Frameworks
              </span>
              <h2 className="font-headline-lg text-2xl sm:text-3xl md:text-4xl text-primary font-serif font-semibold mt-1.5">
                Practices rooted in nervous system regulation.
              </h2>
            </div>
            <p className="font-body-md text-sm sm:text-base text-on-surface-variant max-w-lg leading-relaxed">
              MindBridge combines evidence-based coping modalities with instant, gentle conversational pacing to guide you through distress.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
            {/* Card 1 */}
            <div className="flex flex-col justify-between bg-surface-container-lowest rounded-2xl p-6 sm:p-7 shadow-ambient-sm border border-surface-container">
              <div>
                <div className="w-12 h-12 rounded-xl bg-primary-fixed flex items-center justify-center text-primary mb-4">
                  <Wind className="w-6 h-6" />
                </div>
                <h3 className="font-headline-sm text-lg font-semibold text-on-surface">
                  Grounded Somatic Exercises
                </h3>
                <p className="font-body-md text-xs sm:text-sm text-on-surface-variant mt-2 leading-relaxed">
                  Step-by-step physical and sensory interventions designed to interrupt panic spirals and stimulate the parasympathetic nervous system.
                </p>
                <div className="mt-4 flex flex-col gap-2 text-xs sm:text-sm text-on-surface">
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
                    <span>Tactile 5-4-3-2-1 Sensory Reset</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
                    <span>4-7-8 Somatic Rhythm & Box Breathing</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
                    <span>Physiological Sigh Cadence</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 pt-3.5 border-t border-surface-container text-xs text-secondary font-semibold">
                Technique Engine
              </div>
            </div>

            {/* Card 2 */}
            <div className="flex flex-col justify-between bg-surface-container-lowest rounded-2xl p-6 sm:p-7 shadow-ambient-sm border border-surface-container">
              <div>
                <div className="w-12 h-12 rounded-xl bg-secondary-fixed flex items-center justify-center text-secondary mb-4">
                  <BookOpen className="w-6 h-6" />
                </div>
                <h3 className="font-headline-sm text-lg font-semibold text-on-surface">
                  Evidence-Backed Knowledge
                </h3>
                <p className="font-body-md text-xs sm:text-sm text-on-surface-variant mt-2 leading-relaxed">
                  Transparent, verifiable insights retrieved through dense vector indexing of established Dialectical Behavior Therapy (DBT TIPP) and cognitive reframing literature.
                </p>
                <div className="mt-4 flex flex-col gap-2 text-xs sm:text-sm text-on-surface">
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
                    <span>DBT TIPP Distress Tolerance</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
                    <span>Interactive Provenance Citations</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary" />
                    <span>Response Hallucination Verification</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 pt-3.5 border-t border-surface-container text-xs text-secondary font-semibold">
                Grounded Retrieval
              </div>
            </div>

            {/* Card 3 */}
            <div className="flex flex-col justify-between bg-surface-container-lowest rounded-2xl p-6 sm:p-7 shadow-ambient-sm border border-surface-container">
              <div>
                <div className="w-12 h-12 rounded-xl bg-tertiary-container flex items-center justify-center text-on-tertiary-container mb-4">
                  <ShieldCheck className="w-6 h-6" />
                </div>
                <h3 className="font-headline-sm text-lg font-semibold text-on-surface">
                  Deterministic Safety Arbitration
                </h3>
                <p className="font-body-md text-xs sm:text-sm text-on-surface-variant mt-2 leading-relaxed">
                  Multi-tier safety architecture combining instantaneous regex override triggers with a specialized RoBERTa crisis classifier to guarantee patient safety.
                </p>
                <div className="mt-4 flex flex-col gap-2 text-xs sm:text-sm text-on-surface">
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
                    <span>Instant Safe Harbor Protocol</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
                    <span>Maintained Crisis Helplines</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-tertiary" />
                    <span>Zero Disguised Clinical Claims</span>
                  </div>
                </div>
              </div>
              <div className="mt-6 pt-3.5 border-t border-surface-container text-xs text-tertiary font-semibold">
                Safety First Architecture
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Non-Clinical Boundaries & Ethics Pledge Section */}
      <section className="w-full px-4 sm:px-6 lg:px-8 py-12 md:py-16 lg:py-20" id="boundaries">
        <div className="w-full max-w-[1560px] mx-auto bg-surface-container-low rounded-3xl p-8 sm:p-10 lg:p-14 border border-surface-container flex flex-col lg:flex-row items-center justify-between gap-8">
          <div className="space-y-3.5 max-w-3xl">
            <span className="font-label-md text-xs text-secondary uppercase tracking-widest font-semibold">
              Our Non-Clinical Boundaries
            </span>
            <h2 className="font-headline-md text-2xl sm:text-3xl text-primary font-serif font-semibold">
              Transparent, honest, and respectful of your limits.
            </h2>
            <p className="font-body-md text-sm sm:text-base text-on-surface-variant leading-relaxed">
              MindBridge does not diagnose mental health conditions, prescribe medication dosages, or impersonate licensed clinicians. If you need crisis support, we immediately provide contact information for crisis resources maintained in the project knowledge base.
            </p>
          </div>
          <button
            onClick={onStartChat}
            className="px-7 py-3 rounded-full bg-primary hover:bg-primary-container text-on-primary font-semibold text-sm shadow-xs transition-colors shrink-0"
          >
            Start Conversation
          </button>
        </div>
      </section>
    </div>
  );
}
