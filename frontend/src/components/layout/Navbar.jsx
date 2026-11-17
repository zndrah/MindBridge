import React from 'react';
import { ShieldAlert, Globe, Activity, HeartHandshake } from 'lucide-react';
import { useChat } from '../../context/ChatContext';

export function Navbar({ activePage, setActivePage, onTriggerImmediateHelp }) {
  const { jurisdiction, setJurisdiction, backendStatus } = useChat();

  const navItems = [
    { id: 'landing', label: 'Home' },
    { id: 'chat', label: 'Chat Assistant' },
    { id: 'aiml', label: 'AIML Dashboard' },
    { id: 'safety', label: 'Safety Protocols' },
    { id: 'grounding', label: 'Grounding Tools' },
  ];

  return (
    <header className="sticky top-0 left-0 right-0 z-40 bg-surface/90 backdrop-blur-md border-b border-surface-container shadow-ambient-sm shrink-0">
      <div className="h-16 px-4 md:px-6 lg:px-8 flex items-center justify-between w-full max-w-[1600px] mx-auto gap-2">
        {/* Brand & Subtitle */}
        <div className="flex items-center gap-space-md">
          <button
            onClick={() => setActivePage('landing')}
            className="flex items-center gap-2.5 text-left focus:outline-none group"
            aria-label="MindBridge Home"
          >
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center text-primary-fixed shadow-sm group-hover:bg-primary-container transition-colors">
              <HeartHandshake className="w-5 h-5 text-on-primary" />
            </div>
            <div className="flex flex-col">
              <span className="font-headline-sm text-headline-sm text-primary tracking-tight font-semibold leading-none">
                MindBridge
              </span>
              <span className="font-label-sm text-[10px] text-on-surface-variant uppercase tracking-wider mt-0.5">
                2.0 Safe Harbor
              </span>
            </div>
          </button>

          {/* Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-1 ml-4" aria-label="Main Navigation">
            {navItems.map((item) => {
              const isActive = activePage === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActivePage(item.id)}
                  className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-surface-container-high text-primary font-semibold shadow-xs'
                      : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Right Action Cluster */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Jurisdiction Dropdown */}
          <div className="relative flex items-center bg-surface-container-low rounded-lg px-2 py-1 border border-surface-container">
            <Globe className="w-3.5 h-3.5 text-secondary mr-1 shrink-0" />
            <select
              value={jurisdiction}
              onChange={(e) => setJurisdiction(e.target.value)}
              className="bg-transparent text-xs font-semibold text-on-surface outline-none cursor-pointer pr-1"
              title="Regional Emergency Jurisdiction"
              aria-label="Regional Jurisdiction"
            >
              <option value="IN">India (IN)</option>
              <option value="US">United States (US)</option>
              <option value="UK">United Kingdom (UK)</option>
              <option value="CA">Canada (CA)</option>
              <option value="AU">Australia (AU)</option>
              <option value="GL">Global Resources (GL)</option>
            </select>
          </div>

          {/* Backend Health Dot */}
          <div
            className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-full bg-surface-container-low text-xs text-on-surface-variant"
            title={backendStatus.isReady ? 'Backend AI Models Ready' : 'Backend Connecting / Degraded'}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendStatus.isReady ? 'bg-secondary animate-pulse' : 'bg-tertiary'
              }`}
            />
            <span className="text-[11px] font-medium hidden lg:inline">
              {backendStatus.isReady ? 'System Online' : 'Connecting'}
            </span>
          </div>

          {/* Need Immediate Help CTA Button */}
          <button
            onClick={onTriggerImmediateHelp}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-error-container text-on-error-container font-medium text-xs hover:bg-error hover:text-on-error transition-all shadow-ambient-sm active:scale-95 shrink-0"
            aria-label="Need Immediate Help"
          >
            <ShieldAlert className="w-4 h-4 text-error shrink-0" />
            <span className="font-semibold tracking-tight">Need Immediate Help?</span>
          </button>
        </div>
      </div>
    </header>
  );
}
