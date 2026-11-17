import React, { useState } from 'react';
import { ChatProvider } from './context/ChatContext';
import { useAmbientAudio } from './hooks/useAmbientAudio';
import { EmergencyHeaderBanner } from './components/layout/EmergencyHeaderBanner';
import { Navbar } from './components/layout/Navbar';
import { Footer } from './components/layout/Footer';
import { LandingPage } from './pages/LandingPage';
import { ChatPage } from './pages/ChatPage';
import { SafetyProtocolsPage } from './pages/SafetyProtocolsPage';
import { AimlDashboardPage } from './pages/AimlDashboardPage';
import { AudioSanctuaryModal } from './components/grounding/AudioSanctuaryModal';
import { SensoryGroundingModal } from './components/grounding/SensoryGroundingModal';

function AppContent() {
  const [activePage, setActivePage] = useState('landing'); // 'landing' | 'chat' | 'safety'
  const [isAudioModalOpen, setIsAudioModalOpen] = useState(false);
  const [isGroundingModalOpen, setIsGroundingModalOpen] = useState(false);

  // Procedural Web Audio API soundscape hook
  const audioHook = useAmbientAudio();

  const handleStartChat = () => {
    setActivePage('chat');
  };

  const handleTriggerImmediateHelp = () => {
    setActivePage('safety');
  };

  const handleOpenGroundingModal = () => {
    setIsGroundingModalOpen(true);
  };

  const handleOpenAudioModal = () => {
    setIsAudioModalOpen(true);
  };

  return (
    <div className={`min-h-screen flex flex-col bg-surface font-sans text-on-surface ${activePage === 'chat' ? 'h-screen overflow-hidden' : ''}`}>
      {/* Top Banner */}
      <EmergencyHeaderBanner onTriggerEmergency={handleTriggerImmediateHelp} />

      {/* Main Navbar */}
      <Navbar
        activePage={activePage}
        setActivePage={(page) => {
          if (page === 'grounding') {
            setIsGroundingModalOpen(true);
          } else {
            setActivePage(page);
          }
        }}
        onTriggerImmediateHelp={handleTriggerImmediateHelp}
      />

      {/* Main Application Content Area */}
      <main className={`flex-1 w-full ${activePage === 'chat' ? 'flex flex-col overflow-hidden min-h-0' : ''}`}>
        {activePage === 'landing' && (
          <LandingPage
            onStartChat={handleStartChat}
            onOpenSafety={handleTriggerImmediateHelp}
            onOpenGrounding={handleOpenGroundingModal}
          />
        )}

        {activePage === 'chat' && (
          <ChatPage
            onOpenAudioModal={handleOpenAudioModal}
            onOpenGroundingModal={handleOpenGroundingModal}
            audioHook={audioHook}
          />
        )}

        {activePage === 'safety' && (
          <SafetyProtocolsPage onReturnToChat={() => setActivePage('chat')} />
        )}

        {activePage === 'aiml' && (
          <AimlDashboardPage onStartChat={handleStartChat} />
        )}
      </main>

      {/* Accessible Footer */}
      {activePage !== 'chat' && <Footer onNavigatePage={setActivePage} />}

      {/* Modals */}
      <AudioSanctuaryModal
        isOpen={isAudioModalOpen}
        onClose={() => {
          setIsAudioModalOpen(false);
          audioHook.stopAudio();
        }}
        audioHook={audioHook}
      />

      <SensoryGroundingModal
        isOpen={isGroundingModalOpen}
        onClose={() => setIsGroundingModalOpen(false)}
      />
    </div>
  );
}

export default function App() {
  return (
    <ChatProvider>
      <AppContent />
    </ChatProvider>
  );
}
