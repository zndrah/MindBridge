import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { ChatWindow } from './ChatWindow';
import { ChatInput } from './ChatInput';
import { useChat } from '../../context/ChatContext';

export function ChatWorkspace({ onOpenAudioModal, onOpenGroundingModal, audioHook }) {
  const { sendMessage, isLoading } = useChat();
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);

  const handleSelectPrompt = (prompt) => {
    sendMessage(prompt);
  };

  return (
    <div className="w-full h-full flex overflow-hidden bg-surface relative min-h-0">
      {/* Desktop Left Sidebar: ~280-300px balanced width */}
      <div className="hidden lg:flex w-[290px] xl:w-[310px] shrink-0 h-full border-r border-surface-container bg-surface-container-low/60 flex-col overflow-y-auto z-10">
        <Sidebar onOpenAudioModal={onOpenAudioModal} audioHook={audioHook} />
      </div>

      {/* Mobile Sidebar Slide-out Drawer */}
      {isMobileSidebarOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-on-surface/40 backdrop-blur-xs transition-opacity"
            onClick={() => setIsMobileSidebarOpen(false)}
            aria-hidden="true"
          />
          {/* Drawer Container */}
          <div className="relative w-[300px] max-w-[85vw] h-full bg-surface-container-lowest shadow-ambient-lg flex flex-col z-10 overflow-y-auto">
            <Sidebar
              onOpenAudioModal={() => {
                setIsMobileSidebarOpen(false);
                onOpenAudioModal();
              }}
              audioHook={audioHook}
              onClose={() => setIsMobileSidebarOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Main Conversational Section */}
      <section className="flex-1 flex flex-col h-full min-h-0 overflow-hidden relative">
        <ChatWindow
          onSelectPrompt={handleSelectPrompt}
          onToggleMobileSidebar={() => setIsMobileSidebarOpen(true)}
        />
        <ChatInput
          onSendMessage={sendMessage}
          isLoading={isLoading}
          onOpenGrounding={onOpenGroundingModal}
        />
      </section>
    </div>
  );
}
