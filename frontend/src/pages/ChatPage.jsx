import React from 'react';
import { useChat } from '../context/ChatContext';
import { ChatWorkspace } from '../components/chat/ChatWorkspace';
import { SafetyResponseView } from '../components/safety/SafetyResponseView';

export function ChatPage({ onOpenAudioModal, onOpenGroundingModal, audioHook }) {
  const { activeSafetyEvent, dismissSafetyEvent } = useChat();

  // If a safety escalation is active, render the contextual Safe Harbor response view
  if (activeSafetyEvent) {
    return (
      <div className="w-full flex-1 overflow-y-auto min-h-0 py-4">
        <SafetyResponseView
          onReturnToChat={dismissSafetyEvent}
          customResponseText={activeSafetyEvent.response}
          triggerText={activeSafetyEvent.triggerMessage}
        />
      </div>
    );
  }

  // Otherwise, render standard conversational workspace
  return (
    <div className="w-full flex-1 flex flex-col overflow-hidden min-h-0">
      <ChatWorkspace
        onOpenAudioModal={onOpenAudioModal}
        onOpenGroundingModal={onOpenGroundingModal}
        audioHook={audioHook}
      />
    </div>
  );
}
