import React, { useState, useEffect } from 'react';
import { useChat } from '../context/ChatContext';
import { getEmergencyResources } from '../services/api';
import { SafetyResponseView } from '../components/safety/SafetyResponseView';

export function SafetyProtocolsPage({ onReturnToChat }) {
  const { jurisdiction } = useChat();
  const [resources, setResources] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);

    getEmergencyResources(jurisdiction)
      .then((data) => {
        if (isMounted) {
          setResources(data || []);
          setIsLoading(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setResources([]);
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [jurisdiction]);

  return (
    <div className="w-full py-6 md:py-10">
      <SafetyResponseView
        onReturnToChat={onReturnToChat}
        resources={resources}
        customResponseText={`### Emergency Resources Directory & Safety Protocols
MindBridge 2.0 operates under strict deterministic safety and clinical boundary policies:
- **Universal Emergency Override:** Statements indicating imminent suicidal intent, lethal means, or immediate physical danger instantly bypass open-ended text generation.
- **Emergency Resources Directory:** Crisis support contacts are maintained in the project knowledge base for each supported jurisdiction without fabrication. Operational availability has not been independently verified.
- **Privacy By Design:** Dialogue is held purely in volatile memory. No user utterances or crisis disclosures are stored in persistent databases or browser storage.`}
      />
    </div>
  );
}
