"""
MindBridge 2.0 - Session Safety State Store (`session_store.py`)
=================================================================

Maintains conversation safety state across consecutive HTTP turns.
Ensures positive confirmations to safety follow-ups escalate directly
to emergency bypass without looping questions or degrading to generic fallback.

State Transitions:
- normal -> pending_safety_follow_up (upon passive death wish / ideation)
- pending_safety_follow_up -> emergency_bypass (upon positive confirmation or explicit suicidal thoughts)
- pending_safety_follow_up -> normal (upon explicit negation or clarification)
- emergency_bypass -> emergency_bypass (crisis escalation remains active across continuations)
"""

from collections import OrderedDict
from datetime import datetime, timezone
import threading
from typing import Any, Dict, Optional

from backend.orchestrator.state import EmergencyTriggerID, JurisdictionCode, SafetyState


class SessionSafetyStore:
    """
    Thread-safe, memory-bounded store for multi-turn safety routing state.
    """

    def __init__(self, max_sessions: int = 5000):
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves stored safety state for a session, updating LRU order."""
        if not session_id:
            return None
        with self._lock:
            state = self._store.get(session_id)
            if state is not None:
                self._store.move_to_end(session_id)
                return dict(state)
            return None

    def get_session_safety_route(self, session_id: str) -> Optional[SafetyState]:
        """Convenience accessor for the current safety route of a session."""
        state = self.get_session_state(session_id)
        if state:
            return state.get("safety_route")
        return None

    def get_session_jurisdiction(self, session_id: str) -> Optional[JurisdictionCode]:
        """Convenience accessor for the stored jurisdiction of a session."""
        state = self.get_session_state(session_id)
        if state:
            return state.get("jurisdiction")
        return None

    def update_session_safety_state(
        self,
        session_id: str,
        safety_route: SafetyState,
        pending_question: Optional[str] = None,
        emergency_trigger_id: Optional[EmergencyTriggerID] = None,
        jurisdiction: Optional[JurisdictionCode] = None,
    ) -> None:
        """
        Updates the active safety state for a session.
        Applies LRU eviction if maximum session threshold is reached.
        """
        if not session_id:
            return

        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            existing = self._store.get(session_id, {})
            updated = {
                "session_id": session_id,
                "safety_route": safety_route,
                "pending_question": pending_question or existing.get("pending_question"),
                "emergency_trigger_id": emergency_trigger_id or existing.get("emergency_trigger_id"),
                "jurisdiction": jurisdiction or existing.get("jurisdiction"),
                "last_updated_utc": now_iso,
                "turns_count": existing.get("turns_count", 0) + 1,
            }

            self._store[session_id] = updated
            self._store.move_to_end(session_id)

            # Evict oldest sessions if capacity exceeded
            while len(self._store) > self._max_sessions:
                self._store.popitem(last=False)

    def clear_session_safety_state(self, session_id: str) -> None:
        """Removes a session from the safety store."""
        if not session_id:
            return
        with self._lock:
            self._store.pop(session_id, None)

    def reset_all(self) -> None:
        """Clears all session safety states (for testing purposes)."""
        with self._lock:
            self._store.clear()


# Global thread-safe singleton
session_safety_store = SessionSafetyStore()
