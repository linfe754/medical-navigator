from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SelectedService:
    name: str
    address: str


@dataclass
class SessionState:
    current_topic: Optional[str] = None
    location: Optional[str] = None
    selected_service: Optional[SelectedService] = None
    language: str = "en"


@dataclass
class ConversationTurn:
    user: str
    assistant: str
    topic: Optional[str] = None


@dataclass
class SessionMemory:
    turns: deque = field(default_factory=lambda: deque(maxlen=6))
    state: SessionState = field(default_factory=SessionState)


class MemoryManager:
    def __init__(self):
        self._sessions: dict[str, SessionMemory] = defaultdict(SessionMemory)

    def get_session(self, session_id: str) -> SessionMemory:
        return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        user: str,
        assistant: str,
        topic: Optional[str] = None,
    ) -> None:
        session = self.get_session(session_id)
        session.turns.append(
            ConversationTurn(
                user=user,
                assistant=assistant,
                topic=topic,
            )
        )

    def get_recent_turns(
        self,
        session_id: str,
        topic: Optional[str] = None,
    ) -> list[ConversationTurn]:
        session = self.get_session(session_id)
        turns = list(session.turns)

        if topic is None:
            return turns

        return [turn for turn in turns if turn.topic == topic]

    def get_context(
        self,
        session_id: str,
        topic: Optional[str] = None,
    ) -> str:
        turns = self.get_recent_turns(session_id, topic)

        parts = []
        for turn in turns:
            parts.append(f"User: {turn.user}")
            parts.append(f"Assistant: {turn.assistant}")

        return "\n".join(parts)

    def set_topic(self, session_id: str, topic: str) -> None:
        self.get_session(session_id).state.current_topic = topic

    def set_location(self, session_id: str, location: str) -> None:
        self.get_session(session_id).state.location = location

    def set_language(self, session_id: str, language: str) -> None:
        self.get_session(session_id).state.language = language

    def set_selected_service(
        self,
        session_id: str,
        name: str,
        address: str,
    ) -> None:
        self.get_session(session_id).state.selected_service = SelectedService(
            name=name,
            address=address,
        )

    def clear_selected_service(self, session_id: str) -> None:
        self.get_session(session_id).state.selected_service = None

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


memory = MemoryManager()