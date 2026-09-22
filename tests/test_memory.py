from app.memory import MemoryManager


def test_add_and_get_turns():
    memory = MemoryManager()

    memory.add_turn(
        "session-1",
        "Am I eligible for Medicare?",
        "Medicare eligibility depends on your circumstances.",
        topic="medicare",
    )

    turns = memory.get_recent_turns("session-1")

    assert len(turns) == 1
    assert turns[0].user == "Am I eligible for Medicare?"
    assert turns[0].topic == "medicare"


def test_filter_turns_by_topic():
    memory = MemoryManager()

    memory.add_turn(
        "session-1",
        "Am I eligible for Medicare?",
        "Medicare response",
        topic="medicare",
    )

    memory.add_turn(
        "session-1",
        "Find me a GP in Glenroy",
        "GP response",
        topic="service_search",
    )

    turns = memory.get_recent_turns(
        "session-1",
        topic="medicare",
    )

    assert len(turns) == 1
    assert turns[0].user == "Am I eligible for Medicare?"


def test_session_state():
    memory = MemoryManager()

    memory.set_topic("session-1", "service_search")
    memory.set_location("session-1", "Glenroy")
    memory.set_language("session-1", "en")
    memory.set_selected_service(
        "session-1",
        "Example Medical Centre",
        "123 Example Street, Glenroy VIC 3046",
    )

    state = memory.get_session("session-1").state

    assert state.current_topic == "service_search"
    assert state.location == "Glenroy"
    assert state.language == "en"
    assert state.selected_service.name == "Example Medical Centre"
    assert state.selected_service.address == "123 Example Street, Glenroy VIC 3046"


def test_sessions_are_isolated():
    memory = MemoryManager()

    memory.set_location("session-1", "Glenroy")
    memory.set_location("session-2", "Coburg")

    assert memory.get_session("session-1").state.location == "Glenroy"
    assert memory.get_session("session-2").state.location == "Coburg"


def test_clear_session():
    memory = MemoryManager()

    memory.set_location("session-1", "Glenroy")
    memory.clear_session("session-1")

    session = memory.get_session("session-1")

    assert session.state.location is None
    assert len(session.turns) == 0