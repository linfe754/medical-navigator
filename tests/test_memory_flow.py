from app.memory import MemoryManager
from app.router import Intent, Topic, detect_topic, route_message


def test_medicare_followup_keeps_active_topic():
    memory = MemoryManager()
    session_id = "session-1"

    first_topic = detect_topic("Am I eligible for Medicare?")

    assert first_topic == Topic.MEDICARE

    memory.set_topic(session_id, first_topic.value)
    memory.add_turn(
        session_id,
        "Am I eligible for Medicare?",
        "Medicare eligibility depends on your circumstances.",
        topic=first_topic.value,
    )

    followup_topic = detect_topic("What documents do I need?")

    assert followup_topic is None

    active_topic = memory.get_session(session_id).state.current_topic

    assert active_topic == "medicare"

    context = memory.get_context(
        session_id,
        topic=active_topic,
    )

    assert "Am I eligible for Medicare?" in context
    assert "Medicare eligibility" in context


def test_explicit_new_topic_switches_from_medicare_to_transport():
    memory = MemoryManager()
    session_id = "session-1"

    memory.set_topic(session_id, Topic.MEDICARE.value)

    new_topic = detect_topic(
        "How do I get to the Royal Children's Hospital by public transport?"
    )

    assert new_topic == Topic.TRANSPORT

    memory.set_topic(session_id, new_topic.value)

    assert (
        memory.get_session(session_id).state.current_topic
        == "transport"
    )


def test_selected_service_is_remembered():
    memory = MemoryManager()
    session_id = "session-1"

    memory.set_selected_service(
        session_id,
        "The Royal Children's Hospital Melbourne",
        "50 Flemington Rd, Parkville VIC 3052",
    )

    service = memory.get_session(
        session_id
    ).state.selected_service

    assert service is not None
    assert service.name == "The Royal Children's Hospital Melbourne"
    assert service.address == "50 Flemington Rd, Parkville VIC 3052"


def test_sessions_do_not_share_working_memory():
    memory = MemoryManager()

    memory.set_location(
        "session-1",
        "Glenroy",
    )

    memory.set_selected_service(
        "session-1",
        "The Royal Children's Hospital Melbourne",
        "50 Flemington Rd, Parkville VIC 3052",
    )

    session_1 = memory.get_session("session-1")
    session_2 = memory.get_session("session-2")

    assert session_1.state.location == "Glenroy"
    assert session_1.state.selected_service is not None

    assert session_2.state.location is None
    assert session_2.state.selected_service is None


def test_clinical_safety_overrides_active_memory():
    memory = MemoryManager()
    session_id = "session-1"

    memory.set_topic(
        session_id,
        Topic.MEDICARE.value,
    )

    intent = route_message(
        "What medication should I take?"
    )

    assert intent == Intent.CLINICAL

    assert (
        memory.get_session(session_id).state.current_topic
        == "medicare"
    )