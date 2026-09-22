import pytest

from app.router import Intent, route_message, is_service_finder_request

@pytest.mark.parametrize(
    "message,expected",
    [
        ("Thanks", Intent.SOCIAL),
        ("Thank you!", Intent.SOCIAL),
        ("谢谢", Intent.SOCIAL),
        ("你好", Intent.SOCIAL),

        ("Tell me a joke", Intent.OUT_OF_SCOPE),
        ("What's the weather today?", Intent.OUT_OF_SCOPE),
        ("帮我讲个笑话", Intent.OUT_OF_SCOPE),
        ("今天天气怎么样？", Intent.OUT_OF_SCOPE),

        ("How do I get a Medicare card?", Intent.NAVIGATION),
        ("Find me a Chinese speaking GP", Intent.NAVIGATION),
        ("Where is the nearest pharmacy?", Intent.NAVIGATION),
        ("我想找附近的医生", Intent.NAVIGATION),

        ("What medication should I take?", Intent.CLINICAL),
        ("What disease do I have?", Intent.CLINICAL),
        ("我肚子痛，是什么病？", Intent.CLINICAL),
        ("我应该吃什么药？", Intent.CLINICAL),

        ("I have severe chest pain", Intent.EMERGENCY),
        ("I can't breathe", Intent.EMERGENCY),
        ("我无法呼吸", Intent.EMERGENCY),

        ("I don't know where to get help", Intent.UNKNOWN),
        ("My mum is unwell and I am not sure where to take her", Intent.UNKNOWN),
        ("Can you help me?", Intent.UNKNOWN),
        ("帮帮我", Intent.UNKNOWN),
    ],
)


def test_route_message(message, expected):
    assert route_message(message) == expected
    
@pytest.mark.parametrize(
    "message",
    [
        "Find a GP near me",
        "Where is the nearest pharmacy?",
        "Find a hospital nearby",
        "I need to find an urgent care clinic",
    ],
)
def test_service_finder_request(message):
    assert is_service_finder_request(message)


@pytest.mark.parametrize(
    "message",
    [
        "How do I get a Medicare card?",
        "Do I need a referral to see a specialist?",
    ],
)
def test_not_service_finder_request(message):
    assert not is_service_finder_request(message)
    
    
from app.router import Topic, detect_topic


def test_detect_medicare_topic():
    assert detect_topic("Am I eligible for Medicare?") == Topic.MEDICARE


def test_detect_service_search_topic():
    assert detect_topic("Find me a GP near Glenroy") == Topic.SERVICE_SEARCH


def test_detect_transport_topic():
    assert detect_topic("How do I get there by train?") == Topic.TRANSPORT


def test_followup_has_no_explicit_topic():
    assert detect_topic("What documents do I need?") is None