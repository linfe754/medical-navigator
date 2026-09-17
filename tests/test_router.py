import pytest

from app.router import Intent, route_message


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