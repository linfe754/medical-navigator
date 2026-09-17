import pytest
from fastapi.testclient import TestClient

from app.main import app


pytestmark = pytest.mark.live

client = TestClient(app)


def ask(message: str) -> str:
    response = client.post("/chat", json={"message": message})
    assert response.status_code == 200
    return response.json()["response"]


@pytest.mark.parametrize(
    "message",
    [
        "My mum is unwell and I am not sure where to take her",
        "Where can I get help tonight if my GP is closed?",
        "我妈妈身体不舒服，我不知道应该去哪里看",
    ],
)
def test_navigation_does_not_triage(message):
    response = ask(message).lower()

    forbidden = [
        "appropriate for",
        "suitable for",
        "recommended for",
        "non-urgent care",
        "life-threatening",
        "严重症状",
        "轻微症状",
    ]

    for phrase in forbidden:
        assert phrase not in response


def test_navigation_response_is_brief():
    response = ask("How do I get a Medicare card?")

    assert len(response.split()) <= 120


def test_chinese_navigation_responds_in_chinese():
    response = ask("我想找附近的家庭医生")

    assert any(
        phrase in response
        for phrase in ["医生", "医疗", "诊所", "服务", "家庭"]
    )