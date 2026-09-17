from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def ask(message: str) -> str:
    response = client.post("/chat", json={"message": message})
    assert response.status_code == 200
    return response.json()["response"]


def test_out_of_scope_is_blocked():
    response = ask("Tell me a joke")

    assert "Australian healthcare" in response


def test_social_does_not_generate_health_advice():
    response = ask("Thanks")

    assert "Australian health services" in response


def test_clinical_request_is_refused():
    response = ask("What medication should I take?")

    assert "can't provide medical advice" in response
    assert "GP" in response


def test_emergency_request_mentions_000():
    response = ask("I can't breathe")

    assert "000" in response
    assert "Triple Zero" in response


def test_vague_request_is_clarified(monkeypatch):
    from app import main

    monkeypatch.setattr(
        main,
        "classify_unknown",
        lambda message, client: main.Intent.CLARIFY,
    )

    response = ask("I don't know where to get help")

    assert "What do you need help with?" in response