
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pymilvus.exceptions import MilvusException

from app import main
from app.router import Intent
from app.services import rag_service


client = TestClient(main.app)


def test_milvus_failure_returns_no_chunks(monkeypatch):
    monkeypatch.setattr(rag_service, "RAG_RETRIEVER", "hybrid")

    retriever = MagicMock()
    retriever.retrieve.side_effect = MilvusException(
        code=2,
        message="Milvus unavailable",
    )
    monkeypatch.setattr(
        rag_service,
        "get_hybrid_retriever",
        lambda: retriever,
    )

    assert rag_service.retrieve_chunks("How do I enrol in Medicare?") == []


@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream"])
def test_rag_fallback_skips_generation(monkeypatch, endpoint):
    monkeypatch.setattr(
        main,
        "classify_intent",
        lambda *args, **kwargs: (Intent.NAVIGATION, "test"),
    )
    monkeypatch.setattr(main, "resolve_topic", lambda *args: None)
    monkeypatch.setattr(
        main,
        "is_transport_request",
        lambda *args: False,
    )
    monkeypatch.setattr(
        main,
        "is_service_finder_request",
        lambda *args: False,
    )
    monkeypatch.setattr(
        main,
        "prepare_navigation_input",
        lambda **kwargs: {
            "model_input": "Test question",
            "sources": [],
            "rag_used": False,
        },
    )

    llm = MagicMock()
    monkeypatch.setattr(main, "get_openai_client", lambda: llm)

    response = client.post(
        endpoint,
        json={
            "message": "How do I enrol in Medicare?",
            "session_id": f"fallback-{endpoint}",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": main.RAG_FALLBACK_MESSAGE,
        "sources": [],
        "rag_used": False,
    }
    llm.responses.create.assert_not_called()
    llm.responses.stream.assert_not_called()


def test_successful_rag_generates_grounded_answer(monkeypatch):
    monkeypatch.setattr(
        main,
        "classify_intent",
        lambda *args, **kwargs: (Intent.NAVIGATION, "test"),
    )
    monkeypatch.setattr(main, "resolve_topic", lambda *args: None)
    monkeypatch.setattr(
        main,
        "is_transport_request",
        lambda *args: False,
    )
    monkeypatch.setattr(
        main,
        "is_service_finder_request",
        lambda *args: False,
    )

    sources = [{"citation_id": "S1", "title": "Medicare guide"}]
    monkeypatch.setattr(
        main,
        "prepare_navigation_input",
        lambda **kwargs: {
            "model_input": "Retrieved reference documents: Medicare guide",
            "sources": sources,
            "rag_used": True,
        },
    )

    llm = MagicMock()
    llm.responses.create.return_value.output_text = "Medicare enrolment guidance"
    llm.responses.create.return_value.usage = None
    monkeypatch.setattr(main, "get_openai_client", lambda: llm)

    response = client.post(
        "/chat",
        json={
            "message": "How do I enrol in Medicare?",
            "session_id": "rag-success-test",
        },
    )

    assert response.status_code == 200
    assert response.json()["response"] == "Medicare enrolment guidance"
    assert response.json()["rag_used"] is True
    assert response.json()["sources"] == sources
    llm.responses.create.assert_called_once()
