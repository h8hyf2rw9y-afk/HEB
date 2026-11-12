from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_does_not_fabricate() -> None:
    response = client.post("/chat", json={"question": "¿Cuánto vendimos?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["metrics"]["llm_calls"] == 0
    assert data["evidence"] == []


def test_chat_rejects_empty_question() -> None:
    assert client.post("/chat", json={"question": ""}).status_code == 422


def test_unconfigured_dataset() -> None:
    response = client.post(
        "/query/aggregate",
        json={"dataset": "sales", "metric": "amount"},
    )
    assert response.status_code == 422


def test_rejects_unknown_dataset() -> None:
    response = client.post(
        "/query/aggregate",
        json={"dataset": "unknown", "metric": "amount"},
    )
    assert response.status_code == 422
