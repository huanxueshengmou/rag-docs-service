"""会话历史的读取、注入与写回边界。"""

from copy import deepcopy
from threading import Lock

import pytest
from fastapi.testclient import TestClient
from rag_service import api
from rag_service import service as service_module
from rag_service.service import RagService


@pytest.fixture
def session_service(monkeypatch, tmp_path):
    monkeypatch.setattr(service_module, "LocalEmbeddings", lambda: object())
    monkeypatch.setattr(
        service_module, "build_or_load_index", lambda embeddings: {"records": []}
    )
    monkeypatch.setattr(
        service_module,
        "search",
        lambda question, index, embeddings, k, source=None: [],
    )

    rag = RagService()
    rag.trace_dir = tmp_path
    inputs = []

    def fake_chat(messages):
        inputs.append(deepcopy(messages))
        return "替身回复：" + messages[-1]["content"]

    monkeypatch.setattr(service_module, "call_chat", fake_chat)
    return rag, inputs


def post(client, question, session_id=None):
    body = {"question": question}
    if session_id is not None:
        body["session_id"] = session_id
    return client.post("/ask", json={"requestData": body})


def test_two_sessions_get_only_their_own_history(session_service):
    rag, inputs = session_service

    with TestClient(api.create_app(rag)) as client:
        assert post(client, "A的首轮问题", "A").status_code == 200
        assert post(client, "B的首轮问题", "B").status_code == 200
        response = post(client, "我刚才问了什么？", "A")
        assert response.status_code == 200
        assert response.json()["responseData"]["session_id"] == "A"
        assert post(client, "我刚才问了什么？", "B").status_code == 200

    assert [message["role"] for message in inputs[2]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert inputs[2][1]["content"] == "A的首轮问题"
    assert inputs[3][1]["content"] == "B的首轮问题"
    assert len(rag.histories["A"]) == len(rag.histories["B"]) == 4
    assert [message["role"] for message in rag.histories["A"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert sum(message["content"] == "我刚才问了什么？" for message in inputs[2]) == 1


def test_no_session_stays_stateless(session_service):
    rag, inputs = session_service

    with TestClient(api.create_app(rag)) as client:
        assert post(client, "第一轮").status_code == 200
        assert post(client, "第二轮").status_code == 200

    assert [len(messages) for messages in inputs] == [2, 2]
    assert rag.histories == {}


def test_failed_chat_does_not_save_a_partial_turn(session_service, monkeypatch):
    rag, _ = session_service

    with TestClient(api.create_app(rag)) as client:
        assert post(client, "成功的第一轮", "A").status_code == 200
        before = deepcopy(rag.histories["A"])

        def fail(messages):
            raise service_module.ChatRequestError("替身上游失败")

        monkeypatch.setattr(service_module, "call_chat", fail)
        assert post(client, "失败的第二轮", "A").status_code == 502

    assert rag.histories["A"] == before


@pytest.mark.parametrize("session_id", ["", "   ", "x" * 101, 123])
def test_invalid_session_rejected_before_chat(session_service, session_id):
    rag, inputs = session_service

    with TestClient(api.create_app(rag)) as client:
        response = post(client, "问题", session_id)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "requestData", "session_id"]
    assert inputs == []


def test_overlapping_turns_in_one_session_are_serialized(session_service, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    rag, _ = session_service
    input_lengths = []
    lengths_lock = Lock()

    def slow_chat(messages):
        with lengths_lock:
            input_lengths.append(len(messages))
        time.sleep(0.03)
        return "并发替身回复"

    monkeypatch.setattr(service_module, "call_chat", slow_chat)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(rag.answer, question, 2, "A") for question in ["一", "二"]
        ]
        for future in futures:
            future.result(timeout=5)

    assert input_lengths == [2, 4]
    assert len(rag.histories["A"]) == 4
