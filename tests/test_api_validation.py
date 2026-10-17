"""接口边界、错误码与参数传递；不调用真实模型。"""

from threading import Lock

import pytest
from fastapi.testclient import TestClient
from rag_service import api
from rag_service import service as service_module
from rag_service.config import ChatConfigurationError, ChatRequestError
from rag_service.service import RagService


class StubRag:
    def __init__(self, error=None):
        self.questions = []
        self.top_ks = []
        self.session_ids = []
        self.sources = []
        self.error = error

    def answer(self, question, top_k=2, session_id=None, source=None):
        self.questions.append(question)
        self.top_ks.append(top_k)
        self.session_ids.append(session_id)
        self.sources.append(source)
        if self.error:
            raise self.error
        return "测试用回答", [
            {
                "citation_id": 1,
                "source": "测试资料.txt",
                "chunk_id": "test-chunk",
                "start_index": 0,
            }
        ]


def test_post_passes_question_and_returns_sources():
    stub = StubRag()

    with TestClient(api.create_app(stub)) as client:
        response = client.post(
            "/ask", json={"requestData": {"question": "  中文问题  "}}
        )

    assert response.status_code == 200
    assert stub.questions == ["中文问题"]
    assert stub.top_ks == [2]
    assert response.json()["responseData"]["sources"][0]["source"] == "测试资料.txt"


def test_optional_fields_reach_service():
    stub = StubRag()

    with TestClient(api.create_app(stub)) as client:
        response = client.post(
            "/ask",
            json={
                "requestData": {
                    "question": "问题",
                    "session_id": "A",
                    "source": "工具与检索.md",
                }
            },
        )

    assert response.status_code == 200
    assert stub.session_ids == ["A"]
    assert stub.sources == ["工具与检索.md"]


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"requestData": {}},
        {"requestData": {"question": ""}},
        {"requestData": {"question": "   "}},
        {"requestData": {"question": 123}},
        {"requestData": {"question": "字" * 2001}},
    ],
)
def test_invalid_input_never_reaches_service(body):
    stub = StubRag()

    with TestClient(api.create_app(stub)) as client:
        response = client.post("/ask", json=body)

    assert response.status_code == 422
    assert stub.questions == []


@pytest.mark.parametrize(
    "error,status",
    [
        (ChatConfigurationError("测试配置错误"), 503),
        (ChatRequestError("测试上游错误"), 502),
    ],
)
def test_model_failure_is_mapped_to_http_error(error, status):
    with TestClient(api.create_app(StubRag(error))) as client:
        response = client.post("/ask", json={"requestData": {"question": "问题"}})

    assert response.status_code == status
    assert "responseData" not in response.json()


@pytest.mark.parametrize("top_k", [1, 3, 5])
def test_valid_top_k_reaches_service(top_k):
    stub = StubRag()

    with TestClient(api.create_app(stub)) as client:
        response = client.post(
            "/ask", json={"requestData": {"question": "问题", "top_k": top_k}}
        )

    assert response.status_code == 200
    assert stub.top_ks == [top_k]


@pytest.mark.parametrize(
    "top_k,error_type",
    [(0, "greater_than_equal"), (6, "less_than_equal"), (1.5, "int_from_float")],
)
def test_invalid_top_k_identifies_the_field(top_k, error_type):
    stub = StubRag()

    with TestClient(api.create_app(stub)) as client:
        response = client.post(
            "/ask", json={"requestData": {"question": "问题", "top_k": top_k}}
        )

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "requestData", "top_k"]
    assert error["type"] == error_type
    assert stub.questions == []


def test_http_top_k_and_source_control_the_search_call(monkeypatch, tmp_path):
    rag = object.__new__(RagService)
    rag.index = {"records": []}
    rag.embeddings = object()
    rag.retrieval_lock = Lock()
    rag.histories = {}
    rag.session_locks = {}
    rag.registry_lock = Lock()
    rag.trace_dir = tmp_path

    observed = []
    results = [{"page_content": "命中正文"}]

    def capture_search(question, index, embeddings, k, source=None):
        observed.append((question, k, source))
        return results

    def capture_messages(question, candidates):
        assert candidates is results
        return [
            {"role": "system", "content": "测试规则"},
            {"role": "user", "content": question},
        ], []

    monkeypatch.setattr(service_module, "search", capture_search)
    monkeypatch.setattr(service_module, "make_messages", capture_messages)
    monkeypatch.setattr(service_module, "call_chat", lambda messages: "本地替身回答")

    with TestClient(api.create_app(rag)) as client:
        for body in [
            {"question": "默认"},
            {"question": "指定", "top_k": 3},
            {"question": "过滤", "source": "工具与检索.md"},
        ]:
            response = client.post("/ask", json={"requestData": body})
            assert response.status_code == 200

    assert observed == [
        ("默认", 2, None),
        ("指定", 3, None),
        ("过滤", 2, "工具与检索.md"),
    ]


def test_health_and_method_guard():
    with TestClient(api.create_app(StubRag())) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ask").status_code == 405
