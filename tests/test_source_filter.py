"""来源过滤与检索边界。"""

import pytest
from rag_service.retrieval import search, select_records


class StubEmbeddings:
    """固定查询向量，只检验候选范围与排序，不依赖真实模型。"""

    def __init__(self):
        self.model_identity = {"stub": True}

    def embed_query(self, question):
        return [1.0, 0.0]


def records():
    return [
        {
            "chunk_id": "a",
            "page_content": "甲1",
            "metadata": {"source": "工具与检索.md", "start_index": 0},
            "vector": [0.5, 0.0],
        },
        {
            "chunk_id": "b",
            "page_content": "甲2",
            "metadata": {"source": "工具与检索.md", "start_index": 10},
            "vector": [0.4, 0.0],
        },
        {
            "chunk_id": "c",
            "page_content": "乙1",
            "metadata": {"source": "对话记忆.txt", "start_index": 0},
            "vector": [0.9, 0.0],
        },
    ]


def test_none_keeps_every_candidate():
    assert select_records(records(), None) == records()


def test_matching_source_keeps_only_that_source():
    kept = select_records(records(), "工具与检索.md")

    assert [item["chunk_id"] for item in kept] == ["a", "b"]
    assert kept[0]["page_content"] == "甲1"
    assert kept[0]["metadata"]["source"] == "工具与检索.md"
    assert kept[0]["vector"] == [0.5, 0.0]


def test_unknown_source_returns_empty():
    assert select_records(records(), "不存在的.md") == []


def test_filter_happens_before_top_k():
    index = {"model_identity": {"stub": True}, "records": records()}
    embeddings = StubEmbeddings()

    unfiltered = search("问题", index, embeddings, k=1)
    assert [item["chunk_id"] for item in unfiltered] == ["c"]

    filtered = search("问题", index, embeddings, k=1, source="工具与检索.md")
    assert [item["chunk_id"] for item in filtered] == ["a"]


def test_filter_with_no_match_returns_empty():
    index = {"model_identity": {"stub": True}, "records": records()}

    assert search("问题", index, StubEmbeddings(), k=2, source="不存在.md") == []


def test_model_identity_mismatch_is_rejected():
    index = {"model_identity": {"stub": "other"}, "records": records()}

    with pytest.raises(ValueError):
        search("问题", index, StubEmbeddings(), k=1)
