"""回答规则与消息组装。"""

from langchain_core.documents import Document
from rag_service.policy import ANSWER_POLICY, build_system_content
from rag_service.prompt import build_context, make_messages


def test_messages_start_with_policy_and_keep_source_mapping():
    results = [
        {
            "chunk_id": "chunk-1",
            "page_content": "模型提出工具请求，应用负责实际执行。",
            "metadata": {"source": "工具与检索.md", "start_index": 0},
        }
    ]

    messages, sources = make_messages("工具声明会自动执行吗？", results)

    assert messages[0]["content"].startswith(ANSWER_POLICY)
    assert messages[-1] == {"role": "user", "content": "工具声明会自动执行吗？"}
    assert sources == [
        {
            "citation_id": 1,
            "source": "工具与检索.md",
            "chunk_id": "chunk-1",
            "start_index": 0,
        }
    ]
    assert "工具与检索.md" in messages[0]["content"]
    assert "模型提出工具请求，应用负责实际执行。" in messages[0]["content"]


def test_empty_results_produce_empty_source_table():
    messages, sources = make_messages("问题", [])

    assert sources == []
    assert "[]" in messages[0]["content"]


def test_context_keeps_source_name_next_to_each_block():
    documents = [
        Document(page_content="第一段", metadata={"source": "a.md"}),
        Document(page_content="第二段", metadata={"source": "b.txt"}),
    ]

    context = build_context(documents)

    assert "来源：a.md\n第一段" in context
    assert "来源：b.txt\n第二段" in context


def test_build_system_content_without_source_table():
    content = build_system_content("正文")

    assert content.startswith(ANSWER_POLICY)
    assert "此输入未提供编号表" in content
    assert content.endswith("参考资料：\n正文")
