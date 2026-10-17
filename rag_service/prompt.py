"""把检索结果组装成聊天模型消息。"""

import json

from langchain_core.documents import Document

from .policy import build_system_content


def build_context(documents: list[Document]) -> str:
    blocks = [
        f"来源：{document.metadata['source']}\n{document.page_content}"
        for document in documents
    ]
    return "\n\n".join(blocks)


def make_messages(question: str, results: list[dict]) -> tuple[list[dict], list[dict]]:
    """返回 messages 和来源表；来源编号与回答中的 [n] 一一对应。"""
    documents = []
    sources = []
    for number, result in enumerate(results, start=1):
        metadata = result["metadata"]
        sources.append(
            {
                "citation_id": number,
                "source": metadata["source"],
                "chunk_id": result["chunk_id"],
                "start_index": metadata["start_index"],
            }
        )
        documents.append(
            Document(page_content=result["page_content"], metadata=metadata)
        )

    context = build_context(documents)
    source_table = json.dumps(sources, ensure_ascii=False)
    messages = [
        {"role": "system", "content": build_system_content(context, source_table)},
        {"role": "user", "content": question},
    ]
    return messages, sources
