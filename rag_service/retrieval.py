"""向量检索：先按来源缩小候选，再按余弦相似度取 top-k。"""

from .embeddings import LocalEmbeddings


def select_records(records: list[dict], source: str | None = None) -> list[dict]:
    if source is None:
        return list(records)
    return [record for record in records if record["metadata"]["source"] == source]


def search(
    question: str,
    index: dict,
    embeddings: LocalEmbeddings,
    k: int,
    source: str | None = None,
) -> list[dict]:
    if index["model_identity"] != embeddings.model_identity:
        raise ValueError("查询模型与文档索引不匹配，请重建索引")

    query_vector = embeddings.embed_query(question)
    results = []
    for record in select_records(index["records"], source):
        vector = record["vector"]
        if len(vector) != len(query_vector):
            raise ValueError("向量维度不匹配，请重建索引")

        # 两侧都已归一化，点积等于余弦相似度。
        score = sum(q * v for q, v in zip(query_vector, vector))
        results.append(
            {
                "chunk_id": record["chunk_id"],
                "page_content": record["page_content"],
                "metadata": record["metadata"],
                "score": score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:k]
