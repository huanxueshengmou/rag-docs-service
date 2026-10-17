"""文档加载、切块与向量索引构建。"""

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .paths import ARTIFACT_DIR

CHUNKS_PATH = ARTIFACT_DIR / "chunks.json"
INDEX_PATH = ARTIFACT_DIR / "local_vector_index.json"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50


def prepare_chunks(
    source_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    inspect: bool = False,
) -> list[dict]:
    """把目录下的 .md/.txt 切成片段，并给每片生成稳定 ID。"""
    source_dir = source_dir.resolve()
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    if not files:
        raise ValueError("数据目录没有 TXT 或 Markdown 文件")

    documents = []
    for path in files:
        loaded = TextLoader(str(path), encoding="utf-8").load()
        for document in loaded:
            # 相对路径作为来源标识，便于回查原文。
            document.metadata["source"] = path.relative_to(source_dir).as_posix()
        documents.extend(loaded)
        if inspect:
            print(f"加载 {path.name}：{len(loaded)} 个 Document")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
    )
    pieces = splitter.split_documents(documents)

    source_counters: dict[str, int] = defaultdict(int)
    records = []
    for piece in pieces:
        source = piece.metadata["source"]
        start = piece.metadata["start_index"]
        if start < 0 or not piece.page_content or len(piece.page_content) > chunk_size:
            raise ValueError(f"片段长度或起始位置异常：{source}:{start}")

        chunk_index = source_counters[source]
        source_counters[source] += 1

        identity = f"{source}:{start}:{piece.page_content}"
        metadata = dict(piece.metadata)
        metadata["chunk_index"] = chunk_index
        records.append(
            {
                "chunk_id": sha256(identity.encode("utf-8")).hexdigest(),
                "page_content": piece.page_content,
                "metadata": metadata,
            }
        )

    if inspect:
        originals = {doc.metadata["source"]: doc.page_content for doc in documents}
        for record in records:
            metadata = record["metadata"]
            text = record["page_content"]
            start = metadata["start_index"]
            intact = originals[metadata["source"]][start : start + len(text)] == text
            print(
                {
                    "source": metadata["source"],
                    "start_index": start,
                    "chars": len(text),
                    "matches_source": intact,
                }
            )
    return records


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError("缺少片段文件，请先运行 python prepare_data.py")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("片段文件为空")
    if len({item["chunk_id"] for item in chunks}) != len(chunks):
        raise ValueError("片段 ID 重复")
    return chunks


def build_or_load_index(embeddings, rebuild: bool = False) -> dict:
    """按内容指纹和模型身份复用索引；任一变化就整组重建。"""
    chunks = load_chunks()
    fingerprint = sha256(
        json.dumps(chunks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if INDEX_PATH.is_file() and not rebuild:
        cached = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if (
            cached.get("content_fingerprint") == fingerprint
            and cached.get("model_identity") == embeddings.model_identity
        ):
            print(f"复用已有索引：{len(cached['records'])} 条片段，未重复编码")
            return cached

    texts = [item["page_content"] for item in chunks]
    vectors = embeddings.embed_documents(texts)
    records = [{**item, "vector": vector} for item, vector in zip(chunks, vectors)]
    index = {
        "model_identity": embeddings.model_identity,
        "content_fingerprint": fingerprint,
        "records": records,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"新建索引：{len(records)} 条片段，维度 {embeddings.dimension}")
    return index
