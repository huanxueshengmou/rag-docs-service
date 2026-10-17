"""LangChain 组件版 RAG，用于和手写实现对照。

同一批文档分别走两条链路：
- 手写版：prepare_data → LocalEmbeddings → 余弦排序 → make_messages
- 本文件：TextSplitter → HuggingFaceEmbeddings → Chroma → ChatOpenAI

运行需要额外的可选依赖：pip install -r requirements-baseline.txt
用法：python langchain_baseline.py --question "工具声明会自动执行函数吗？"
"""

import argparse
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_service.config import load_chat_settings
from rag_service.paths import DATA_DIR, MODEL_DIR


def load_texts(data_dir: Path = DATA_DIR) -> list[str]:
    texts = []
    for pattern in ("*.md", "*.txt"):
        for file in Path(data_dir).glob(pattern):
            if file.stem in {"请求", "响应"}:
                continue
            texts.append(file.read_text(encoding="utf-8"))
    return texts


def build_vectorstore():
    texts = load_texts()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = [piece for text in texts for piece in splitter.split_text(text)]

    embeddings = HuggingFaceEmbeddings(model_name=str(MODEL_DIR))
    vectorstore = Chroma.from_texts(chunks, embeddings)
    print(f"加载 {len(texts)} 个文档，切成 {len(chunks)} 块")
    return vectorstore


def answer(question: str, vectorstore, top_k: int = 2) -> str:
    documents = vectorstore.similarity_search(question, k=top_k)
    print(f"检索到 {len(documents)} 块：")
    for number, document in enumerate(documents, start=1):
        preview = document.page_content[:80].replace("\n", " ")
        print(f"  {number}. {preview}...")

    context = "\n\n".join(document.page_content for document in documents)
    settings = load_chat_settings()
    llm = ChatOpenAI(
        model=settings["model"],
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        timeout=settings["timeout"],
        max_tokens=settings["max_tokens"],
    )
    prompt = (
        "根据下面的资料回答问题。资料不足时明确说明“知识库资料不足”，不要编造。\n\n"
        f"{context}\n\n问题：{question}\n回答："
    )
    return llm.invoke(prompt).content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    vectorstore = build_vectorstore()
    print(f"\n问题：{args.question}")
    print(f"\n回答：{answer(args.question, vectorstore, args.top_k)}")


if __name__ == "__main__":
    main()
