"""本地检索，可选调用聊天模型生成回答。

用法：
    python ask.py --question "工具声明会自动执行函数吗？"
    python ask.py --question "..." --answer
"""

import argparse
import json
from pathlib import Path

from rag_service.config import ChatConfigurationError, ChatRequestError
from rag_service.embeddings import LocalEmbeddings
from rag_service.indexing import build_or_load_index
from rag_service.llm import call_chat
from rag_service.paths import ARTIFACT_DIR
from rag_service.prompt import make_messages
from rag_service.retrieval import search

DEFAULT_QUESTION = "模型给出 tool_calls 后，应用还需要做什么？"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--source", default=None, help="只检索指定来源文件")
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量索引")
    parser.add_argument(
        "--answer", action="store_true", help="检索后调用聊天模型生成回答"
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    if args.top_k < 1:
        raise SystemExit("--top-k 必须大于 0")
    if not args.question.strip():
        raise SystemExit("--question 不能为空")

    embeddings = LocalEmbeddings()
    index = build_or_load_index(embeddings, rebuild=args.rebuild)
    results = search(args.question, index, embeddings, args.top_k, source=args.source)

    print(f"\n问题：{args.question}")
    if args.source is not None:
        print(f"来源过滤：{args.source}")
    print("全量余弦排序，分数越高越靠前（不是回答正确率）：")
    for number, result in enumerate(results, start=1):
        metadata = result["metadata"]
        preview = result["page_content"][:90].replace("\n", " ")
        print(
            f"[{number}] {metadata['source']} "
            f"score={result['score']:.4f} "
            f"chunk_id={result['chunk_id'][:12]}"
        )
        print(f"    {preview}")

    messages, sources = make_messages(args.question, results)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "last_retrieval.json").write_text(
        json.dumps(
            {"question": args.question, "source": args.source, "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "last_rag_model_input.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n检索记录：{ARTIFACT_DIR / 'last_retrieval.json'}")
    print(f"模型消息：{ARTIFACT_DIR / 'last_rag_model_input.json'}")

    if not args.answer:
        print("未调用聊天模型；加 --answer 生成回答。")
        return

    try:
        answer = call_chat(messages, args.env_file, args.model, args.base_url)
    except (ChatConfigurationError, ChatRequestError) as error:
        raise SystemExit(str(error)) from None

    print(f"\n模型回答：\n{answer}")
    (ARTIFACT_DIR / "last_rag_answer.json").write_text(
        json.dumps(
            {"question": args.question, "answer": answer, "sources": sources},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
