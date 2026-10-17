"""把 data/ 下的文档切块，生成 artifacts/chunks.json。

用法：
    python prepare_data.py
    python prepare_data.py --inspect
"""

import argparse
import json
from hashlib import sha256
from pathlib import Path

from rag_service.indexing import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    prepare_chunks,
)
from rag_service.paths import ARTIFACT_DIR, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "chunks.json")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--inspect", action="store_true", help="打印加载与切块细节，不调用模型"
    )
    args = parser.parse_args()

    records = prepare_chunks(
        args.source,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        inspect=args.inspect,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sources = {record["metadata"]["source"] for record in records}
    print(f"源文件 {len(sources)} 个，片段 {len(records)} 个")
    print(f"输出：{args.output}")
    if args.inspect:
        ids = "\n".join(record["chunk_id"] for record in records)
        print(f"ID 清单摘要：{sha256(ids.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
