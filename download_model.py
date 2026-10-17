"""下载固定版本的 BGE 模型到 models/。

只取配置、tokenizer 和 safetensors 权重；运行时完全从本地加载。
"""

import os

from huggingface_hub import snapshot_download
from rag_service.paths import MODEL_DIR

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "special_tokens_map.json",
        ],
        max_workers=3,
    )
    print(f"模型已就位：{model_path}")


if __name__ == "__main__":
    main()
