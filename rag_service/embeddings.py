"""本地 BGE 中文向量模型。

检索是高频且隐私敏感的一步，因此文档和问题都在本机 CPU 上编码。权重版本
和哈希会写进 model_identity，索引与查询必须使用同一份模型配置。
"""

from hashlib import sha256
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .paths import MODEL_DIR

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
BATCH_SIZE = 8
MAX_LENGTH = 512


class LocalEmbeddings:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        weights = self.model_dir / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(
                f"本地模型不存在：{weights}。请先运行 python download_model.py"
            )

        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.model.eval()
        self.dimension = self.model.config.hidden_size

        self.model_identity = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": sha256(weights.read_bytes()).hexdigest(),
            "dimension": self.dimension,
            "pooling": "CLS",
            "normalize": True,
            "max_length": MAX_LENGTH,
            "query_instruction": QUERY_INSTRUCTION,
        }

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            with torch.inference_mode():
                # BGE 取 CLS 向量；L2 归一化后，点积即余弦相似度。
                hidden = self.model(**tokens).last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(hidden, p=2, dim=1)
            vectors.extend(normalized.tolist())
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([QUERY_INSTRUCTION + question])[0]
