"""项目内的固定路径。"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

# 模型权重不入库；默认放在 models/，也允许用环境变量指向本机已有副本。
MODEL_DIR = Path(
    os.getenv("RAG_MODEL_DIR") or PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
)
