"""RAG 服务：检索、会话历史、消息组装与模型调用。"""

import json
from threading import Lock
from uuid import uuid4

from .config import ChatConfigurationError, ChatRequestError
from .embeddings import LocalEmbeddings
from .indexing import build_or_load_index
from .llm import call_chat
from .paths import ARTIFACT_DIR
from .policy import ANSWER_POLICY_ID
from .prompt import make_messages
from .retrieval import search

__all__ = [
    "ChatConfigurationError",
    "ChatRequestError",
    "RagService",
]


class RagService:
    """启动时加载一次模型和索引；每次请求只处理当前问题。"""

    def __init__(self):
        self.embeddings = LocalEmbeddings()
        self.index = build_or_load_index(self.embeddings)

        # 本地 CPU 编码串行执行；等待模型时释放该锁。
        self.retrieval_lock = Lock()

        # 每个 session 一份历史和一把锁，避免同一会话的并发轮次互相覆盖。
        self.histories: dict[str, list[dict]] = {}
        self.session_locks: dict[str, Lock] = {}
        self.registry_lock = Lock()
        self.trace_dir = ARTIFACT_DIR

    def answer(
        self,
        question: str,
        top_k: int = 2,
        session_id: str | None = None,
        source: str | None = None,
    ) -> tuple[str, list[dict]]:
        if session_id is None:
            # 不传 session_id 就是单轮问答，不与其他请求共享历史。
            return self._answer_turn(question, top_k, [], None, source)

        history, session_lock = self._session_state(session_id)
        with session_lock:
            return self._answer_turn(question, top_k, history, session_id, source)

    def _session_state(self, session_id: str) -> tuple[list[dict], Lock]:
        with self.registry_lock:
            self.histories.setdefault(session_id, [])
            self.session_locks.setdefault(session_id, Lock())
            return self.histories[session_id], self.session_locks[session_id]

    def _answer_turn(
        self,
        question: str,
        top_k: int,
        history: list[dict],
        session_id: str | None,
        source: str | None,
    ) -> tuple[str, list[dict]]:
        with self.retrieval_lock:
            results = search(
                question, self.index, self.embeddings, k=top_k, source=source
            )

        base_messages, sources = make_messages(question, results)
        messages = [base_messages[0], *history, base_messages[1]]
        self._save_trace(
            {
                "answer_policy": ANSWER_POLICY_ID,
                "session_id": session_id,
                "question": question,
                "top_k": top_k,
                "source": source,
                "history_message_count": len(history),
                "retrieval": results,
                "messages": messages,
                "sources": sources,
            },
            session_id,
        )

        answer = call_chat(messages)
        if session_id is not None:
            # 只有模型成功返回才写入本轮，失败不会留下半轮历史。
            history.extend([base_messages[1], {"role": "assistant", "content": answer}])
        return answer, sources

    def _save_trace(self, trace: dict, session_id: str | None) -> None:
        """保存本次模型输入，便于回查检索结果与提示词。"""
        serialized = json.dumps(trace, ensure_ascii=False, indent=2)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "last_api_model_input.json").write_text(
            serialized, encoding="utf-8"
        )
        _write_unique(self.trace_dir / "api_inputs", serialized)
        if session_id is not None:
            (self.trace_dir / "last_session_model_input.json").write_text(
                serialized, encoding="utf-8"
            )
            _write_unique(self.trace_dir / "session_inputs", serialized)


def _write_unique(directory, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{uuid4().hex}.json").write_text(content, encoding="utf-8")
