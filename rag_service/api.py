"""FastAPI 入口：POST /ask、GET /health。"""

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import ChatConfigurationError, ChatRequestError
from .service import RagService


class AskData(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(2, ge=1, le=5)
    session_id: str | None = Field(None, min_length=1, max_length=100)
    source: str | None = Field(None, min_length=1, max_length=200)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question 不能只包含空白字符")
        return value

    @field_validator("session_id", "source")
    @classmethod
    def optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("session_id 和 source 不能只包含空白字符")
        return value


class AskRequest(BaseModel):
    requestData: AskData


def create_app(service: RagService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # 测试可以注入替身服务，避免加载真实模型。
        application.state.rag = service if service is not None else RagService()
        yield

    app = FastAPI(title="技术文档问答服务", version="0.3.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(payload: AskRequest):
        data = payload.requestData
        try:
            answer, sources = app.state.rag.answer(
                data.question,
                top_k=data.top_k,
                session_id=data.session_id,
                source=data.source,
            )
        except ChatConfigurationError:
            raise HTTPException(
                status_code=503, detail="聊天模型未配置，请检查项目 .env"
            ) from None
        except ChatRequestError:
            raise HTTPException(
                status_code=502, detail="聊天模型调用失败，请稍后重试"
            ) from None

        return {
            "responseData": {
                "answer": answer,
                "sources": sources,
                "session_id": data.session_id,
                "source": data.source,
            }
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
