"""发送真实 HTTP 请求，打印并保存本次往返。

用法：
    python http_client.py --health
    python http_client.py --question "工具声明会自动执行函数吗？"
    python http_client.py --session A --question "我刚才问的是什么？"
"""

import argparse
import json
from datetime import UTC, datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from rag_service.paths import ARTIFACT_DIR

ARCHIVE_DIR = ARTIFACT_DIR / "http_exchanges"


def build_body(args) -> dict | None:
    if args.health:
        return None
    data = {"question": args.question}
    if args.top_k is not None:
        data["top_k"] = args.top_k
    if args.session is not None:
        data["session_id"] = args.session
    if args.source is not None:
        data["source"] = args.source
    return {"requestData": data}


def send(method: str, url: str, body: dict | None) -> tuple[int, str, object]:
    headers = {} if body is None else {"Content-Type": "application/json"}
    encoded = (
        None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    )
    request = Request(url, data=encoded, headers=headers, method=method)

    try:
        try:
            response = urlopen(request, timeout=90)
        except HTTPError as error:
            response = error
        with response:
            status = response.code
            content_type = response.headers.get("Content-Type", "")
            text = response.read().decode("utf-8")
    except OSError:
        raise SystemExit(
            "连接失败：请先在另一个终端启动 python -m rag_service.api"
        ) from None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = text
    return status, content_type, parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", action="store_true", help="GET /health")
    parser.add_argument("--question", default="不同用户的聊天历史为什么会混在一起？")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数 1–5")
    parser.add_argument("--session", default=None, help="会话编号；不填为单轮")
    parser.add_argument("--source", default=None, help="只检索指定来源文件")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    method = "GET" if args.health else "POST"
    path = "/health" if args.health else "/ask"
    url = f"http://127.0.0.1:{args.port}{path}"
    body = build_body(args)

    print(f"请求：{method} {url}")
    print(
        "请求体：" + (json.dumps(body, ensure_ascii=False, indent=2) if body else "无")
    )
    status, content_type, response_body = send(method, url, body)
    print(f"响应状态码：{status}")
    print(f"响应 Content-Type：{content_type}")
    print("响应体：" + json.dumps(response_body, ensure_ascii=False, indent=2))

    trace = {
        "request": {"method": method, "url": url, "path": path, "body": body},
        "response": {
            "status_code": status,
            "headers": {"Content-Type": content_type},
            "body": response_body,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_DIR / "last_http_exchange.json"
    destination.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 每次往返单独留档，连续多次请求不会覆盖前面的记录。
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    (ARCHIVE_DIR / f"{stamp}_{uuid4().hex[:8]}.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"往返记录：{destination}")
    if status >= 400:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
