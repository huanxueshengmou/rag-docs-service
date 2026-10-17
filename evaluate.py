"""固定评测集：默认只打印样例；--run 发起真实 HTTP 请求。

用法：
    python evaluate.py
    python evaluate.py --run
"""

import argparse
import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
from rag_service.paths import ARTIFACT_DIR, PROJECT_ROOT

CASES_PATH = PROJECT_ROOT / "evaluation_cases.json"
EXPECTED_API_VERSION = "0.3.0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="发起真实请求并保存结果")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()

    definition = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not args.run:
        for case in definition["cases"]:
            print(f"{case['id']}：{case['question']}")
            print(f"  预期：{case['expectation']}")
        print("未发请求；加 --run 才会调用 4 次真实聊天模型。")
        return

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    base = f"http://127.0.0.1:{args.port}"
    destination = ARTIFACT_DIR / "evaluations"
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"{run_id}.json"
    report = {
        "run_id": run_id,
        "actor": "operator_unconfirmed",
        "policy_id": definition["policy_id"],
        "cases": [],
        "note": (
            "status_code 只检查通信；quality_state 需要按 expectation 人工检查，"
            "不能自动按 200 判质量通过。"
        ),
    }

    try:
        with httpx.Client(timeout=90, follow_redirects=False) as client:
            schema = client.get(base + "/openapi.json")
            schema.raise_for_status()
            if schema.json()["info"]["version"] != EXPECTED_API_VERSION:
                raise SystemExit(
                    "运行中的服务版本不符，请重启 python -m rag_service.api"
                )

            for case in definition["cases"]:
                request_data = {"question": case["question"], "top_k": 2}
                if case.get("session_group"):
                    request_data["session_id"] = "eval-" + run_id

                started = datetime.now(UTC).isoformat()
                response = client.post(
                    base + "/ask", json={"requestData": request_data}
                )
                try:
                    body = response.json()
                except json.JSONDecodeError:
                    body = {"unparsed_body": response.text}

                actual = body.get("responseData", {})
                report["cases"].append(
                    {
                        "id": case["id"],
                        "kind": case["kind"],
                        "question": case["question"],
                        "started_at": started,
                        "request_data": request_data,
                        "status_code": response.status_code,
                        "expectation": case["expectation"],
                        "expected_source": case.get("expected_source"),
                        "response": body,
                        "quality_state": "needs_review",
                    }
                )
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                print(f"\n[{case['id']}] HTTP {response.status_code}", flush=True)
                print(f"问题：{case['question']}", flush=True)
                print(f"预期：{case['expectation']}", flush=True)
                print(
                    f"实际回答：{actual.get('answer') or body.get('detail') or '无 answer'}",
                    flush=True,
                )
                print(
                    f"检索来源：{[item.get('source') for item in actual.get('sources', [])]}",
                    flush=True,
                )
                if response.status_code >= 400:
                    print("接口返回错误，停止后续样例；现有记录已保存。", flush=True)
                    break
    except httpx.RequestError as error:
        print(f"请求未完成：{type(error).__name__}")
        report["transport_error"] = type(error).__name__
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n评测记录：{report_path}")
    print("答案是否符合规则需要逐例人工判断，脚本不会自动计为通过。")


if __name__ == "__main__":
    main()
