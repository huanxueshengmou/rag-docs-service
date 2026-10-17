# P1 架构与逻辑

图示与 `rag_service/` 下的模块一一对应。Mermaid 可直接在 Obsidian、GitHub、Typora 中渲染。

## 1. 系统全景

```mermaid
flowchart TB
    subgraph client["客户端"]
        HTTP["http_client.py"]
        TESTS["pytest / TestClient"]
        EVAL["evaluate.py"]
    end

    subgraph api["接口层"]
        APP["rag_service/api.py<br/>POST /ask · GET /health"]
        REQ["AskData 校验<br/>question / top_k / session_id / source"]
    end

    subgraph core["业务层"]
        SVC["service.py<br/>RagService"]
        RET["retrieval.py<br/>search / select_records"]
        PRM["prompt.py<br/>build_context / make_messages"]
        POL["policy.py<br/>kb-only 规则"]
    end

    subgraph infra["基础设施层"]
        EMB["embeddings.py<br/>LocalEmbeddings"]
        IDX["indexing.py<br/>prepare_chunks / build_or_load_index"]
        LLM["llm.py<br/>call_chat"]
        CFG["config.py<br/>CHAT_*"]
        ART[("artifacts/")]
    end

    HTTP --> APP
    EVAL --> APP
    TESTS --> APP
    APP --> REQ
    APP --> SVC
    SVC --> RET
    SVC --> PRM
    SVC --> LLM
    SVC --> ART
    RET --> EMB
    RET --> IDX
    PRM --> POL
    IDX --> EMB
    IDX --> ART
    LLM --> CFG
```

## 2. 模块依赖

```mermaid
flowchart LR
    API["api.py"] --> SERVICE["service.py"]
    SERVICE --> RETRIEVAL["retrieval.py"]
    SERVICE --> PROMPT["prompt.py"]
    SERVICE --> LLM["llm.py"]
    SERVICE --> INDEXING["indexing.py"]
    SERVICE --> POLICY["policy.py"]
    PROMPT --> POLICY
    RETRIEVAL --> EMBEDDINGS["embeddings.py"]
    INDEXING --> EMBEDDINGS
    LLM --> CONFIG["config.py"]
    EMBEDDINGS --> PATHS["paths.py"]
    INDEXING --> PATHS
    SERVICE --> PATHS
```

依赖方向单向向下：接口层不直接读文件，检索层不关心 HTTP，模型调用只依赖配置。

## 3. 离线索引流程

```mermaid
flowchart LR
    FILES["data/*.md<br/>data/*.txt"] --> LOADER["TextLoader<br/>保留 source"]
    LOADER --> SPLIT["RecursiveCharacterTextSplitter<br/>300 字符 / 50 重叠"]
    SPLIT --> ID["chunk_id = sha256(source:start:text)"]
    ID --> CHUNKS[("chunks.json")]

    CHUNKS --> FINGERPRINT["内容指纹<br/>sha256(排序后的 chunks)"]
    CHUNKS --> ENCODE["embed_documents<br/>BGE · CLS · L2"]
    ENCODE --> VECTORS["512 维向量"]

    FINGERPRINT --> INDEX[("local_vector_index.json")]
    VECTORS --> INDEX
    INDEX --> CACHE{"指纹和 model_identity<br/>都与当前一致？"}
    CACHE -->|是| REUSE["直接复用，跳过编码"]
    CACHE -->|否| REBUILD["整组重建索引"]
```

`indexing.py` 把“正文是否变化”和“编码模型是否变化”分开校验：内容指纹覆盖正文、来源、起始位置；
`model_identity` 覆盖模型 ID、revision、权重 sha256、维度和池化方式。

## 4. 在线问答时序

```mermaid
sequenceDiagram
    autonumber
    participant C as 客户端
    participant A as FastAPI
    participant S as RagService
    participant E as LocalEmbeddings
    participant X as 向量索引
    participant M as 聊天模型

    C->>A: POST /ask {question, top_k, session_id, source}
    A->>A: Pydantic 校验（失败返回 422）
    A->>S: answer(question, top_k, session_id, source)
    alt session_id 存在
        S->>S: 取该会话的历史与锁
    else 单轮
        S->>S: history = []
    end
    S->>E: embed_query(question)
    E-->>S: 查询向量
    S->>X: select_records(source) + 余弦排序
    X-->>S: top-k 片段
    S->>S: make_messages（规则 + 正文 + 来源表）
    S->>M: system + history + user
    M-->>S: answer
    S->>S: 成功后才追加本轮历史
    S-->>A: answer, sources
    A-->>C: responseData
```

失败路径：配置缺失 `503`，上游调用失败 `502`；两种情况都不会写入会话历史。

## 5. 会话隔离与锁

```mermaid
flowchart TB
    REQ["POST /ask"] --> HAS{"session_id?"}
    HAS -->|无| SINGLE["单轮：history = []"]
    HAS -->|有| REG["registry_lock<br/>取出或创建 histories[id] 与 session_locks[id]"]
    REG --> LOCK["持有该会话的 Lock"]
    LOCK --> BUILD["messages = [system, *history, user]"]
    SINGLE --> BUILD
    BUILD --> RETRIEVE["retrieval_lock<br/>串行执行本地编码与检索"]
    RETRIEVE --> CALL["call_chat<br/>等待模型时释放 retrieval_lock"]
    CALL -->|成功| APPEND["history += [user, assistant]"]
    CALL -->|失败| SKIP["不写入，向上抛 ChatRequestError"]
    APPEND --> RESP["返回 answer + sources"]
    SKIP --> RESP
```

锁的粒度：

| 锁 | 作用范围 | 目的 |
| --- | --- | --- |
| `retrieval_lock` | 全进程 | CPU 编码与全量排序串行，避免多线程争用 |
| `session_lock` | 单个 session | 同一会话的“读历史 → 调模型 → 写历史”原子化 |
| `registry_lock` | 全进程 | 保护 `histories` / `session_locks` 字典的创建 |

## 6. 检索细节

```mermaid
flowchart TB
    Q["question"] --> INSTR["加查询前缀<br/>为这个句子生成表示以用于检索相关文章："]
    INSTR --> ENCODE["BGE 编码 + L2 归一化"]
    ENCODE --> V["查询向量 q"]

    V --> CHECK{"index.model_identity<br/>== embeddings.model_identity?"}
    CHECK -->|否| ERR["ValueError：请重建索引"]
    CHECK -->|是| FILTER{"source 指定?"}

    FILTER -->|是| KEEP["只保留 metadata.source == source"]
    FILTER -->|否| ALL["保留全部记录"]
    KEEP --> DOT["score = Σ q[i] × v[i]"]
    ALL --> DOT
    DOT --> SORT["按 score 降序"]
    SORT --> TOP["取前 k 条"]
    TOP --> OUT["chunk_id + page_content + metadata + score"]
```

来源过滤发生在排序之前，因此 `source` 会真正缩小候选集合，而不是先取全局 top-k 再过滤。

## 7. 提示词结构

```mermaid
flowchart TB
    subgraph messages["发送给模型的消息"]
        SYS["system<br/>kb-only 规则 + 来源表 + 参考资料"]
        HIST["history<br/>user / assistant 交替（可选）"]
        USER["user<br/>当前问题"]
    end
    SYS --> MODEL["聊天模型"]
    HIST --> MODEL
    USER --> MODEL
    MODEL --> ANSWER["回答中的 [n] 对应 sources[n-1]"]
```

`policy.py` 的规则明确：资料不足时只能说明不足，不能补通用知识；引用必须来自本轮资料。
`prompt.py` 的 `build_context` 在每段正文前标注来源文件名，减少模型把不同文档的内容混在一起。

## 8. 错误边界

| 场景 | 检测位置 | HTTP 状态 | 是否进入检索/模型 |
| --- | --- | --- | --- |
| 空问题、超长问题、`top_k` 越界、空 `session_id`/`source` | Pydantic 校验 | 422 | 否 |
| `CHAT_*` 配置不完整 | `config.py` | 503 | 检索可能已完成，模型未调用 |
| 聊天模型网络/额度错误 | `llm.py` | 502 | 检索完成，模型调用失败 |
| 模型身份与索引不一致 | `retrieval.py` | 500 | 未编码查询，未调用模型 |
| 片段文件缺失或损坏 | `indexing.py` | 启动失败 | 服务不启动 |

## 9. 评测流程

```mermaid
flowchart LR
    CASES["evaluation_cases.json<br/>4 个固定样例"] --> SCRIPT["evaluate.py --run"]
    SCRIPT --> API["POST /ask"]
    API --> RECORD["artifacts/evaluations/{run_id}.json"]
    RECORD --> HUMAN["人工按 expectation 判定"]
    HUMAN --> STATE["quality_state<br/>passed / failed"]
    RECORD -.->|HTTP 200 只代表通信成功| NOTE["不能自动判质量"]
```

评测样例覆盖两类失败模式：

1. **资料支持**：问题在知识库中有明确答案，检查引用是否来自正确文件。
2. **资料不足**：问题超出知识库范围，检查模型是否拒答而不是补通用知识。

第 4 个样例依赖同一会话的前一轮历史，用来验证会话读写在真实 HTTP 链路中生效。
