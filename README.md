# AstroBot - 生产级电商智能客服系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Milvus](https://img.shields.io/badge/Milvus-Lite-blueviolet.svg)](https://milvus.io/)
[![MCP](https://img.shields.io/badge/MCP-1.3%2B-green.svg)](https://modelcontextprotocol.io/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.17%2B-lightgrey.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **AstroBot** 是一个经过 10 个迭代阶段演进、面向企业级电商落地场景的端到端智能客服系统。系统融合了 **FastAPI 异步流式框架**、**LangGraph 确定性工作流与 ReAct Agent 协同编排**、**高阶混合检索 RAG**、**三层动态 Token 上下文管理**、**MCP (Model Context Protocol) 开放协议工具生态**、**Langfuse 全链路可观测性与数据飞轮**，以及**全参微调的中文 17 类多标签主题分类器与 ONNX 旁路推理架构**。

---

## 目录 (Table of Contents)

- [核心特性](#-核心特性)
- [系统架构图](#-系统架构图)
- [十章演进全景路线](#-十章演进全景路线)
- [技术栈概览](#-技术栈概览)
- [项目目录结构](#-项目目录结构)
- [快速开始](#-快速开始)
  - [1. 环境准备](#1-环境准备)
  - [2. 环境变量配置](#2-环境变量配置)
  - [3. 一键智能启动与自愈](#3-一键智能启动与自愈)
  - [4. 启动旁路与微服务 (可选)](#4-启动旁路与微服务-可选)
- [Web 控制台与功能路由导航](#-web-控制台与功能路由导航)
- [实证验收与质量保障](#-实证验收与质量保障)
- [开源协议](#-开源协议)

---

## 🌟 核心特性

### 1. 多轮流式会话与电商专业人设
- **SSE (Server-Sent Events) 打字机响应**：毫秒级首字出字，保障极佳的用户交互体验。
- **电商客服人设约束**：具备严格的话术规范与拒识机制，对敏感、越权、竞品攻击等问题主动防御。
- **售后工单自动化提取**：支持从自然语言对话中结构化提取订单号、问题类型、诉求与凭据。

### 2. LangGraph 确定性工作流与 Agent 协同
- **意图核心分流器 (Intent Router)**：基于语义判断用户意图，精准分流至确定性售后流水线或知识问答 Agent。
- **确定性退货退款子流程**：针对高敏感业务（如退货、换货、退款）采用有限状态机（FSM）严格闭环执行，杜绝大模型随机幻觉。
- **ReAct 动态调度 Agent**：针对复杂开放场景，支持动态多步规划、多工具连续调用与结果整合。

### 3. 高精度 RAG 进阶检索体系
- **密集与稀疏混合检索**：结合 **BM25** 关键词匹配与 **BGE-M3** 密集语义向量检索。
- **RRF 倒数排名融合 (Reciprocal Rank Fusion)**：多路召回打分动态归一化，攻克错别字、专有名词与口语表达。
- **BGE-Reranker 二次重排与置信度硬闸**：引入 Cross-Encoder 深度重排打分，配置置信度硬闸门（默认 0.40），低于阈值自动降级并流转至人工飞轮，杜绝一本正经胡说八道。

### 4. 三层上下文工程与 Token 动态预算
- **三层递进式管理**：
  - **L1 基础层**：系统 Prompt 与工具契约固定开销。
  - **L2 短期层**：最近 $N$ 轮对话精确滑动窗口。
  - **L3 长期层**：历史对话自适应异步摘要压缩，保留关键业务实体与槽位。
- **Token 动态预算与自适应裁剪**：基于模型上下文窗口自动计算安全余量，防止超长上下文导致调用崩溃或成本激增。

### 5. 即插即用工具体系与 MCP 协议集成
- **开放 Model Context Protocol (MCP)**：将物流轨迹追踪、售后工单系统等核心外部能力抽离为独立标准 MCP 服务。
- **工具权限沙箱与审计**：具备完整的工具白名单校验、调用前后参数审计与安全沙箱执行机制。

### 6. 全链路可观测性与数据飞轮
- **Langfuse 深度集成**：完整记录用户会话 Trace、Span、Agent 思考步数与 LLM 原始输入输出。
- **Token 消耗与成本透视**：实时追踪 Input/Output Tokens 及单次会话美元/人民币成本。
- **数据飞轮闭环 (Data Flywheel)**：自动将检索低置信度、用户负向反馈及异常会话流转至人工审核队列（Review Queue），沉淀为高质量测试集与知识库增量。

### 7. 中文多标签分类器全参微调与 ONNX 独立服务
- **17 类电商权威类目体系**：退换货、物流、尺码、发票、保修、价保、优惠等 17 类权威术语，定义清晰的近邻边界（如“修归保修、退归退换；运费管钱、物流管货”）。
- **RoBERTa-wwm-ext 全参微调**：采用 `BCEWithLogitsLoss` 解决一句话多诉求的多标签分类问题。
- **ONNX Runtime 独立推理服务 (`:8110`)**：动态轴导出，去除 PyTorch/Transformers 重依赖，实现低延迟轻量推理。
- **旁路批处理归类**：主链路无感旁路批处理 (`classify_pool.py`)，自动聚合飞轮问题并生成主题分布看板。

### 8. 九项实证验收体系与白名单作业系统
- 包含**三份考卷零泄漏硬闸**、**三档分级 F1 容错红线**、**错因三向平衡记账（漏打/多打/错位）**、**阈值 9 线复算重演**等 9 大维度的实证验收体系。
- 提供 Web 端可视化看板及白名单安全作业运行器 (`/api/jobs`)，支持一键就地复验。

---

## 🏗️ 系统架构图

```mermaid
flowchart TB
    subgraph ClientLayer ["用户端 & 前端管理"]
        User["用户 (Web / H5 / App)"]
        Admin["客服管理员 / 运营人员"]
    end

    subgraph GatewayLayer ["接入与网关 (FastAPI)"]
        MainApp["FastAPI 主服务 (main.py)"]
        SSEStream["SSE 打字机流式通道"]
        ContextMgr["三层上下文管理 & Token预算"]
    end

    subgraph CoreBrain ["核心编排与认知大脑"]
        Router["意图核心分流器 (Intent Router)"]
        RefundFSM["确定性售后状态机 (Refund Workflow)"]
        ReActAgent["ReAct 动态智能体 (Agent Loop)"]
    end

    subgraph ServiceMesh ["服务与工具网络"]
        RAGSystem["混合 RAG 检索 (BM25 + BGE-M3 + Rerank)"]
        MilvusDB[(Milvus-Lite 向量库)]
        MySQLDB[(MySQL 8.0 业务库)]
        MCPLogistics["物流 MCP 服务 (:8001)"]
        MCPAftersale["售后 MCP 服务 (:8002)"]
    end

    subgraph FlywheelAndML ["数据飞轮与微调模型旁路"]
        Langfuse["Langfuse 可观测性追踪"]
        ReviewQueue["低置信度审核池 (Review Queue)"]
        BatchClassifier["旁路批处理 (classify_pool.py)"]
        ONNXService["ONNX 独立分类推理服务 (:8110)"]
        TopicDashboard["17类主题分布看板"]
    end

    %% 主对话链路
    User -->|SSE 请求| MainApp
    MainApp --> SSEStream
    MainApp --> ContextMgr
    ContextMgr --> Router

    Router -->|售后确定性意图| RefundFSM
    Router -->|复杂咨询/通用问答| ReActAgent

    RefundFSM -->|读写业务| MySQLDB
    RefundFSM -->|标准化工具| MCPAftersale

    ReActAgent -->|知识查询| RAGSystem
    RAGSystem --> MilvusDB
    ReActAgent -->|物流跟踪| MCPLogistics
    ReActAgent -->|售后处理| MCPAftersale

    %% 观测与飞轮链路
    MainApp -.->|全链路追踪| Langfuse
    RAGSystem -.->|低置信度触发| ReviewQueue
    ReviewQueue --> Admin
    ReviewQueue -.-> BatchClassifier
    BatchClassifier -->|轻量推理请求| ONNXService
    BatchClassifier -->|沉淀主题标签| MySQLDB
    MySQLDB --> TopicDashboard
    TopicDashboard --> Admin
```

---

## 🗺️ 十章演进全景路线

本项目历经 10 个章节的完整研发沉淀，每个章节均有详尽的开发记录（详见 [`dev-notes/`](dev-notes/)）：

| 章节 | 主题 | 核心成果与突破 |
|:---|:---|:---|
| **Ch01** | **纯对话跑通** | 搭建 FastAPI + LangChain 基础框架，打通 SSE 打字机流式长连接与基础 Web 对话页面。 |
| **Ch02** | **Function Calling 工具链** | 实现大模型结构化提取售后工单，接入 MySQL 8.0 存储，跑通订单查询与售后申请工具链路。 |
| **Ch03** | **RAG 基础与知识库向量检索** | 引入 Milvus-Lite 向量库与 BGE-M3 语义向量模型，构建客服 FAQ 知识库与基础语义检索。 |
| **Ch04** | **RAG 进阶 · 混合检索与评估** | 落地 BM25 + 密集向量双路召回、RRF 融合算法与 BGE-Reranker 二次重排，建立 Hit@K / MRR 评估体系。 |
| **Ch05** | **Workflow 确定性编排与 Agent** | 引入 LangGraph 状态图，区分规则驱动的确定性流程与大模型驱动的 ReAct 决策，解决幻觉失控。 |
| **Ch06** | **核心分流器与退款子流程** | 完善意图分流路由器，构建标准退货退款有限状态机，支持多轮槽位补全与人机协同交接。 |
| **Ch07** | **三层会话上下文管理** | 研发 L1 人设/L2 滑动窗口/L3 长期摘要三层上下文体系，引入启动自检与动态 Token 预算保护机制。 |
| **Ch08** | **即插即用 MCP 协议工具体系** | 将物流与售后系统改造为标准 MCP 协议微服务，实现工具注册解耦、参数校验与安全审计。 |
| **Ch09** | **可观测性监控与数据飞轮** | 接入 Langfuse 实现 Token 级成本透视，上线低置信度会话拦截与人工审核队列（Review Queue）。 |
| **Ch10** | **多标签分类器微调与实证验收** | 基于 RoBERTa-wwm-ext 微调 17 类电商多标签模型，导出 ONNX 并搭建独立服务 (`:8110`)，上线九项实证验收体系。 |

---

## 💻 技术栈概览

| 模块类别 | 核心技术选型 | 说明 |
|:---|:---|:---|
| **Web 框架** | [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) | 高性能异步 ASGI 框架，支持 SSE 流式响应 |
| **大模型框架** | [LangChain](https://www.langchain.com/), [LangGraph](https://github.com/langchain-ai/langgraph) | 智能体状态图编排、Prompt 管理与 ReAct 决策 |
| **向量检索与 RAG** | [Milvus-Lite](https://milvus.io/), [BGE-M3](https://huggingface.co/BAAI/bge-m3), [BGE-Reranker](https://huggingface.co/BAAI/bge-reranker-large), BM25 | 混合召回、RRF 倒数排名融合、交叉编码二次重排 |
| **关系存储** | [MySQL 8.0](https://www.mysql.com/), [SQLAlchemy 2.0 (Async)](https://www.sqlalchemy.org/), [aiomysql](https://github.com/aio-libs/aiomysql) | 异步 ORM、FAQ 知识库主数据、售后工单与主题分类沉淀 |
| **协议与标准** | [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) | Anthropic 主导的标准工具上下文协议，解耦工具生态 |
| **微调与推理** | [PyTorch](https://pytorch.org/), [Transformers](https://huggingface.co/docs/transformers), [ONNX Runtime](https://onnxruntime.ai/), Tokenizers | RoBERTa 中文全参微调、动态轴导出、轻量独立推理服务 |
| **监控与可观测** | [Langfuse](https://langfuse.com/) | LLM 交互链路 Trace 追踪、Latency 监控与成本透视 |
| **验证与测试** | [pytest](https://pytest.org/), pytest-asyncio, scikit-learn | 异步端到端测试套件、分类器指标测评与实证验收 |

---

## 📁 项目目录结构

```text
AstroBot/
├── app/                          # 核心业务应用源码
│   ├── api/                      # API 路由层 (REST & SSE)
│   │   ├── acceptance.py         # 九项实证验收与主题看板 API
│   │   ├── jobs.py               # 白名单安全作业调度 API
│   │   ├── observability.py      # 可观测性监控 API
│   │   ├── review_queue_routes.py# 人工审核队列 API
│   │   └── routes.py             # 核心对话、知识库与基础业务路由
│   ├── core/                     # 核心运行组件 (作业运行器等)
│   ├── db/                       # 数据库连接池与异步会话管理
│   ├── models/                   # SQLAlchemy ORM 数据模型
│   ├── prompts/                  # 客服人设与系统提示词工程
│   ├── schemas/                  # Pydantic 请求与响应数据契约
│   ├── services/                 # 核心业务服务实现
│   │   ├── classifier/           # 分类器推理与客户端
│   │   ├── context/              # 三层上下文管理与 Token 动态预算
│   │   ├── flywheel/             # 数据飞轮与低置信度拦截
│   │   ├── observability/        # Langfuse 与成本监控服务
│   │   ├── rag/                  # 混合检索、Milvus 向量库与重排引擎
│   │   ├── workflow/             # LangGraph 状态图与退款子流程
│   │   └── chat_service.py       # 对话协调中枢
│   ├── static/                   # 前端静态单页面应用 (HTML/CSS/JS)
│   ├── tools/                    # 本地工具注册、权限沙箱与审计
│   └── config.py                 # 全局统一配置中心 (Pydantic Settings)
├── data/                         # 本地数据目录 (向量库文件、微调报告等)
├── dev-notes/                    # Ch01 ~ Ch10 完整技术演进与实战手记
├── docs/                         # 规范文档与架构方案 (Superpowers Specs & Plans)
├── scripts/                      # 运维、自检、微调训练与旁路作业脚本
│   ├── classifier_service.py     # ONNX 独立推理服务进程 (:8110)
│   ├── classify_pool.py          # 旁路低置信度批量归类流水线
│   ├── train_classifier.py       # RoBERTa 多标签微调训练脚本
│   ├── export_onnx.py            # PyTorch 模型转 ONNX 导出脚本
│   ├── scan_threshold_replay.py  # 阈值 9 线扫描与复算重演
│   ├── mcp_logistics_server.py   # 物流独立 MCP 服务 (:8001)
│   └── mcp_aftersale_server.py   # 售后独立 MCP 服务 (:8002)
├── sql/                          # 数据库 DDL 与初始化脚本
│   ├── ch02-ddl.sql              # 基础订单与工单表
│   └── ch09-flywheel.sql         # 审核队列与主题分类表
├── docker-compose.yml            # MySQL 8.0 容器化编排文件
├── requirements.txt              # 核心 Python 依赖项
├── run.py                        # 一键智能启动与环境自愈入口
└── main.py                       # FastAPI 主应用入口
```

---

## 🚀 快速开始

### 1. 环境准备

- **操作系统**：Linux / macOS / Windows (推荐 WSL2)
- **Python**：`>= 3.10`
- **Docker**：已安装并启动（用于一键运行 MySQL 8.0）

克隆项目并安装依赖：

```bash
git clone https://github.com/Brinkzee/AstroBot.git
cd AstroBot

# 建议在虚拟环境中安装
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 环境变量配置

复制环境变量示例并配置您的模型 API Key：

```bash
cp .env.example .env
```

在 `.env` 中按需填写关键参数：

```ini
# OpenAI 兼容模型配置 (支持 DeepSeek, OpenAI, SiliconFlow, Ollama 等)
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL_NAME=deepseek-flash
OPENAI_TEMPERATURE=0.7

# 数据库连接 (默认本地 Docker MySQL)
DATABASE_URL=mysql+aiomysql://root:root123456@127.0.0.1:3306/astro_bot?charset=utf8mb4

# Milvus-Lite 本地数据路径
MILVUS_URI=./data/milvus/astro_bot.db

# 可选：HuggingFace Token (若使用线上 BGE 模型)
# HUGGINGFACE_TOKEN=hf_xxx

# 可选：Langfuse 可观测性配置
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
LANGFUSE_HOST=http://localhost:3000
```

### 3. 一键智能启动与自愈

AstroBot 提供了强大的启动器 `run.py`。它会自动执行：
1. 环境检测（Python 版本、必要依赖库）；
2. 检查并自动拉起 MySQL Docker 容器并导入 DDL；
3. 检查并自愈 Milvus-Lite 向量库数据；
4. 进行 Token 预算自检与控制台编码适配；
5. 启动 FastAPI + Uvicorn 主服务。

```bash
# 开发模式启动 (支持代码热重载)
python run.py

# 或指定端口启动
python run.py --port 8000

# 仅自检环境，不启动 Web 服务
python run.py --check-only
```

> **Windows 快捷启动**：可直接双击运行根目录下的 `start.bat` 或在 PowerShell 中执行 `start.ps1`。

服务启动后，在浏览器访问 [http://127.0.0.1:8000](http://127.0.0.1:8000) 即可进入客服主界面。

---

### 4. 启动旁路与微服务 (可选)

#### A. 启动独立 MCP 服务（物流与售后）
若需体验标准 MCP 协议解耦工具：
```bash
# 启动物流 MCP 服务 (默认端口 8001)
python scripts/mcp_logistics_server.py

# 启动售后 MCP 服务 (默认端口 8002)
python scripts/mcp_aftersale_server.py
```

#### B. 启动 ONNX 分类器独立推理服务
若需体验第十章的多标签分类旁路与主题看板：
```bash
# 启动轻量推理服务 (默认端口 8110)
python scripts/classifier_service.py
```

#### C. 执行旁路批量归类流水线
自动拉取低置信度审核池中未归类的样本进行批量分类打标：
```bash
python scripts/classify_pool.py --min-batch 10
```

---

## 🖥️ Web 控制台与功能路由导航

AstroBot 内置了完整的前端单页面应用矩阵，启动后可通过浏览器直接访问：

| 页面名称 | 访问路由 | 页面功能说明 |
|:---|:---|:---|
| **客服对话中心** | [`/`](http://127.0.0.1:8000/) 或 [`/chat`](http://127.0.0.1:8000/chat) | 终端用户对话界面，支持 SSE 流式打字机、意图路由提示、上下文历史展示与订单操作。 |
| **知识库管理** | [`/kb`](http://127.0.0.1:8000/kb) | 电商 FAQ 问答库运维，支持知识点增删改查及一键同步向量库。 |
| **RAG 评估看板** | [`/rag-eval`](http://127.0.0.1:8000/rag-eval) | 检索效果评估控制台，可视化展示 Hit@K、MRR 及不同召回策略效果对比。 |
| **可观测性大盘** | [`/observability`](http://127.0.0.1:8000/observability) | 系统调用链路监控、Token 消耗统计、各意图请求分布与成本透视。 |
| **人工审核队列** | [`/review-queue`](http://127.0.0.1:8000/review-queue) | 数据飞轮重要一环，集中呈现低置信度会话供人工标注，支持一键沉淀为正例。 |
| **主题分布看板** | [`/topic-distribution`](http://127.0.0.1:8000/topic-distribution) | 展现经过多标签分类器归类后的 17 类用户诉求分布图表，指导业务补全哪块知识。 |
| **九项实证验收** | [`/acceptance`](http://127.0.0.1:8000/acceptance) | 第十章实证验收大盘，涵盖数据零泄漏硬闸、F1 阈值矩阵、错因记账及就地重跑测试。 |
| **评测指标细览** | [`/acceptance/eval`](http://127.0.0.1:8000/acceptance/eval) | 17 类目分档红线达标情况（严档 0.9 / 中档 0.8 / 宽档 —）及混淆矩阵。 |
| **考卷数据自检** | [`/acceptance/data`](http://127.0.0.1:8000/acceptance/data) | 训练/验证/测试集 80/10/10 分布与零样本泄漏验证详情。 |
| **错因三向记账** | [`/acceptance/errors`](http://127.0.0.1:8000/acceptance/errors) | 错例三向归因分析（漏打/多打/错位）与复核报告。 |

---

## 🔬 实证验收与质量保障

AstroBot 秉持严谨的软件工程与机器学习落地标准，在第十章建立了**九项实证验收硬闸体系**：

1. **三份考卷零泄漏硬闸**：训练集、验证集、测试集严格隔离，文本交集严格为 0；
2. **17 类权威术语表统一**：全链路采用单一权威标准定义标签体系；
3. **分档容错红线判定**：依业务影响面设立严（F1 $\ge 0.9$）、中（$\ge 0.8$）、宽（—）三档红线；
4. **错因三向记账闭环**：细分“漏打”、“多打”、“错位”，与混淆矩阵笔数保持数学闭环；
5. **判定阈值 9 线重演**：支持通过 `scan_threshold_replay.py` 动态重演扫描候选阈值；
6. **ONNX 与 PyTorch 一致性校验**：输出 Logits 误差在容差内，预测标签达成 100% 一致；
7. **独立推理服务轻量化**：FastAPI + ONNX Runtime，彻底移除推理阶段重型框架依赖；
8. **旁路批处理幂等性**：批处理归类支持断点续跑与重复执行幂等；
9. **白名单安全作业运行器**：系统提供受限白名单作业通道，可在验收看板中就地一键复测。

---

## 📄 开源协议

本项目采用 [Apache License 2.0](LICENSE) 开源许可证。
