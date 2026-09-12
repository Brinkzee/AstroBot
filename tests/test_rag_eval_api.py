import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from main import app
from app.db.session import get_db
from app.models.faith_case import FaithCase
from app.services.rag.evaluator import parse_evaluation_report_md

client = TestClient(app)


# ==============================================================================
# 1. 评估结果 Markdown 解析器单测
# ==============================================================================

def test_parse_evaluation_report_md_valid_legacy_and_structured():
    """测试 parse_evaluation_report_md 能正确解析现有及增强后的 Markdown 评估产物。"""
    sample_md = """# AstroBot RAG Chapter 4 四策略对比评测报告

- **评测时间**: 2026-09-08 23:20:46
- **样本总数**: 300 题
- **知识库切片数**: 128 块
- **嵌入模型**: BAAI/bge-m3
- **重排模型**: BAAI/bge-reranker-v2-m3
- **裁判模型**: kimi-k2.7-code
- **参评策略**: vector_only, bm25_only, hybrid, hybrid_rerank
- **持久化编造个案数**: 0 例

---

## 1. 检索召回率对比 (Recall@3 / Recall@5 / Recall@10)

### Recall@3 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 0.950 | 0.700 | 0.850 | 0.600 | **0.775** |
| `bm25_only` | 0.800 | 0.980 | 0.650 | 0.550 | **0.745** |
| `hybrid` | 0.980 | 0.990 | 0.880 | 0.750 | **0.900** |
| `hybrid_rerank` | 1.000 | 1.000 | 0.950 | 0.850 | **0.950** |

### Recall@5 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 0.980 | 0.750 | 0.900 | 0.650 | **0.820** |
| `bm25_only` | 0.850 | 1.000 | 0.700 | 0.600 | **0.787** |
| `hybrid` | 1.000 | 1.000 | 0.920 | 0.800 | **0.930** |
| `hybrid_rerank` | 1.000 | 1.000 | 0.980 | 0.900 | **0.970** |

## 2. 平均倒数排名 (MRR) 对比
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 0.920 | 0.650 | 0.780 | 0.550 | **0.725** |
| `bm25_only` | 0.750 | 0.950 | 0.580 | 0.500 | **0.695** |
| `hybrid` | 0.950 | 0.960 | 0.820 | 0.680 | **0.853** |
| `hybrid_rerank` | 0.990 | 0.980 | 0.920 | 0.810 | **0.925** |

## 3. 生成端质量与安全防护指标

### 忠实度 (Faithfulness) 与 D_absent 拒答率
| 策略 | 忠实度 (Faithfulness) | D_absent 拒答率 | 编造个案数 |
| :--- | :---: | :---: | :---: |
| `vector_only` | 0.880 | 0.850 | 7 |
| `bm25_only` | 0.850 | 0.800 | 9 |
| `hybrid` | 0.920 | 0.950 | 5 |
| `hybrid_rerank` | 0.980 | 1.000 | 1 |
"""
    parsed = parse_evaluation_report_md(sample_md)
    assert parsed is not None
    assert "meta" in parsed
    assert parsed["meta"]["eval_set_size"] == 300
    assert parsed["meta"]["embedding_model"] == "BAAI/bge-m3"
    assert parsed["meta"]["reranker_model"] == "BAAI/bge-reranker-v2-m3"

    # 四项 KPI 断言
    assert "kpis" in parsed
    kpis = parsed["kpis"]
    assert kpis["best_overall_mrr"] == 0.925
    assert kpis["best_mrr_strategy"] == "hybrid_rerank"
    assert "colloquial_mrr_lift" in kpis
    assert kpis["out_of_scope_refusal_rate"] == 1.0

    # 检索质量对比
    assert "retrieval" in parsed
    assert "mrr" in parsed["retrieval"]
    assert "recall_at_5" in parsed["retrieval"]
    assert "evidence_coverage" in parsed["retrieval"]

    # 生成质量对比
    assert "generation" in parsed
    assert parsed["generation"] is not None
    assert "faithfulness" in parsed["generation"]

    # 完整数据汇总表
    assert "full_table" in parsed
    assert len(parsed["full_table"]) == 4
    # hybrid_rerank 必须带 is_best=True 底色标记
    best_rows = [r for r in parsed["full_table"] if r.get("is_best")]
    assert len(best_rows) == 1
    assert best_rows[0]["strategy"] == "hybrid_rerank"


def test_parse_evaluation_report_md_corrupted_raises_value_error():
    """测试损坏或严重截断的 Markdown 抛出 ValueError。"""
    bad_md = "这是一个随意的错误文本，没有包含任何合法的报告结构"
    with pytest.raises(ValueError):
        parse_evaluation_report_md(bad_md)


def test_parse_evaluation_report_md_half_report_generation_none():
    """测试裁判上游挂掉时的半份结果：generation 为 None，检索段照常解析。"""
    half_md = """# AstroBot RAG Chapter 4 四策略对比评测报告

- **评测时间**: 2026-09-08 23:20:46
- **样本总数**: 300 题

## 1. 检索召回率对比 (Recall@3 / Recall@5 / Recall@10)

### Recall@5 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `hybrid_rerank` | 1.000 | 1.000 | 0.980 | 0.900 | **0.970** |

## 2. 平均倒数排名 (MRR) 对比
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `hybrid_rerank` | 0.990 | 0.980 | 0.920 | 0.810 | **0.925** |
"""
    parsed = parse_evaluation_report_md(half_md)
    assert parsed["generation"] is None
    assert parsed["retrieval"]["mrr"]["hybrid_rerank"]["overall"] == 0.925


# ==============================================================================
# 2. 页面路由与 GET /api/rag-eval/report 单测
# ==============================================================================

def test_get_rag_eval_page_html():
    """测试访问 /rag-eval 与 /rag_eval 均能成功返回包含工作台核心结构的 HTML。"""
    for route in ["/rag-eval", "/rag_eval"]:
        resp = client.get(route)
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "RAG 评估" in resp.text or "rag-eval" in resp.text.lower()
        assert "四项 KPI" in resp.text or "kpi" in resp.text.lower()
        assert "重跑 RAG 评估" in resp.text or "重跑" in resp.text


def test_api_rag_eval_report_present_false_when_file_missing(tmp_path):
    """测试产物文件不存在时，返回 present=false 与 suggested_job='eval-rag'。"""
    with patch("app.api.rag_eval_routes.REPORT_PATH", tmp_path / "non_existent.md"):
        resp = client.get("/api/rag-eval/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["present"] is False
        assert data["suggested_job"] == "eval-rag"


def test_api_rag_eval_report_success():
    """测试已存在合法报告时，成功返回解析后的数据指标。"""
    resp = client.get("/api/rag-eval/report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["present"] is True
    assert "meta" in data
    assert "kpis" in data
    assert "retrieval" in data
    assert "full_table" in data


# ==============================================================================
# 3. 编造个案台账 GET/POST /api/rag-eval/faith-cases 单测
# ==============================================================================

def test_api_rag_eval_faith_cases_query_and_metrics():
    """测试获取编造个案列表，验证双口径幻觉率、状态过滤与 cited_indices 提取。"""
    mock_case_1 = MagicMock(spec=FaithCase)
    mock_case_1.id = 1
    mock_case_1.eval_id = "A43"
    mock_case_1.bucket = "A_policy"
    mock_case_1.query = "退款支持私下赔偿吗"
    mock_case_1.strategy = "hybrid_rerank"
    mock_case_1.answer = "退款不支持私下赔偿，支持7天无理由退货 [1]，款项原路返还 [2]。"
    mock_case_1.reason = "回答中承诺了未收录的政策"
    mock_case_1.citations = [
        {"n": 1, "chunk_id": 10, "section_path": "售后政策", "question": "退货规则", "answer": "7天无理由"},
        {"n": 2, "chunk_id": 11, "section_path": "售后政策", "question": "退款路径", "answer": "原路返还"},
        {"n": 3, "chunk_id": 12, "section_path": "其他说明", "question": "其他规则", "answer": "无"},
    ]
    mock_case_1.judge_model = "kimi-k2.7-code"
    mock_case_1.status = "未解决"
    mock_case_1.seen_count = 2
    mock_case_1.first_seen_at = None
    mock_case_1.last_seen_at = None
    mock_case_1.resolution = None
    mock_case_1.resolved_at = None

    mock_session = AsyncMock()
    # 模拟 count 与 list 查询
    mock_count_res = MagicMock()
    mock_count_res.scalar.return_value = 1
    mock_list_res = MagicMock()
    mock_list_res.scalars.return_value.all.return_value = [mock_case_1]

    # 总计统计 mock
    mock_total_cnt = MagicMock()
    mock_total_cnt.scalar.return_value = 1
    mock_unres_cnt = MagicMock()
    mock_unres_cnt.scalar.return_value = 1
    mock_res_cnt = MagicMock()
    mock_res_cnt.scalar.return_value = 0
    mock_wont_cnt = MagicMock()
    mock_wont_cnt.scalar.return_value = 0

    mock_session.execute = AsyncMock(
        side_effect=[
            mock_count_res,
            mock_list_res,
            mock_total_cnt,
            mock_unres_cnt,
            mock_res_cnt,
            mock_wont_cnt,
        ]
    )

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        resp = client.get("/api/rag-eval/faith-cases?status=未解决&page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "current_round_judge_rate" in data["metrics"]
        assert "current_round_human_confirmed_rate" in data["metrics"]
        assert "ledger_summary" in data["metrics"]
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["eval_id"] == "A43"
        # 验证从 "[1]...[2]" 中精准提取出的角标列表
        assert item["cited_indices"] == [1, 2]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_api_rag_eval_faith_cases_database_error_graceful_fallback():
    """测试数据库离线或查询异常时，接口优雅降级返回空列表与友好提示，杜绝 500。"""
    from sqlalchemy.exc import OperationalError

    mock_session = AsyncMock()
    mock_session.execute.side_effect = OperationalError("Can't connect to MySQL server", params=None, orig=Exception("Connection refused"))

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        resp = client.get("/api/rag-eval/faith-cases?page=1&page_size=20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert "warning" in data
        assert "数据库" in data["warning"]
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_api_rag_eval_faith_cases_resolve_requires_resolution():
    """测试处置编造个案时，resolution 说明为空则拒绝提交 (400)。"""
    resp = client.post(
        "/api/rag-eval/faith-cases/1/resolve",
        json={"status": "已解决", "resolution": ""},
    )
    assert resp.status_code == 400
    assert "说明" in resp.json()["detail"]


def test_api_rag_eval_faith_cases_resolve_success():
    """测试处置编造个案成功更新状态。"""
    mock_case = MagicMock(spec=FaithCase)
    mock_case.id = 1
    mock_case.status = "未解决"
    mock_case.resolution = None
    mock_case.resolved_at = None

    mock_session = AsyncMock()
    mock_get_res = MagicMock()
    mock_get_res.scalar_one_or_none.return_value = mock_case
    mock_session.execute = AsyncMock(return_value=mock_get_res)
    mock_session.commit = AsyncMock()

    async def override_get_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        resp = client.post(
            "/api/rag-eval/faith-cases/1/resolve",
            json={"status": "已解决", "resolution": "修正了知识库中不规范的赔付条款"},
        )
        assert resp.status_code == 200
        assert mock_case.status == "已解决"
        assert mock_case.resolution == "修正了知识库中不规范的赔付条款"
    finally:
        app.dependency_overrides.pop(get_db, None)


# ==============================================================================
# 4. 白名单作业运行器 /api/jobs 单测
# ==============================================================================

def test_api_jobs_reject_non_whitelist_job():
    """测试提交不在白名单内的危险作业名被 400 拒绝。"""
    resp = client.post("/api/jobs", json={"job_name": "rm -rf /"})
    assert resp.status_code == 400
    assert "白名单" in resp.json()["detail"]


def test_api_jobs_start_and_status_and_logs():
    """测试合法白名单作业 eval-rag 的启动、状态轮询与日志读取。"""
    resp = client.post("/api/jobs", json={"job_name": "eval-rag"})
    assert resp.status_code == 200
    job_info = resp.json()
    assert "job_id" in job_info
    assert job_info["job_name"] == "eval-rag"

    job_id = job_info["job_id"]

    # 查询状态
    status_resp = client.get(f"/api/jobs/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ["pending", "running", "completed", "failed"]

    # 查询日志
    log_resp = client.get(f"/api/jobs/{job_id}/logs")
    assert log_resp.status_code == 200
    assert "logs" in log_resp.json()
