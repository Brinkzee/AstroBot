"""Task 3 单测套件：BGE-Reranker-v2-m3 重排客户端与 Lost in the Middle 首尾重排。

覆盖内容：
1. lost_in_the_middle_reorder:
   - 输入 10 条相关性降序的文档 [D1..D10]，断言输出严格符合 [D1, D3, D5, D7, D9, D10, D8, D6, D4, D2]；
   - 边界情况测试：空列表、1 条、2 条、奇数条（3 条、5 条）、偶数条（4 条）；
2. build_citation_items:
   - 验证输出格式严格符合 [{n, chunk_id, section_path, question, answer}]；
   - 序号 n 从 1 递增至 len(docs)；
   - 验证 chunk_id/id 回填与 question/questions 容错；
3. BGERerankerClient:
   - Mock 模式下的语义重叠度打分与排序（高匹配文本得分显著高于不相关文本）；
   - Top-K 截断与空候选列表处理；
   - 在线调用请求构造（inputs: [{text, text_pair}]）与响应解析；
   - 异常重试机制（3 次指数退避重试成功）；
   - 异常降级机制（3 次重试全部失败后自动回退为确定性 Mock 打分）。
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from app.services.rag.reorder import (
    build_citation_items,
    lost_in_the_middle_reorder,
)
from app.services.rag.reranker import BGERerankerClient


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """每次测试前后重置熔断标记，避免跨测试状态污染。"""
    BGERerankerClient._circuit_broken_until = 0.0
    yield
    BGERerankerClient._circuit_broken_until = 0.0


# ==============================================================================
# 1. lost_in_the_middle_reorder 单测
# ==============================================================================

def test_lost_in_the_middle_reorder_10_items():
    """测试 10 条降序候选文档的首尾放置重排。
    
    输入顺序：[D1, D2, D3, D4, D5, D6, D7, D8, D9, D10]
    预期输出顺序：[D1, D3, D5, D7, D9, D10, D8, D6, D4, D2]
    最高分 (D1) 位于首位 (index 0)，次高分 (D2) 位于尾部 (index 9)。
    """
    input_docs = [{"id": i, "score": 100 - i * 5} for i in range(1, 11)]
    reordered = lost_in_the_middle_reorder(input_docs)

    assert len(reordered) == 10
    reordered_ids = [d["id"] for d in reordered]
    assert reordered_ids == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert reordered[0]["id"] == 1  # 最高分在最前
    assert reordered[-1]["id"] == 2  # 次高分在最后


def test_lost_in_the_middle_reorder_edge_cases():
    """测试首尾放置重排的各种边界条件：空列表、单条、双条、奇数条（3/5条）、偶数条（4条）。"""
    # 边界 1: 空列表
    assert lost_in_the_middle_reorder([]) == []

    # 边界 2: 单条记录
    single = [{"id": 1, "text": "only one"}]
    assert lost_in_the_middle_reorder(single) == [{"id": 1, "text": "only one"}]

    # 边界 3: 2 条记录 [D1, D2] -> [D1, D2]
    two_items = [{"id": 1}, {"id": 2}]
    assert [d["id"] for d in lost_in_the_middle_reorder(two_items)] == [1, 2]

    # 边界 4: 3 条记录 [D1, D2, D3] -> [D1, D3, D2]
    three_items = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert [d["id"] for d in lost_in_the_middle_reorder(three_items)] == [1, 3, 2]

    # 边界 5: 4 条记录 [D1, D2, D3, D4] -> [D1, D3, D4, D2]
    four_items = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
    assert [d["id"] for d in lost_in_the_middle_reorder(four_items)] == [1, 3, 4, 2]

    # 边界 6: 5 条记录 [D1, D2, D3, D4, D5] -> [D1, D3, D5, D4, D2]
    five_items = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
    assert [d["id"] for d in lost_in_the_middle_reorder(five_items)] == [1, 3, 5, 4, 2]


# ==============================================================================
# 2. build_citation_items 单测
# ==============================================================================

def test_build_citation_items_standard_format():
    """测试 build_citation_items 输出符合 [{n, chunk_id, section_path, question, answer}] 规范。"""
    docs = [
        {
            "chunk_id": 12,
            "section_path": "售后政策 > 退货流程",
            "question": "退货政策是什么样的",
            "answer": "支持7天无理由退货，商品需完好。",
        },
        {
            "chunk_id": 15,
            "section_path": "售后政策 > 换货流程",
            "question": "怎么申请换货",
            "answer": "可在订单页点击申请售后选择换货。",
        },
        {
            "chunk_id": 20,
            "section_path": "物流说明 > 配送时效",
            "question": "几天能送到",
            "answer": "顺丰特快默认次日达。",
        },
    ]

    citations = build_citation_items(docs)

    assert len(citations) == 3
    for idx, item in enumerate(citations, start=1):
        assert item["n"] == idx
        assert item["chunk_id"] == docs[idx - 1]["chunk_id"]
        assert item["section_path"] == docs[idx - 1]["section_path"]
        assert item["question"] == docs[idx - 1]["question"]
        assert item["answer"] == docs[idx - 1]["answer"]


def test_build_citation_items_fallback_and_edge_cases():
    """测试 build_citation_items 在字段缺失或别名时的容错与回填逻辑。"""
    # 边界 1: 空列表
    assert build_citation_items([]) == []

    # 边界 2: 只有 id 无 chunk_id；questions 为列表；缺失 section_path
    docs = [
        {
            "id": 101,
            "questions": ["主问题1", "别名问题2"],
            "answer": "回答内容1",
        },
        {
            "id": 102,
            "questions": "多行问题问法A\n多行问题问法B",
            "answer": "回答内容2",
            "section_path": "类目A > 子类B",
        },
    ]

    citations = build_citation_items(docs)
    assert len(citations) == 2

    # 第 1 条验证
    assert citations[0]["n"] == 1
    assert citations[0]["chunk_id"] == 101
    assert citations[0]["section_path"] == ""
    assert citations[0]["question"] == "主问题1"
    assert citations[0]["answer"] == "回答内容1"

    # 第 2 条验证
    assert citations[1]["n"] == 2
    assert citations[1]["chunk_id"] == 102
    assert citations[1]["section_path"] == "类目A > 子类B"
    assert citations[1]["question"] == "多行问题问法A"
    assert citations[1]["answer"] == "回答内容2"


# ==============================================================================
# 3. BGERerankerClient 单测
# ==============================================================================

@pytest.mark.asyncio
async def test_bge_reranker_mock_scoring_and_ranking():
    """测试 BGERerankerClient 在 Mock 模式下的语义重排与打分。
    
    高匹配文本得分必须显著高于不相关文本，且输出候选严格按 rerank_score 降序排列。
    """
    reranker = BGERerankerClient(mock=True)

    query = "星光PRO-X99智能手表支持游泳防水吗"
    candidates = [
        {
            "id": 1,
            "category": "智能穿戴",
            "question": "手表电池能用几天",
            "answer": "星光PRO-X99正常使用续航可达14天。",
        },
        {
            "id": 2,
            "category": "售后政策",
            "question": "退货退款到账要多久",
            "answer": "平台审核通过后，原路退款一般3-5个工作日到账。",
        },
        {
            "id": 3,
            "category": "智能穿戴",
            "question": "星光PRO-X99防水等级和游泳佩戴",
            "answer": "星光PRO-X99具备5ATM防水等级，支持在泳池或浅海游泳佩戴，并能实时记录泳姿与划水次数。",
        },
    ]

    reranked = await reranker.rerank(query, candidates, top_k=3)

    assert len(reranked) == 3
    # 候选 3 与提问高度契合（包含 PRO-X99、防水、游泳佩戴），应排在第 1 位
    assert reranked[0]["id"] == 3
    # 候选 1 部分匹配（PRO-X99 手表），应排在第 2 位
    assert reranked[1]["id"] == 1
    # 候选 2 完全不相关（售后退款），应排在第 3 位
    assert reranked[2]["id"] == 2

    # 分数断言：高匹配分数必须显著高于完全不相关分数
    assert reranked[0]["rerank_score"] > reranked[2]["rerank_score"] + 0.3
    # 检查降序
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"] >= reranked[2]["rerank_score"]


@pytest.mark.asyncio
async def test_bge_reranker_empty_candidates_and_top_k():
    """测试 BGERerankerClient 空输入与 top_k 截断逻辑。"""
    reranker = BGERerankerClient(mock=True)

    # 空候选集
    res_empty = await reranker.rerank("测试", [], top_k=5)
    assert res_empty == []

    # 截断测试：输入 5 条，top_k=2，仅返回前 2 条
    candidates = [
        {"id": i, "answer": f"商品测试描述信息 {i} 防水游泳"} for i in range(1, 6)
    ]
    res_top2 = await reranker.rerank("防水游泳", candidates, top_k=2)
    assert len(res_top2) == 2


@pytest.mark.asyncio
async def test_bge_reranker_online_api_call_and_parsing():
    """测试 HuggingFace Inference API 批量调用流程与 Cross-Encoder 分数解析。"""
    reranker = BGERerankerClient(token="hf_test_token", mock=False)

    candidates = [
        {"id": 101, "chunk_text": "【类目】售后\n【标准问法】退换货政策\n【解答】支持7天无理由退换。"},
        {"id": 102, "chunk_text": "【类目】商品\n【标准问法】手表尺寸\n【解答】表盘直径44毫米。"},
    ]

    # 模拟 HF 返回标准分类得分数据
    fake_hf_response = [
        [{"label": "LABEL_0", "score": 0.92}],
        [{"label": "LABEL_0", "score": 0.18}],
    ]

    with patch.object(
        reranker.client,
        "_inner_post",
        return_value=json.dumps(fake_hf_response).encode("utf-8"),
    ) as mock_post:
        reranked = await reranker.rerank("退换货政策是什么", candidates, top_k=2)

        # 验证调用了 _inner_post
        assert mock_post.call_count == 1
        call_req = mock_post.call_args[0][0]
        # 验证请求结构符合 text-classification 的 inputs: [{text, text_pair}]
        assert call_req.task == "text-classification"
        assert "inputs" in call_req.json
        assert len(call_req.json["inputs"]) == 2
        assert call_req.json["inputs"][0]["text"] == "退换货政策是什么"
        assert "退换货政策" in call_req.json["inputs"][0]["text_pair"]

        # 验证解析后的分数与排序
        assert len(reranked) == 2
        assert reranked[0]["id"] == 101
        assert reranked[0]["rerank_score"] == pytest.approx(0.92, abs=1e-4)
        assert reranked[1]["id"] == 102
        assert reranked[1]["rerank_score"] == pytest.approx(0.18, abs=1e-4)


@pytest.mark.asyncio
async def test_bge_reranker_retry_mechanism_success():
    """测试网络或 API 临时波动时，重试 3 次指数退避最终成功。"""
    reranker = BGERerankerClient(
        token="hf_test_token",
        max_retries=3,
        retry_delay=0.001,  # 极短延时加速测试
        mock=False,
    )

    candidates = [
        {"id": 1, "answer": "回答 1"},
        {"id": 2, "answer": "回答 2"},
    ]

    fake_hf_response = [
        [{"label": "LABEL_0", "score": 0.85}],
        [{"label": "LABEL_0", "score": 0.45}],
    ]

    call_count = 0

    def mock_inner_post(req):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("503 Service Unavailable")
        return json.dumps(fake_hf_response).encode("utf-8")

    with patch.object(reranker.client, "_inner_post", side_effect=mock_inner_post):
        reranked = await reranker.rerank("测试", candidates, top_k=2)

        # 验证重试了 3 次（2 次失败 + 1 次成功）
        assert call_count == 3
        assert len(reranked) == 2
        assert reranked[0]["rerank_score"] == pytest.approx(0.85, abs=1e-4)


@pytest.mark.asyncio
async def test_bge_reranker_retry_failure_fallback_to_mock():
    """测试 3 次重试全部失败后，平滑优雅降级为确定性 Mock 打分。"""
    reranker = BGERerankerClient(
        token="hf_test_token",
        max_retries=3,
        retry_delay=0.001,
        mock=False,
    )

    candidates = [
        {"id": 1, "answer": "完全不相干的内容"},
        {"id": 2, "answer": "支持7天无理由退货的退款政策说明"},
    ]

    with patch.object(
        reranker.client,
        "_inner_post",
        side_effect=RuntimeError("Connection refused by peer"),
    ) as mock_post:
        # 全部失败后降级为 Mock
        reranked = await reranker.rerank("退款政策说明", candidates, top_k=2)

        # 验证确实重试了 3 次
        assert mock_post.call_count == 3
        # 降级后结果不应为空，且仍按相关性正确排序（候选 2 明显包含关键词）
        assert len(reranked) == 2
        assert reranked[0]["id"] == 2
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
