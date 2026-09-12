import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.workflow.state import create_initial_state
from app.services.workflow.nodes.refund_nodes import (
    QUERY_EXPANSION_SYSTEM_PROMPT,
    merge_and_deduplicate_docs,
    refund_expansion_retrieval_node,
)
from app.services.workflow.nodes import (
    refund_expansion_retrieval_node as exported_refund_expansion_retrieval_node,
)


@pytest.mark.asyncio
async def test_query_expansion_json_parsing_and_include_original_query():
    """测试用例 1: 大模型扩写输出强 JSON {"queries": [...]} 成功解析并包含原 query"""
    # 验证 exported 符号与内部定义为同一函数
    assert exported_refund_expansion_retrieval_node is refund_expansion_retrieval_node

    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=json.dumps({
                "queries": [
                    "七天无理由退货规则与时效",
                    "服装类退换货二次销售要求",
                    "已发货商品退货运费承担原则",
                ]
            })
        )
    )

    # 模拟检索器
    mock_retriever = MagicMock()
    called_queries = []

    async def mock_retrieve_with_strategy(query, **kwargs):
        called_queries.append(query)
        mock_hit = MagicMock()
        mock_hit.score = 0.8
        mock_hit.to_dict.return_value = {
            "chunk_id": f"chunk_{len(called_queries)}",
            "text": f"命中政策文本_{query}",
            "score": 0.8,
        }
        res = MagicMock()
        res.hits = [mock_hit]
        return res

    mock_retriever.retrieve_with_strategy = AsyncMock(side_effect=mock_retrieve_with_strategy)

    state = create_initial_state(1, "我的衣服能退吗")
    state["resolved_query"] = "订单1001极简保暖羽绒服支持退货吗"
    state["order_id"] = "1001"
    state["order_data"] = {
        "order_id": "1001",
        "订单状态": "已发货",
        "商品明细": [{"商品名称": "极简保暖羽绒服"}],
    }

    result = await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)

    assert "retrieved_docs" in result
    assert isinstance(result["retrieved_docs"], list)
    # 原 resolved_query 与 3 条扩写 query 均被检索
    assert "订单1001极简保暖羽绒服支持退货吗" in called_queries
    assert "七天无理由退货规则与时效" in called_queries
    assert "服装类退换货二次销售要求" in called_queries
    assert "已发货商品退货运费承担原则" in called_queries
    assert len(called_queries) == 4


def test_merge_and_deduplicate_docs_max_score_and_descending_sort():
    """测试用例 2: 并发多路检索命中重叠文档时，以内容去重并保留最高 score，按分值降序排列"""
    # 模拟多路召回的文档列表（包含重叠文本与不同分值）
    route1_docs = [
        {"chunk_id": 1, "text": "七天无理由退货规则：在商品完好前提下支持7天内退货。", "score": 0.65},
        {"chunk_id": 2, "text": "服装类商品若吊牌已剪或有洗涤痕迹，影响二次销售不支持退货。", "score": 0.82},
    ]
    route2_docs = [
        # 同一篇文档，但在 route2 分数更高为 0.95
        {"chunk_id": 1, "text": "七天无理由退货规则：在商品完好前提下支持7天内退货。", "score": 0.95},
        {"chunk_id": 3, "text": "已发货商品申请退款需等待拦截或拒收后原路返还款项。", "score": 0.78},
    ]
    route3_docs = [
        # 相同文本但包含空白字符
        {"chunk_id": 4, "text": "已发货商品申请退款需等待拦截或拒收后原路返还款项。  ", "score": 0.70},
        {"chunk_id": 5, "text": "退货运费由责任方承担，非质量问题买家自理。", "score": 0.88},
    ]

    merged = merge_and_deduplicate_docs([route1_docs, route2_docs, route3_docs], top_k=5)

    # 4篇唯一文档（文档1保留0.95，文档2为0.82，文档3保留0.78，文档5为0.88）
    assert len(merged) == 4

    # 验证按分值严格降序排列
    scores = [doc["score"] for doc in merged]
    assert scores == [0.95, 0.88, 0.82, 0.78]

    # 验证第一篇为文档1且分数为最高分 0.95
    assert "七天无理由退货规则" in merged[0]["text"]
    assert merged[0]["score"] == 0.95

    # 验证 top_k 截断
    top2 = merge_and_deduplicate_docs([route1_docs, route2_docs, route3_docs], top_k=2)
    assert len(top2) == 2
    assert [d["score"] for d in top2] == [0.95, 0.88]


@pytest.mark.asyncio
async def test_query_expansion_fallback_on_llm_exception():
    """测试用例 3a: 大模型调用异常时，安全降级为原样单 query 检索"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM service unavailable"))

    mock_retriever = MagicMock()
    called_queries = []

    async def mock_retrieve(query, **kwargs):
        called_queries.append(query)
        return [{"text": f"兜底检索文档_{query}", "score": 0.75}]

    mock_retriever.retrieve_with_strategy = AsyncMock(side_effect=mock_retrieve)

    state = create_initial_state(1, "退款怎么退")
    state["resolved_query"] = "退款怎么退"

    result = await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)

    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 1
    assert called_queries == ["退款怎么退"]
    assert result["retrieved_docs"][0]["text"] == "兜底检索文档_退款怎么退"


@pytest.mark.asyncio
async def test_query_expansion_fallback_on_invalid_json():
    """测试用例 3b: 大模型返回非 JSON 或格式错误时，安全降级为原样单 query 检索"""
    mock_model = MagicMock()
    # 模拟大模型输出解释性文字或畸形内容，无有效 JSON
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="抱歉，我无法生成相关的检索词。")
    )

    mock_retriever = MagicMock()
    called_queries = []

    async def mock_retrieve(query, **kwargs):
        called_queries.append(query)
        return [{"text": f"单路文档_{query}", "score": 0.80}]

    mock_retriever.retrieve = AsyncMock(side_effect=mock_retrieve)

    state = create_initial_state(1, "能退货吗")
    state["resolved_query"] = "订单1002潮流连帽卫衣能退货吗"

    result = await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)

    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) == 1
    assert called_queries == ["订单1002潮流连帽卫衣能退货吗"]


@pytest.mark.asyncio
async def test_order_and_product_context_injected_into_prompt():
    """测试用例 4: 结合 order_data（商品名称为极简保暖羽绒服、状态为已发货），生成的提示词包含订单与商品上下文"""
    mock_model = MagicMock()
    prompt_captured = []

    async def capture_ainvoke(messages):
        prompt_captured.extend(messages)
        return AIMessage(content='{"queries": ["羽绒服退货要求"]}')

    mock_model.ainvoke = AsyncMock(side_effect=capture_ainvoke)

    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(return_value=[{"text": "羽绒服退货政策", "score": 0.9}])

    state = create_initial_state(1, "这件衣服可以退款吗")
    state["resolved_query"] = "订单1001极简保暖羽绒服可以申请退款退货吗"
    state["order_id"] = "1001"
    state["order_data"] = {
        "order_id": "1001",
        "订单状态": "已发货",
        "支付金额": "299.00元",
        "商品明细": [
            {"商品名称": "极简保暖羽绒服", "数量": 1, "单价": "299.00元"}
        ],
    }

    await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)

    # 验证传入模型的 System Prompt 包含预设规范
    sys_msgs = [m for m in prompt_captured if isinstance(m, SystemMessage)]
    assert len(sys_msgs) >= 1
    assert "QUERY_EXPANSION_SYSTEM_PROMPT" in globals()
    assert sys_msgs[0].content == QUERY_EXPANSION_SYSTEM_PROMPT
    assert "queries" in sys_msgs[0].content

    # 验证传入模型的 Human Message 注入了真实订单状态与商品名称
    human_msgs = [m for m in prompt_captured if isinstance(m, HumanMessage)]
    assert len(human_msgs) >= 1
    human_content = human_msgs[0].content
    assert "极简保暖羽绒服" in human_content
    assert "已发货" in human_content
    assert "1001" in human_content


@pytest.mark.asyncio
async def test_retriever_compatibility_retrieve_with_strategy():
    """测试用例 5a: 检索器接口兼容性 - 支持 retrieve_with_strategy"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"queries": ["退款时效", "退货条件"]}')
    )

    mock_retriever = MagicMock()
    # 仅提供 retrieve_with_strategy，删除 retrieve 属性
    del mock_retriever.retrieve

    mock_hit = MagicMock()
    mock_hit.score = 0.85
    mock_hit.to_dict.return_value = {"text": "退货时效政策说明", "score": 0.85}
    res = MagicMock()
    res.hits = [mock_hit]
    mock_retriever.retrieve_with_strategy = AsyncMock(return_value=res)

    state = create_initial_state(1, "退货规则")
    state["resolved_query"] = "退货规则"

    result = await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)

    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) > 0
    assert result["retrieved_docs"][0]["score"] == 0.85
    assert result["retrieved_docs"][0]["text"] == "退货时效政策说明"


@pytest.mark.asyncio
async def test_retriever_compatibility_legacy_retrieve():
    """测试用例 5b: 检索器接口兼容性 - 支持旧版 retrieve"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"queries": ["七天无理由"]}')
    )

    # 仅具有 retrieve 方法的对象
    class LegacyRetriever:
        async def retrieve(self, query, **kwargs):
            return [{"text": f"旧版检索命中: {query}", "score": 0.72}]

    legacy_retriever = LegacyRetriever()

    state = create_initial_state(1, "衣服能退吗")
    state["resolved_query"] = "衣服能退吗"

    result = await refund_expansion_retrieval_node(state, model=mock_model, retriever=legacy_retriever)

    assert "retrieved_docs" in result
    assert len(result["retrieved_docs"]) > 0
    assert result["retrieved_docs"][0]["score"] == 0.72
    assert "旧版检索命中" in result["retrieved_docs"][0]["text"]


@pytest.mark.asyncio
async def test_query_expansion_markdown_codeblock_parsing():
    """测试大模型输出被 ```json 代码块包裹时的容错解析"""
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(
        return_value=AIMessage(
            content="""```json
{
  "queries": [
    "退换货二次销售标准",
    "退货运费承担"
  ]
}
```"""
        )
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(
        return_value=[{"text": "二次销售标准政策", "score": 0.89}]
    )

    state = create_initial_state(1, "退货运费谁出")
    state["resolved_query"] = "退货运费谁出"

    result = await refund_expansion_retrieval_node(state, model=mock_model, retriever=mock_retriever)
    assert len(result["retrieved_docs"]) > 0
    assert result["retrieved_docs"][0]["score"] == 0.89
