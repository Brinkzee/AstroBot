"""BGE-Reranker-v2-m3 在线重排客户端与确定性降级服务。

特性：
1. 生产环境对接 Hugging Face Inference Router 端点 (BAAI/bge-reranker-v2-m3)；
2. 构造 text-classification 任务请求，传入 [{"text": query, "text_pair": doc_text}, ...] 批量推理；
3. 内置 3 次指数退避重试 (Exponential Backoff)；
4. 内置基于词覆盖率与 Jaccard 相似度的确定性 Mock 语义打分，供离线单测或线上 API 故障平滑降级；
5. 提供异步 rerank 接口与结果 top_k 截断。
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import jieba

from app.config import settings

logger = logging.getLogger(__name__)


class BGERerankerClient:
    """BAAI/bge-reranker-v2-m3 交叉编码 (Cross-Encoder) 重排客户端。"""

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = "BAAI/bge-reranker-v2-m3",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        mock: bool = False,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if mock or os.getenv("MOCK_RERANKER") == "1" or os.getenv("ASTRO_MOCK_RERANKER") == "1":
            self.token = "mock"
            self.is_mock = True
            self.client = None
            logger.info("BGERerankerClient: 显式启用 Mock 语义打分模式")
            return

        # 解析 Hugging Face Token：参数 > settings.HUGGINGFACE_TOKEN > settings.huggingface_token > 环境变量
        resolved_token = token
        if resolved_token is None:
            resolved_token = (
                getattr(settings, "HUGGINGFACE_TOKEN", None)
                or getattr(settings, "huggingface_token", None)
                or os.getenv("HUGGINGFACE_TOKEN")
            )

        self.token = resolved_token
        self.is_mock = False
        self.client = None

        if not self.token or self.token == "mock":
            self.is_mock = True
            logger.info("BGERerankerClient: 未配置有效 Token，自动启用内置 Mock 语义打分模式")
        else:
            try:
                from huggingface_hub import InferenceClient

                self.client = InferenceClient(token=self.token)
                logger.info(
                    f"BGERerankerClient: 已初始化 HuggingFace InferenceClient (model={self.model})"
                )
            except Exception as e:
                logger.warning(f"初始化 HuggingFace InferenceClient 失败: {e}，优雅降级为 Mock 模式")
                self.is_mock = True

    def _extract_doc_text(self, doc: Dict[str, Any]) -> str:
        """从候选字典中提取代表文本，供与 query 进行 Cross-Encoder 匹配。"""
        if not isinstance(doc, dict):
            return str(doc)
        if doc.get("chunk_text"):
            return str(doc["chunk_text"])

        parts = []
        if doc.get("category"):
            parts.append(f"【类目】{doc['category']}")
        if doc.get("questions"):
            qs = doc["questions"]
            if isinstance(qs, list):
                qs = "\n".join(str(q) for q in qs)
            parts.append(f"【标准问法】{qs}")
        elif doc.get("question"):
            parts.append(f"【标准问法】{doc['question']}")
        if doc.get("answer"):
            parts.append(f"【解答】{doc['answer']}")

        if parts:
            return "\n".join(parts)
        if doc.get("content"):
            return str(doc["content"])
        if doc.get("text"):
            return str(doc["text"])
        return str(doc)

    def _compute_mock_score(self, query: str, doc_text: str) -> float:
        """基于词重叠 (Word Recall / Jaccard) 与字符匹配的确定性 Mock 打分算法。"""
        if not query or not doc_text:
            return 0.0

        q_clean = query.strip().lower()
        d_clean = doc_text.strip().lower()

        # 分词并过滤空白
        q_words = [w for w in jieba.cut(q_clean) if len(w.strip()) > 0]
        d_words = [w for w in jieba.cut(d_clean) if len(w.strip()) > 0]

        q_set = set(q_words)
        d_set = set(d_words)

        if not q_set:
            return 0.0

        # 词汇覆盖率 (Recall) 与 Jaccard 相似度
        intersect = q_set & d_set
        union = q_set | d_set
        recall = len(intersect) / len(q_set)
        jaccard = len(intersect) / len(union) if union else 0.0

        # 字符集合匹配
        q_chars = set(q_clean.replace(" ", ""))
        d_chars = set(d_clean.replace(" ", ""))
        char_recall = len(q_chars & d_chars) / len(q_chars) if q_chars else 0.0

        # 基础综合得分 (权重: 词覆盖 0.55, Jaccard 0.25, 字符覆盖 0.20)
        score = 0.55 * recall + 0.25 * jaccard + 0.20 * char_recall

        # 子串完全包含加权
        if q_clean in d_clean:
            score = min(1.0, score + 0.2)
        elif len(q_clean) > 4 and any(
            q_clean[i : i + 4] in d_clean for i in range(len(q_clean) - 3)
        ):
            score = min(1.0, score + 0.1)

        return round(float(score), 4)

    def _call_hf_api(self, query: str, doc_texts: List[str]) -> List[float]:
        """调用 HuggingFace Inference Router 执行批量 text-classification 推理。"""
        from huggingface_hub.inference._providers import get_provider_helper

        helper = get_provider_helper(None, task="text-classification", model=self.model)
        headers = getattr(self.client, "headers", {})
        req = helper.prepare_request(
            inputs="dummy",
            parameters={},
            headers=headers,
            model=self.model,
            api_key=self.token,
        )
        req.json = {
            "inputs": [
                {"text": query, "text_pair": doc_text}
                for doc_text in doc_texts
            ]
        }

        raw_res = self.client._inner_post(req)
        if isinstance(raw_res, bytes):
            res_json = json.loads(raw_res.decode("utf-8"))
        elif isinstance(raw_res, str):
            res_json = json.loads(raw_res)
        else:
            res_json = raw_res

        if isinstance(res_json, dict) and "error" in res_json:
            raise RuntimeError(f"HuggingFace API 错误: {res_json['error']}")

        scores: List[float] = []
        if isinstance(res_json, list):
            for item in res_json:
                if isinstance(item, list) and len(item) > 0:
                    scores.append(float(item[0].get("score", 0.0)))
                elif isinstance(item, dict):
                    scores.append(float(item.get("score", 0.0)))
                elif isinstance(item, (int, float)):
                    scores.append(float(item))
                else:
                    scores.append(0.0)
        elif isinstance(res_json, dict) and "score" in res_json:
            scores.append(float(res_json["score"]))

        if len(scores) != len(doc_texts):
            raise ValueError(
                f"API 返回分数数量 ({len(scores)}) 与输入文档数量 ({len(doc_texts)}) 不匹配"
            )

        return scores

    def _rerank_with_retry(self, query: str, doc_texts: List[str]) -> List[float]:
        """带 3 次指数退避重试与 Mock 兜底的推理调用。"""
        if self.is_mock or not self.client:
            return [self._compute_mock_score(query, text) for text in doc_texts]

        success = False
        last_err = None
        scores: List[float] = []

        for attempt in range(self.max_retries):
            try:
                scores = self._call_hf_api(query, doc_texts)
                success = True
                break
            except Exception as exc:
                last_err = exc
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"BGE-Reranker 推理异常（尝试 {attempt + 1}/{self.max_retries}）: {exc}，"
                        f"{sleep_time:.3f}s 后重试"
                    )
                    time.sleep(sleep_time)

        if not success:
            logger.warning(
                f"BGE-Reranker 重试 {self.max_retries} 次后仍失败 ({last_err})，优雅降级为 Mock 打分"
            )
            scores = [self._compute_mock_score(query, text) for text in doc_texts]

        return scores

    def rerank_sync(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """同步执行候选列表重排与打分。

        Args:
            query: 用户提问
            candidates: 待重排的文档字典列表
            top_k: 截断保留的最大条目数，默认 10

        Returns:
            按 rerank_score 降序排列的 Top-K 候选列表
        """
        if not candidates:
            return []

        doc_texts = [self._extract_doc_text(c) for c in candidates]
        scores = self._rerank_with_retry(query, doc_texts)

        scored_candidates = []
        for cand, score in zip(candidates, scores):
            item = dict(cand)
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """异步执行候选列表重排与打分。"""
        return await asyncio.to_thread(self.rerank_sync, query, candidates, top_k)
