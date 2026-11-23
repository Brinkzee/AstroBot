import asyncio
import hashlib
import logging
import os
import time
from typing import List, Optional

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class BGEEmbeddingClient:
    """BGE-M3 语义向量嵌入客户端。
    
    固定输出 1024 维 Dense 浮点向量，支持在线 HuggingFace Inference API 推理，
    内置重试、多批次批量处理、L2 归一化与离线/未配置 Token 时的确定性 Mock 优雅降级机制。
    """

    def __init__(
        self,
        token: Optional[str] = None,
        model: str = "BAAI/bge-m3",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        fallback_to_mock: bool = True,
        mock: bool = False,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_to_mock = fallback_to_mock

        if timeout is not None:
            self.timeout = float(timeout)
        else:
            resolved_timeout = (
                getattr(settings, "HUGGINGFACE_TIMEOUT", None)
                or getattr(settings, "huggingface_timeout", None)
                or os.getenv("HUGGINGFACE_TIMEOUT")
            )
            self.timeout = float(resolved_timeout) if resolved_timeout is not None else 15.0

        if mock or os.getenv("MOCK_EMBEDDING") == "1" or os.getenv("ASTRO_MOCK_EMBEDDING") == "1":
            self.token = "mock"
            self.is_mock = True
            self.client = None
            logger.info("BGEEmbeddingClient: 显式启用 Mock 向量模式")
            return

        # 解析 HuggingFace Token：参数 > settings.HUGGINGFACE_TOKEN > settings.huggingface_token > 环境变量
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
            logger.info("BGEEmbeddingClient: 未配置有效 Token 或显式指定 mock 模式，启用内置 Mock 向量模式")
        else:
            try:
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(token=self.token, timeout=self.timeout)
                logger.info(
                    f"BGEEmbeddingClient: 已初始化 HuggingFace InferenceClient (model={self.model}, timeout={self.timeout}s)"
                )
            except Exception as e:
                logger.warning(f"初始化 HuggingFace InferenceClient 失败: {e}，优雅降级为 Mock 模式")
                self.is_mock = True

    def _generate_mock_vector(self, text: str, dim: int = 1024) -> List[float]:
        """基于文本 MD5 散列生成确定性、已 L2 归一化的 1024 维 Dense 向量"""
        md5_bytes = hashlib.md5(text.encode("utf-8")).digest()
        seed = int.from_bytes(md5_bytes[:4], "little")
        rng = np.random.RandomState(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def _normalize_vector(self, vec: np.ndarray) -> List[float]:
        """对单个向量执行 L2 归一化并转换为标准 float 列表"""
        vec = vec.astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def _process_hf_response(self, raw_resp, expected_count: int, dim: int = 1024) -> List[List[float]]:
        """解析 HuggingFace feature_extraction 的返回值，兼容 1D/2D/3D ndarray 及变长嵌套序列输出。"""
        vectors: List[List[float]] = []

        try:
            arr = np.asarray(raw_resp, dtype=np.float32)
        except ValueError:
            # 兼容变长 token 序列列表: List[Array(seq_len_i, dim)]
            if isinstance(raw_resp, (list, tuple)) and len(raw_resp) == expected_count:
                for item in raw_resp:
                    item_arr = np.asarray(item, dtype=np.float32)
                    if item_arr.ndim == 1:
                        vectors.append(self._normalize_vector(item_arr))
                    elif item_arr.ndim == 2:
                        vectors.append(self._normalize_vector(np.mean(item_arr, axis=0)))
                    else:
                        raise ValueError(f"无法识别的单条特征提取维度: {item_arr.shape}")
            else:
                raise ValueError(
                    f"变长特征提取结果数量与预期不符: expected={expected_count}"
                )
        else:
            # 1D 输出 (单句直接返回 dim 维向量)
            if arr.ndim == 1:
                vectors = [self._normalize_vector(arr)]
            # 2D 输出
            elif arr.ndim == 2:
                if expected_count == 1:
                    # 可能是单个输入的 token 序列向量 (seq_len, dim)，采用平均池化
                    pooled = np.mean(arr, axis=0)
                    vectors = [self._normalize_vector(pooled)]
                else:
                    # 批量输出 (batch_size, dim)
                    vectors = [self._normalize_vector(row) for row in arr]
            # 3D 输出 (batch_size, seq_len, dim)，按 seq_len 做平均池化
            elif arr.ndim == 3:
                pooled = np.mean(arr, axis=1)
                vectors = [self._normalize_vector(row) for row in pooled]
            else:
                raise ValueError(f"无法识别的特征提取输出维度: {arr.shape}")

        if len(vectors) != expected_count:
            raise ValueError(
                f"特征提取输出条数 ({len(vectors)}) 与输入条数 ({expected_count}) 不匹配"
            )
        for idx, vec in enumerate(vectors):
            if len(vec) != dim:
                raise ValueError(
                    f"第 {idx} 条向量维度 ({len(vec)}) 与预期维度 ({dim}) 不匹配"
                )
        return vectors

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 16,
        fallback_to_mock: Optional[bool] = None,
    ) -> List[List[float]]:
        """批量生成文本的 1024 维 Dense 语义向量"""
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，当前值: {batch_size}")

        allow_fallback = self.fallback_to_mock if fallback_to_mock is None else fallback_to_mock
        all_vectors: List[List[float]] = []

        # 按 batch_size 分批处理
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if t is not None else "" for t in texts[i : i + batch_size]]

            if self.is_mock or not self.client:
                batch_vectors = [self._generate_mock_vector(t) for t in batch]
                all_vectors.extend(batch_vectors)
                continue

            # 真实 API 推理并带有网络重试和按需 Mock 兜底机制
            success = False
            last_err = None
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.feature_extraction(batch, model=self.model)
                    parsed_vectors = self._process_hf_response(resp, expected_count=len(batch))
                    all_vectors.extend(parsed_vectors)
                    success = True
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt < self.max_retries - 1:
                        sleep_time = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"BGE-M3 推理请求异常（尝试 {attempt + 1}/{self.max_retries}）: {exc}，{sleep_time:.2f}s 后重试"
                        )
                        time.sleep(sleep_time)

            if not success:
                if not allow_fallback:
                    raise RuntimeError(
                        f"BGE-M3 推理重试 {self.max_retries} 次后仍失败: {last_err}"
                    ) from last_err
                logger.warning(
                    f"BGE-M3 推理重试 {self.max_retries} 次后仍失败 ({last_err})，优雅降级为 Mock 向量兜底"
                )
                batch_vectors = [self._generate_mock_vector(t) for t in batch]
                all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """单句文本向量化，返回 1024 维 float 列表"""
        results = self.embed_documents([text], batch_size=1)
        return results[0]

    async def aembed_query(self, text: str) -> List[float]:
        """异步单句文本向量化"""
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """异步批量文本向量化"""
        return await asyncio.to_thread(self.embed_documents, texts, batch_size=batch_size)
