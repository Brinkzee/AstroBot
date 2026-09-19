from typing import Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    openai_api_key: str = Field(default="sk-placeholder", description="OpenAI 兼容模型密钥")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI 兼容端点 Base URL")
    openai_api_base: Optional[str] = Field(default=None, description="OpenAI 兼容端点 Base URL 别名")
    openai_model_name: str = Field(default="deepseek-flash", description="模型名称，可配置 deepseek-flash、deepseek-chat 等")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_context_tokens: int = Field(default=2000, gt=0, description="上下文历史最大保留 Token 预算")
    database_url: str = Field(
        default="mysql+aiomysql://root:root123456@127.0.0.1:3306/astro_bot?charset=utf8mb4",
        description="MySQL 异步连接串"
    )
    milvus_uri: str = Field(
        default="./data/milvus/astro_bot.db",
        description="Milvus-Lite 本地数据文件路径或独立 Milvus 连接 URI"
    )
    huggingface_token: Optional[str] = Field(
        default=None,
        description="HuggingFace API Token 用于在线 BGE-M3 推理"
    )
    huggingface_timeout: float = Field(
        default=15.0,
        gt=0.0,
        description="HuggingFace API 推理请求超时时间（秒），适配公网及跨境访问延迟"
    )
    reranker_batch_size: int = Field(
        default=4,
        gt=0,
        description="BGE-Reranker 批量推理批次大小，降低单请求负载与推理耗时"
    )
    reranker_circuit_breaker_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="BGE-Reranker 连续异常熔断保护窗口（秒），避免频繁阻塞并支持快速自愈"
    )
    model_context_window: int = Field(default=128000, description="模型物理上下文窗口")
    max_output_tokens: int = Field(default=2000, description="模型回复预留最大输出 Token")
    max_user_input_tokens: int = Field(default=2000, description="单轮用户输入预估峰值")
    max_agent_steps: int = Field(default=3, description="ReAct Agent 最大迭代步数")
    tool_result_max_tokens: int = Field(default=1200, description="单步工具结果最大预留 Token")
    rerank_top_k: int = Field(default=5, description="检索召回文档篇数")
    system_prompt_tokens: int = Field(default=1500, description="系统人设与工具定义固定开销")
    doc_tokens_per_chunk: int = Field(default=500, description="单篇检索文档估算 Token")
    summary_max_tokens: int = Field(default=250, description="注入梗概上限预留")
    safety_margin_tokens: int = Field(default=500, description="安全缓冲余量")
    target_history_turns: int = Field(default=20, description="期望留存历史轮数")
    steady_turn_tokens: int = Field(default=500, description="单轮稳态 Token 占用")
    mcp_logistics_server_url: str = Field(
        default="http://127.0.0.1:8001/mcp",
        description="物流 MCP Server URL"
    )
    mcp_aftersale_server_url: str = Field(
        default="http://127.0.0.1:8002/mcp",
        description="售后 MCP Server URL"
    )
    mcp_client_timeout: float = Field(
        default=5.0,
        gt=0.0,
        description="MCP Client 连接与调用超时时间（秒）"
    )
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse Public Key")
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse Secret Key")
    langfuse_host: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("langfuse_host", "langfuse_base_url", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
        description="Langfuse Host URL"
    )
    langfuse_enabled: bool = Field(default=True, description="Whether Langfuse tracing is enabled")
    evidence_confidence_threshold: float = Field(default=0.40, description="置信度闸门判定阈值，由评估集网格搜索校准推荐")

    @property
    def MCP_LOGISTICS_SERVER_URL(self) -> str:
        return self.mcp_logistics_server_url

    @property
    def MCP_AFTERSALE_SERVER_URL(self) -> str:
        return self.mcp_aftersale_server_url

    @property
    def MCP_CLIENT_TIMEOUT(self) -> float:
        return self.mcp_client_timeout

    @property
    def MILVUS_URI(self) -> str:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.milvus_uri

    @property
    def HUGGINGFACE_TOKEN(self) -> Optional[str]:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.huggingface_token

    @property
    def HUGGINGFACE_TIMEOUT(self) -> float:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.huggingface_timeout

    @property
    def RERANKER_BATCH_SIZE(self) -> int:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.reranker_batch_size

    @property
    def RERANKER_CIRCUIT_BREAKER_SECONDS(self) -> float:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.reranker_circuit_breaker_seconds

    @property
    def MODEL_CONTEXT_WINDOW(self) -> int:
        return self.model_context_window

    @property
    def MAX_OUTPUT_TOKENS(self) -> int:
        return self.max_output_tokens

    @property
    def MAX_USER_INPUT_TOKENS(self) -> int:
        return self.max_user_input_tokens

    @property
    def MAX_AGENT_STEPS(self) -> int:
        return self.max_agent_steps

    @property
    def TOOL_RESULT_MAX_TOKENS(self) -> int:
        return self.tool_result_max_tokens

    @property
    def RERANK_TOP_K(self) -> int:
        return self.rerank_top_k

    @property
    def SYSTEM_PROMPT_TOKENS(self) -> int:
        return self.system_prompt_tokens

    @property
    def DOC_TOKENS_PER_CHUNK(self) -> int:
        return self.doc_tokens_per_chunk

    @property
    def SUMMARY_MAX_TOKENS(self) -> int:
        return self.summary_max_tokens

    @property
    def SAFETY_MARGIN_TOKENS(self) -> int:
        return self.safety_margin_tokens

    @property
    def TARGET_HISTORY_TURNS(self) -> int:
        return self.target_history_turns

    @property
    def STEADY_TURN_TOKENS(self) -> int:
        return self.steady_turn_tokens

    @property
    def effective_base_url(self) -> Optional[str]:
        """统一解析有效的 base_url，优先使用 openai_base_url，其次 fallback 到 openai_api_base，默认回退至 https://api.deepseek.com"""
        return self.openai_base_url or self.openai_api_base or "https://api.deepseek.com"

settings = Settings()
