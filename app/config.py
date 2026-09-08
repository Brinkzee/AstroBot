from typing import Optional
from pydantic import Field
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
    openai_model_name: str = Field(default="gpt-4o-mini", description="模型名称，可配置 deepseek-chat、gpt-4o 等")
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

    @property
    def MILVUS_URI(self) -> str:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.milvus_uri

    @property
    def HUGGINGFACE_TOKEN(self) -> Optional[str]:
        """保持与 SDD 规范中大写属性名称的完全兼容"""
        return self.huggingface_token

    @property
    def effective_base_url(self) -> Optional[str]:
        """统一解析有效的 base_url，优先使用 openai_base_url，其次 fallback 到 openai_api_base"""
        return self.openai_base_url or self.openai_api_base

settings = Settings()
