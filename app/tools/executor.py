import asyncio
import json
import logging
from typing import Any, Dict, Optional
from pydantic import ValidationError

from app.tools.registry import ToolRegistry, default_tool_registry

logger = logging.getLogger(__name__)


def _build_failure_response(
    tool_name: str,
    tool_call_id: str,
    error_detail: str,
) -> Dict[str, Any]:
    """构造标准化的工具执行失败响应结构，适配异常优雅回灌给模型"""
    return {
        "success": False,
        "output": f"工具 [{tool_name}] 调用失败: {error_detail}，请结合此情况向用户做解释并提供帮助",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "error": error_detail,
    }


class ToolExecutor:
    """具备参数校验、超时控制、自动重试与异常优雅回灌的工具执行器"""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        timeout: float = 5.0,
        max_retries: int = 1,
    ) -> None:
        self.registry = registry if registry is not None else default_tool_registry
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    async def execute(
        self,
        tool_call: Dict[str, Any],
        db: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行单次工具调用并返回结构化结果

        Args:
            tool_call: 工具调用字典，包含 'name', 'args', 'id'
            db: 可选的数据库会话（保留扩展性）

        Returns:
            {
                "success": bool,
                "output": str,
                "tool_name": str,
                "tool_call_id": str,
                "error": Optional[str],
            }
        """
        tool_name = str(tool_call.get("name") or "unknown_tool")
        tool_call_id = str(tool_call.get("id") or "")
        raw_args = tool_call.get("args")

        # 1. 参数格式解析与预处理
        if raw_args is None:
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception as e:
                error_detail = f"参数 JSON 反序列化失败: {e}"
                logger.warning(f"Tool [{tool_name}] args JSON parse error: {e}")
                return _build_failure_response(tool_name, tool_call_id, error_detail)
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            error_detail = f"参数类型错误: 期望字典或 JSON 字符串，实际为 {type(raw_args).__name__}"
            return _build_failure_response(tool_name, tool_call_id, error_detail)

        # 2. 工具查找与存在性校验
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            error_detail = f"未找到工具: '{tool_name}'"
            logger.warning(f"Tool not found: {tool_name}")
            return _build_failure_response(tool_name, tool_call_id, error_detail)

        # 3. 参数 Schema 校验 (Pydantic ValidationError 捕获与格式化)
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None:
            try:
                if hasattr(args_schema, "model_validate"):
                    validated_model = args_schema.model_validate(args)
                    if hasattr(validated_model, "model_dump"):
                        validated_args = validated_model.model_dump()
                    else:
                        validated_args = dict(validated_model)
                elif callable(args_schema):
                    validated_model = args_schema(**args)
                    if hasattr(validated_model, "model_dump"):
                        validated_args = validated_model.model_dump()
                    elif hasattr(validated_model, "dict"):
                        validated_args = validated_model.dict()
                    else:
                        validated_args = dict(validated_model)
                else:
                    validated_args = args
            except (ValidationError, Exception) as e:
                error_detail = f"参数校验失败: {e}"
                logger.warning(f"Tool [{tool_name}] schema validation failed: {e}")
                return _build_failure_response(tool_name, tool_call_id, error_detail)
        else:
            validated_args = args

        # 4. 超时控制与退避重试执行循环
        last_error: Optional[Exception] = None
        total_attempts = 1 + max(0, self.max_retries)

        for attempt in range(total_attempts):
            try:
                # 统一调用工具：优先使用 LangChain 的 ainvoke（自动支持同步与异步工具）
                if hasattr(tool, "ainvoke"):
                    invoke_coro = tool.ainvoke(validated_args)
                elif hasattr(tool, "invoke"):
                    invoke_coro = asyncio.to_thread(tool.invoke, validated_args)
                elif callable(tool):
                    if asyncio.iscoroutinefunction(tool):
                        invoke_coro = tool(**validated_args)
                    else:
                        invoke_coro = asyncio.to_thread(tool, **validated_args)
                else:
                    raise TypeError(f"工具对象 {tool} 不可调用")

                raw_output = await asyncio.wait_for(invoke_coro, timeout=self.timeout)

                # 格式化输出为字符串（供 ToolMessage.content 使用）
                if isinstance(raw_output, (dict, list)):
                    output_str = json.dumps(raw_output, ensure_ascii=False)
                else:
                    output_str = str(raw_output)

                return {
                    "success": True,
                    "output": output_str,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "error": None,
                }

            except (asyncio.TimeoutError, TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"Tool [{tool_name}] (call_id: {tool_call_id}) attempt {attempt + 1}/{total_attempts} timed out after {self.timeout}s"
                )
                if attempt < total_attempts - 1:
                    await asyncio.sleep(0.05)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Tool [{tool_name}] (call_id: {tool_call_id}) attempt {attempt + 1}/{total_attempts} failed with error: {e}"
                )
                if attempt < total_attempts - 1:
                    await asyncio.sleep(0.05)

        # 5. 重试耗尽，优雅回灌失败响应
        if isinstance(last_error, (asyncio.TimeoutError, TimeoutError)):
            error_detail = f"执行超时 (超过 {self.timeout} 秒)"
        elif last_error is not None:
            error_detail = f"{type(last_error).__name__}: {last_error}"
        else:
            error_detail = "未知执行错误"

        return _build_failure_response(tool_name, tool_call_id, error_detail)
