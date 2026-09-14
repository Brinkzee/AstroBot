import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Set, Tuple
import httpx
import jsonschema
from pydantic import ValidationError

from app.tools.registry import ToolRegistry, default_tool_registry

logger = logging.getLogger(__name__)

# 可重试异常白名单：仅限网络物理与连接瞬时抖动
RETRYABLE_EXCEPTIONS: Tuple[type, ...] = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionResetError,
    ConnectionRefusedError,
    ConnectionAbortedError,
    BrokenPipeError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
    httpx.TimeoutException,
)

# 状态字段键名（用于枚举翻译定位）
STATUS_KEYS: Set[str] = {
    "status",
    "state",
    "order_status",
    "订单状态",
    "物流状态",
    "质保状态",
    "状态",
    "sign_status",
    "warranty_status",
    "ticket_status",
}

# 内部状态枚举人话翻译映射表
STATUS_VALUE_TRANSLATIONS: Dict[str, str] = {
    "SHIPPED": "已发货",
    "PENDING": "待支付",
    "PAID": "已支付",
    "SIGNED": "已签收",
    "CANCELLED": "已取消",
    "COMPLETED": "已完成",
    "FINISHED": "已完成",
    "UNPAID": "未支付",
    "DELIVERING": "派送中",
    "TRANSPORTING": "运输中",
    "RECEIVED": "已揽收",
    "ACTIVE": "在保",
    "WARRANTY_ACTIVE": "在保",
    "EXPIRED": "已过保",
    "WARRANTY_EXPIRED": "已过保",
    "PROCESSING": "处理中",
    "REFUNDING": "退款处理中",
    "REFUNDED": "退款已完成",
    "AGREED": "已同意",
    "REJECTED": "已拒绝",
}


def _translate_enums_recursive(obj: Any, parent_key: Optional[str] = None) -> Any:
    """递归遍历数据结构并将英文/数字状态枚举翻译为人话"""
    if isinstance(obj, dict):
        return {k: _translate_enums_recursive(v, parent_key=k) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_translate_enums_recursive(item, parent_key=parent_key) for item in obj]
    elif isinstance(obj, str):
        stripped = obj.strip()
        upper = stripped.upper()
        if upper in STATUS_VALUE_TRANSLATIONS:
            return STATUS_VALUE_TRANSLATIONS[upper]
        if parent_key and parent_key.lower() in STATUS_KEYS and stripped == "1":
            return "已发货"
        return obj
    elif isinstance(obj, int):
        if parent_key and parent_key.lower() in STATUS_KEYS and obj == 1:
            return "已发货"
        return obj
    return obj


def _format_output(raw_output: Any) -> str:
    """结果人话翻译与中文字符不转义序列化（ensure_ascii=False）"""
    if hasattr(raw_output, "model_dump"):
        raw_output = raw_output.model_dump()
    elif hasattr(raw_output, "dict") and callable(raw_output.dict):
        raw_output = raw_output.dict()

    if isinstance(raw_output, (dict, list)):
        translated = _translate_enums_recursive(raw_output)
        return json.dumps(translated, ensure_ascii=False)

    if isinstance(raw_output, str):
        text = raw_output.strip()
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, (dict, list)):
                    translated = _translate_enums_recursive(parsed)
                    return json.dumps(translated, ensure_ascii=False)
            except Exception:
                pass
        return raw_output

    return str(raw_output)


def _is_query_missed(raw_output: Any, output_str: str) -> bool:
    """分诊：判断工具执行结果是否为业务空结果（查询落空）"""
    if raw_output is None:
        return True
    if isinstance(raw_output, (list, dict, set, tuple)) and len(raw_output) == 0:
        return True

    text = output_str.strip()
    if not text or text in ("{}", "[]", "null", "None"):
        return True

    if isinstance(raw_output, dict):
        if raw_output.get("found") is False or raw_output.get("exists") is False:
            return True
        if raw_output.get("total") == 0 or raw_output.get("count") == 0:
            return True
        if "error" in raw_output and isinstance(raw_output["error"], str):
            err_msg = raw_output["error"]
            if any(k in err_msg for k in ("未找到", "不存在", "查无", "无相关", "未查询到")):
                return True

    missed_keywords = (
        "未找到",
        "未查到",
        "查无",
        "不存在",
        "无相关",
        "暂无",
        "没有找到",
        "不在保修期",
        "无退货记录",
        "无记录",
        "未查询到",
        "空结果",
        "not found",
    )
    for kw in missed_keywords:
        if kw in text:
            return True

    return False


def _is_write_tool(tool: Any, tool_name: str, **kwargs: Any) -> bool:
    """判断工具是否为写操作（写操作严禁自动重试）"""
    if kwargs.get("is_write") is True:
        return True
    if tool_name == "create_ticket":
        return True
    if getattr(tool, "is_write", False) is True:
        return True
    if getattr(tool, "write", False) is True:
        return True
    if getattr(tool, "tool_type", "").upper() == "WRITE":
        return True
    if getattr(tool, "metadata", None) and getattr(tool, "metadata", {}).get("is_write") is True:
        return True
    tags = getattr(tool, "tags", None) or []
    if "write" in tags:
        return True
    return False


def _format_pydantic_error(e: ValidationError) -> str:
    """提取 Pydantic ValidationError 并转为清晰友好的中文说明"""
    messages = []
    for err in e.errors():
        field_path = ".".join(str(loc) for loc in err.get("loc", []) if loc != "__root__")
        err_type = err.get("type", "")
        msg = err.get("msg", "")

        if not field_path:
            messages.append(msg)
            continue

        if err_type == "missing":
            messages.append(f"缺少必填参数 '{field_path}'")
        elif "greater_than" in err_type or "less_than" in err_type:
            messages.append(f"参数 '{field_path}' 数值越界: {msg}")
        elif "type" in err_type:
            messages.append(f"参数 '{field_path}' 类型不匹配: {msg}")
        else:
            messages.append(f"参数 '{field_path}' 校验未通过: {msg}")
    return "; ".join(messages) if messages else str(e)


def _format_jsonschema_error(e: jsonschema.ValidationError) -> str:
    """提取 jsonschema 校验错误信息"""
    path = ".".join(str(p) for p in e.path)
    if e.validator == "required":
        return f"缺少必填参数: {e.message}"
    if path:
        return f"参数 '{path}' 校验未通过: {e.message}"
    return f"参数校验未通过: {e.message}"


def _build_failure_response(
    tool_name: str,
    tool_call_id: str,
    error_detail: str,
    status: str = "失败",
    error_type: Optional[str] = "SYSTEM_FAULT",
    output_override: Optional[str] = None,
    retry_count: int = 0,
    duration_ms: int = 0,
) -> Dict[str, Any]:
    """构造标准化的工具执行失败响应结构，适配异常优雅回灌给模型"""
    if output_override is not None:
        output_str = output_override
    elif status == "校验拦下":
        output_str = f"工具 [{tool_name}] 调用失败: 参数校验未通过: {error_detail}，请结合此情况向用户做解释或追问补充必要信息"
    else:
        output_str = f"工具 [{tool_name}] 调用失败: {error_detail}，请结合此情况向用户做解释并提供帮助"

    return {
        "success": False,
        "output": output_str,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "status": status,
        "error": error_detail,
        "error_type": error_type,
        "retry_count": retry_count,
        "duration_ms": duration_ms,
    }


class ToolExecutor:
    """具备参数 Schema 校验、超时控制、网络白名单退避重试、错误分诊与脱敏格式化的工具执行中枢"""

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
            **kwargs: 扩展参数，支持 is_write 等覆盖

        Returns:
            {
                "success": bool,
                "output": str,
                "tool_name": str,
                "tool_call_id": str,
                "status": str,  # '成功', '失败', '超时', '校验拦下', '权限拒绝'
                "error": Optional[str],
                "error_type": Optional[str],  # 'INVALID_ARGS', 'QUERY_MISSED', 'SYSTEM_FAULT', None
                "retry_count": int,
                "duration_ms": int,
            }
        """
        start_time = time.perf_counter()
        tool_name = str(tool_call.get("name") or "unknown_tool")
        tool_call_id = str(tool_call.get("id") or "")
        raw_args = tool_call.get("args")

        # 1. 参数格式解析与预处理（反序列化失败直接拦下，不发起底层调用）
        if raw_args is None:
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception as e:
                error_detail = f"参数 JSON 反序列化失败: {e}"
                logger.warning(f"Tool [{tool_name}] args JSON parse error: {e}")
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return _build_failure_response(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    error_detail=error_detail,
                    status="校验拦下",
                    error_type="INVALID_ARGS",
                    retry_count=0,
                    duration_ms=duration_ms,
                )
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            error_detail = f"参数类型错误: 期望字典或 JSON 字符串，实际为 {type(raw_args).__name__}"
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return _build_failure_response(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                error_detail=error_detail,
                status="校验拦下",
                error_type="INVALID_ARGS",
                retry_count=0,
                duration_ms=duration_ms,
            )

        # 2. 工具查找与存在性校验
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            error_detail = f"未找到工具: '{tool_name}'"
            logger.warning(f"Tool not found: {tool_name}")
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            return _build_failure_response(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                error_detail=error_detail,
                status="失败",
                error_type="SYSTEM_FAULT",
                retry_count=0,
                duration_ms=duration_ms,
            )

        # 3. 参数 Schema 校验 (校验必填项缺失、类型错误、值域越界)
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None and hasattr(tool, "get_input_schema"):
            try:
                args_schema = tool.get_input_schema()
            except Exception:
                args_schema = None

        if args_schema is not None:
            try:
                if isinstance(args_schema, dict):
                    jsonschema.validate(instance=args, schema=args_schema)
                    validated_args = args
                elif hasattr(args_schema, "model_validate"):
                    validated_model = args_schema.model_validate(args)
                    if hasattr(validated_model, "model_dump"):
                        validated_args = validated_model.model_dump()
                    else:
                        validated_args = dict(validated_model)
                elif hasattr(args_schema, "parse_obj"):
                    validated_model = args_schema.parse_obj(args)
                    if hasattr(validated_model, "dict"):
                        validated_args = validated_model.dict()
                    else:
                        validated_args = dict(validated_model)
                elif callable(args_schema) and isinstance(args_schema, type):
                    validated_model = args_schema(**args)
                    if hasattr(validated_model, "model_dump"):
                        validated_args = validated_model.model_dump()
                    elif hasattr(validated_model, "dict"):
                        validated_args = validated_model.dict()
                    else:
                        validated_args = dict(validated_model)
                else:
                    validated_args = args
            except ValidationError as e:
                error_detail = f"参数校验失败: {_format_pydantic_error(e)}"
                logger.warning(f"Tool [{tool_name}] schema validation failed: {error_detail}")
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return _build_failure_response(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    error_detail=error_detail,
                    status="校验拦下",
                    error_type="INVALID_ARGS",
                    retry_count=0,
                    duration_ms=duration_ms,
                )
            except jsonschema.ValidationError as e:
                error_detail = _format_jsonschema_error(e)
                logger.warning(f"Tool [{tool_name}] jsonschema validation failed: {error_detail}")
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return _build_failure_response(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    error_detail=error_detail,
                    status="校验拦下",
                    error_type="INVALID_ARGS",
                    retry_count=0,
                    duration_ms=duration_ms,
                )
            except Exception as e:
                error_detail = f"参数校验异常: {e}"
                logger.warning(f"Tool [{tool_name}] schema validation error: {error_detail}")
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                return _build_failure_response(
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    error_detail=error_detail,
                    status="校验拦下",
                    error_type="INVALID_ARGS",
                    retry_count=0,
                    duration_ms=duration_ms,
                )
        else:
            validated_args = args

        # 4. 判断读写操作与重试白名单控制
        is_write = _is_write_tool(tool, tool_name, **kwargs)
        # 写操作绝对不自动重试 (Hard Gate: retry_count 恒为 0)
        total_retries_allowed = 0 if is_write else max(0, self.max_retries)
        total_attempts = 1 + total_retries_allowed
        actual_retries = 0
        last_error: Optional[Exception] = None

        for attempt in range(total_attempts):
            if attempt > 0:
                actual_retries += 1

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
                duration_ms = int((time.perf_counter() - start_time) * 1000)

                output_str = _format_output(raw_output)
                is_missed = _is_query_missed(raw_output, output_str)

                # 查询落空（业务空结果）标记为成功且 error_type="QUERY_MISSED"，严禁重试
                error_type = "QUERY_MISSED" if is_missed else None
                retry_cnt = 0 if is_missed else actual_retries

                return {
                    "success": True,
                    "output": output_str,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "status": "成功",
                    "error": None,
                    "error_type": error_type,
                    "retry_count": retry_cnt,
                    "duration_ms": duration_ms,
                }

            except (asyncio.TimeoutError, TimeoutError) as e:
                last_error = e
                logger.warning(
                    f"Tool [{tool_name}] (call_id: {tool_call_id}) attempt {attempt + 1}/{total_attempts} timed out after {self.timeout}s"
                )
                if not is_write and attempt < total_attempts - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    break

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Tool [{tool_name}] (call_id: {tool_call_id}) attempt {attempt + 1}/{total_attempts} failed with error: {e}"
                )
                # 仅限网络瞬时抖动白名单异常重试，写操作或业务异常坚决不重试
                if not is_write and isinstance(e, RETRYABLE_EXCEPTIONS) and attempt < total_attempts - 1:
                    await asyncio.sleep(0.1)
                    continue
                else:
                    break

        # 5. 重试耗尽或不可重试真故障 (SYSTEM_FAULT)
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        if isinstance(last_error, (asyncio.TimeoutError, TimeoutError)):
            error_detail = f"执行超时 (超过 {self.timeout} 秒)"
            status = "超时"
        elif last_error is not None:
            error_detail = f"{type(last_error).__name__}: {last_error}"
            status = "失败"
        else:
            error_detail = "未知执行错误"
            status = "失败"

        return _build_failure_response(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            error_detail=error_detail,
            status=status,
            error_type="SYSTEM_FAULT",
            retry_count=0 if is_write else actual_retries,
            duration_ms=duration_ms,
        )


# 默认工具执行器单例 (适配复杂混合检索/重排链，超时设为 15.0s)
default_tool_executor = ToolExecutor(registry=default_tool_registry, timeout=15.0)
