# Task 2 Brief: 结构感知文档切分引擎

## 1. 任务目标
实现纯内存运行的结构感知 Markdown 切分引擎 `app/services/rag/splitter.py`，支持标题层级感知、超长内容递归切分、重叠区域句号对齐（不留半截话）、以及大表格按行切分且强制复制表头。

## 2. 涉及文件
- Create: `app/services/rag/splitter.py`
- Test: `tests/test_rag_splitter.py`

## 3. 全局约束
- 纯 Python 算法，不依赖外部网络和数据库，执行极速；
- 遵循 TDD：先编写失败测试 `tests/test_rag_splitter.py` -> 验证测试失败 -> 编写 `splitter.py` 实现 -> 验证测试全绿。

## 4. 关键算法与数据结构要求
### 4.1 数据类 `DocChunk`
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocChunk:
    category: str
    questions: str
    answer: str
    section_path: Optional[str] = None
    content_type: Optional[str] = "policy"
    is_key_clause: bool = False
    order_index: int = 0
```

### 4.2 标题层级感知
- 扫描 Markdown 标题行（`#` ~ `######`），用栈维护当前标题路径；
- `section_path`: 完整祖先路径（如 `商城服务指南 > 退货与退款政策 > 七天无理由退货`）；
- `category`: 祖先路径（如 `商城服务指南 > 退货与退款政策`）；
- `questions`: 当前小节标题（如 `七天无理由退货`），若遇到 `问：... 答：...` 形式则提取真实问题且 `content_type = "faq"`；
- 关键条款检测：标题或正文中若包含【重要提示】、【特别说明】、【注意】、【不可退换】等字样，将 `is_key_clause` 标记为 `True`。

### 4.3 句号对齐重叠切分算法 (不留半截话)
- 默认 `chunk_size = 500`，`overlap_size = 80`（支持初始化自定义）；
- 超长正文在分块时，后一块需要向前回退重叠；
- **核心对齐规则**：在重叠滑动区间内，向前或向后搜寻最近的断句标点符号（`。`、`！`、`？`、`；`、`\n`）；
- 切分起始点严格对齐至标点之后，确保后一块以完整句子开头，前一块以完整句子结束，不留半截词句。

### 4.4 大表格按行拆分与表头强制广播
- 检测 Markdown 表格块（首行 Header、次行 Divider `|---|---|`、数据行 Rows）；
- 当表格总长度或行数超过单块预算时，按数据行进行拆分；
- **强制规则**：每个切分出来的子 chunk，顶部必须完整保留原始表格的 Header 行和 Divider 行，然后再拼装当前切分包含的数据行。

## 5. 验收标准
- `pytest tests/test_rag_splitter.py -v` 全部通过；
- 覆盖层级感知、句号对齐重叠、大表格表头复制、关键条款识别四大核心能力；
- 全量 `pytest` 保持 100% 通过。
