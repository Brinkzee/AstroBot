import pytest
from app.services.rag.splitter import MarkdownStructureSplitter, DocChunk


def test_markdown_hierarchy_splitting():
    md_content = """# 商城服务指南
## 退货与退款政策
### 七天无理由退货
在签收商品7天内，在不影响二次销售的前提下可申请无理由退货。
【特别说明】已激活的电子数码产品不支持无理由退货。

### 退货运费说明
| 责任方 | 运费承担 | 说明 |
|---|---|---|
| 买家原因 | 买家自理 | 个人喜好退换 |
| 质量问题 | 商家全额承担 | 需提供检测报告 |
| 偏远地区 | 双方协商 | 新疆西藏等特殊区域 |
"""
    splitter = MarkdownStructureSplitter(chunk_size=150, overlap_size=40)
    chunks = splitter.split_text(md_content)

    assert len(chunks) >= 2
    # 验证标题感知
    assert "退货与退款政策" in chunks[0].category
    assert "七天无理由退货" in chunks[0].questions
    # 验证完整路径
    assert chunks[0].section_path == "商城服务指南 > 退货与退款政策 > 七天无理由退货"
    # 验证特别说明识别为关键条款
    assert chunks[0].is_key_clause is True
    # 验证非关键条款块
    assert chunks[1].is_key_clause is False


def test_sentence_boundary_overlap_no_half_sentences():
    text = "# 规则\n## 细则\n第一句话非常明确完整。第二句话说明了退换货的截止时间与包裹寄回方式。第三句话强调了快递单号上传的必要性。"
    splitter = MarkdownStructureSplitter(chunk_size=45, overlap_size=20)
    chunks = splitter.split_text(text)

    # 验证切分出多个 chunk
    assert len(chunks) >= 2
    # 验证重叠区域不留半截话，切分点必在句末标点（。！？）之后
    for chunk in chunks:
        assert not chunk.answer.startswith("化说明了")
        assert not chunk.answer.startswith("截止时间")
        # 确保每个 chunk 的开头不是被腰斩的中间词
        assert chunk.answer.startswith("第一句话") or chunk.answer.startswith("第二句话") or chunk.answer.startswith("第三句话")


def test_table_splitting_preserves_header():
    table_md = """# 运费明细
## 资费标准
| 品类 | 基础运费 | 超重费 | 偏远补贴 | 备注说明 |
|---|---|---|---|---|
| 服饰鞋包 | 8元 | 2元/kg | 5元 | 满99包邮 |
| 家用电器 | 15元 | 5元/kg | 15元 | 大件送货上门 |
| 食品生鲜 | 12元 | 4元/kg | 不发货 | 冷链配送 |
| 电子数码 | 10元 | 3元/kg | 10元 | 顺丰特快 |
"""
    splitter = MarkdownStructureSplitter(chunk_size=90, overlap_size=0)
    chunks = splitter.split_text(table_md)

    assert len(chunks) > 1
    # 验证每一个切出来的表格 chunk 都完整保留了表头
    for chunk in chunks:
        assert "| 品类 | 基础运费 |" in chunk.answer
        assert "|---|---|" in chunk.answer


def test_faq_extraction_and_content_type():
    faq_md = """# 售后帮助
## 常见问答
问：退货运费由谁来承担？
答：商品若存在质量问题，运费由商家承担；个人原因退换由买家自理。

问：退款什么时候到账？
答：商家确认收货并验收无误后，款项将在1-3个工作日原路退回。
"""
    splitter = MarkdownStructureSplitter(chunk_size=300, overlap_size=30)
    chunks = splitter.split_text(faq_md)

    faq_chunks = [c for c in chunks if c.content_type == "faq"]
    assert len(faq_chunks) == 2
    assert "退货运费由谁来承担" in faq_chunks[0].questions
    assert "质量问题" in faq_chunks[0].answer
    assert "退款什么时候到账" in faq_chunks[1].questions
    assert "1-3个工作日" in faq_chunks[1].answer


def test_key_clause_detection():
    normal_md = "# 政策\n## 普通条款\n这是一段普通的政策说明，无特殊限制。"
    warning_md = "# 政策\n## 重要提醒\n【重要提示】定制商品一经售出，概不退换。"
    notice_md = "# 政策\n## 贴心提示\n【注意】生鲜冷冻商品签收时请当面验货。"

    splitter = MarkdownStructureSplitter()
    c_normal = splitter.split_text(normal_md)
    c_warning = splitter.split_text(warning_md)
    c_notice = splitter.split_text(notice_md)

    assert c_normal[0].is_key_clause is False
    assert c_warning[0].is_key_clause is True
    assert c_notice[0].is_key_clause is True


def test_order_index_sequential():
    md = """# 指南
## 章节一
内容一。内容二。内容三。
## 章节二
内容四。内容五。内容六。
"""
    splitter = MarkdownStructureSplitter(chunk_size=30, overlap_size=0)
    chunks = splitter.split_text(md)
    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.order_index == idx


def test_empty_and_no_heading_markdown():
    splitter = MarkdownStructureSplitter()
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n\n  ") == []

    raw_text = "这是一段没有各级标题的纯文本内容。系统应当具备兜底能力，赋予其默认分类与常规说明问法。"
    chunks = splitter.split_text(raw_text)
    assert len(chunks) == 1
    assert chunks[0].category == "未分类"
    assert chunks[0].questions == "正文"
    assert "纯文本内容" in chunks[0].answer
