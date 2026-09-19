# 第十章：客服系统多标签主题分类器微调、ONNX独立推理服务、旁路批处理与九项实证验收设计规范

本文档为第十章技术设计规范（Spec）。本章旨在为客服系统微调一个自己的中文多标签主题分类器（基于哈工大讯飞 `chinese-roberta-wwm-ext` 全参微调），将数据飞轮攒下的低置信度问题按 17 类权威主题批量归类，赋能飞轮后台精准决策「先补哪块知识」，并通过独立 ONNX 推理服务、旁路批处理流水线与九项实证验收看板构建完整的生产级闭环。

---

## 1. 17 类权威术语表与多标签规则

### 1.1 17 类权威类目与模糊近邻边界定义
全系统统一一份 17 类权威类目表，退换货、物流、尺码、发票领头，并通过边界说明将近邻类目严格划开：

| 序号 | 类目名称 | 核心定义与覆盖范围 | 近邻边界划分说明 |
| :--- | :--- | :--- | :--- |
| 1 | **退换货** | 7天无理由退货、换货流程、退款到账时效、退货地址 | **修归保修维修、退归退换货** |
| 2 | **物流** | 快递轨迹、发货时效、包裹破损/滞留、催发货 | **物流管货、运费管钱** |
| 3 | **尺码** | 服装/鞋帽/用品规格尺寸推荐、偏大偏小偏肥偏瘦 | - |
| 4 | **发票** | 电子发票、专票/普票开具、发票抬头修改与重开 | - |
| 5 | **质量问题** | 破损、变质、瑕疵、做工粗糙、异味、异物 | 与商品本身缺陷相关；退货诉求同时打退换货 |
| 6 | **运费** | 运费险抵扣、退换货运费谁承担、偏远地区补运费 | **运费管钱、物流管货** |
| 7 | **优惠活动** | 店铺满减、优惠券领取/核销、限时折扣、拼团活动 | **优惠活动是券和满减、价保是补差价** |
| 8 | **价保** | 购买后降价申请退差额、价保有效期咨询 | **价保是补差价、优惠活动是券和满减** |
| 9 | **支付** | 微信/支付宝/银行卡扣款失败、重复扣款、花呗分期 | - |
| 10 | **订单修改** | 修改收货地址/电话、变更颜色尺码、取消订单 | - |
| 11 | **库存补货** | 缺货催上架、补货到货通知、预售发货时间 | - |
| 12 | **商品信息** | 保质期、成分材质、产地、使用说明、适用对象 | - |
| 13 | **保修维修** | 质保期内免费维修、售后检测、更换零配件 | **修归保修维修、退归退换货** |
| 14 | **账号** | 登录受限、手机号解绑/换绑、密码重置、注销 | - |
| 15 | **会员积分** | 积分抵扣、会员升级、积分清零规则 | - |
| 16 | **评价** | 修改评价、中差评申诉、晒单返现 | - |
| 17 | **其他** | 无法归入上述 16 类的通用或边缘问答 | - |

### 1.2 多标签判定原则
- **字面诉求对齐**：一句话字面提到几个诉求就打几个标签，一个不多一个不少。
  - 例 1：“买大了想退” $\rightarrow$ `["尺码", "退换货"]`
  - 例 2：“刚下单就降价了，麻烦帮我改个地址或者补差价” $\rightarrow$ `["价保", "订单修改"]`
  - 例 3：“寄过来包装烂了，运费谁出，我要换一个” $\rightarrow$ `["质量问题", "运费", "退换货"]`

---

## 2. 数据处理流水线与考卷泄漏自检硬闸

### 2.1 四步处理流程
1. **清洗（Clean）**：脱敏（手机号、姓名、订单号替换为占位符）、修错别字（如“降介” $\rightarrow$ “降价”）、格式规范化。
2. **归并（Map）**：对齐 17 类权威类目表，确保无脏标签。
3. **分层抽样（Stratified Split）**：按 80/10/10 划分为训练集、验证集、测试集，各类目按比例进入三份数据。
4. **训练集数据增强（Augmentation）**：**只扩训练集**。通过同义词替换（“差价”$\leftrightarrow$“差额”）与句式微调（疑问句倒装、语气助词调整）进行扩充，严格禁止污染验证集和测试集。

### 2.2 语料存储与自检
- 权威语料目录：`mewhelp-ch10-dataset/`（`train.jsonl` 2594 条、`val.jsonl` 161 条、`test.jsonl` 161 条）。
- **三份考卷泄漏自检（硬闸 Hard Gate）**：
  $$\text{Text}(train) \cap \text{Text}(val) = \emptyset$$
  $$\text{Text}(train) \cap \text{Text}(test) = \emptyset$$
  $$\text{Text}(val) \cap \text{Text}(test) = \emptyset$$
  重叠文本数必须严格为 0。若存在交叉泄漏，验收状态直接判 `fail`，后续所有评测分数不作数。

---

## 3. 模型微调、正则化与早停

### 3.1 训练架构与超参数
- **基座模型**：`hfl/chinese-roberta-wwm-ext`（哈工大讯飞中文预训练模型）。
- **微调方式**：**全参微调（Full Fine-Tuning）**，不使用 LoRA / QLoRA。
- **输出层**：17 维线性分类头，搭配多标签损失函数 `BCEWithLogitsLoss`（各类别独立 Sigmoid 概率计算）。
- **优化与正则化**：
  - 优化器：`AdamW`，学习率 $2 \times 10^{-5}$，权重衰减 `weight_decay = 0.01`（$L_2$ 正则化防过拟合）。
  - 学习率调度：带预热的线性衰减（warmup ratio = 0.1）。
  - Batch Size：16 或 32。
- **早停机制（Early Stopping）**：
  - 监控指标：`val_loss` 与 `val_macro_f1`。
  - 耐心轮数：`patience = 3`。连续 3 个 Epoch 无提升则终止训练并自动保存最优模型至 `data/ch10/best_model/`。

---

## 4. 严密评测、分档红线与错因归因

### 4.1 容错红线按档位设闸 (Tiered Redlines)
按「归类错误是否会带偏知识库补齐的优先级」设闸：

| 档位 | 包含类目 (共 17 类) | F1 红线指标 | 验收卡片呈现与达标逻辑 |
| :--- | :--- | :--- | :--- |
| **严 (Strict)** | 退换货、物流、尺码、发票、质量问题 (5 类) | **F1 $\ge 0.90$** | 达标标 ✅；不达标标红并提示「先回头搞数据」 |
| **中 (Medium)** | 运费、优惠活动、价保、支付、订单修改、库存补货、商品信息、保修维修 (8 类) | **F1 $\ge 0.80$** | 达标标 ✅；不达标标红并提示「先回头搞数据」 |
| **宽 (Loose)** | 账号、会员积分、评价、其他 (4 类) | **不设线** | 状态列统一画 **`—`**（严禁标 ✅，避免误导） |

### 4.2 判错样本导出与错因三向归因 (`error_samples.md`)
测试集上预测 $\neq$ 标准答案的样本落 `data/ch10/reports/error_samples.md`，错因分类如下：
- **漏打（放跑）**：真实有、预测无，记 1 笔放跑（真实类目 FN +1）；
- **多打（冤枉）**：真实无、预测有，记 1 笔冤枉（误判类目 FP +1）；
- **错位（位移）**：真实该打 A 类没打、反而打了 B 类。**一次记 2 笔**（A 类放跑 FN +1，B 类冤枉 FP +1）；
- **错例与混淆矩阵笔数平衡**：
  $$\text{Total Errors} = \text{漏打条数} + \text{多打条数} + \text{错位条数}$$
  $$\sum (\text{FP} + \text{FN}) = \text{漏打笔数}(1) + \text{多打笔数}(1) + \text{错位笔数}(2)$$
  通过上述数学等式严密闭环，差额全在错位上。

### 4.3 判定阈值可复算验证 (`scripts/scan_threshold_replay.py`)
- 验证集由 `:8110` 打分一次，固化分数表；
- 在 9 个候选阈值（$0.30 \sim 0.70$，步长 $0.05$）下套用同一张表分别计算 Micro-F1；
- 选出的最优阈值必须与 `data/ch10/threshold.json` 运行时阈值完全一致，证明阈值是系统扫描得出而非主观臆断。

---

## 5. ONNX 导出与独立轻量推理服务 (:8110)

### 5.1 ONNX 导出与一致性卡点 (`scripts/export_onnx.py`)
- 使用 `torch.onnx.export` 导出为 `data/ch10/onnx/model.onnx`，配置动态轴（`dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}, "logits": {0: "batch_size"}}`）。
- **一致性卡点（Hard Gate）**：拿一批样本对比 PyTorch 与 ONNX Runtime 输出：
  - Logits 绝对误差 $|logits_{\text{torch}} - logits_{\text{onnx}}| < 1 \times 10^{-4}$；
  - 套用阈值后的标签输出 **100% 完全一致**，方视为导出成功。
- 输出配套：保存 `tokenizer.json`, `vocab.txt` 及 `data/ch10/threshold.json`。

### 5.2 独立推理服务 (Port :8110, `app/services/classifier_service/`)
- **轻运行时（No Torch / No Transformers）**：仅依赖 `FastAPI` + `uvicorn` + `onnxruntime` + `tokenizers`，内存占用几十 MB，启动秒级。
- **PID 管理**：PID 写入 `data/ch10/classifier.pid`。
  - Makefile 提供 `make classifier-up` 与 `make classifier-down`；
  - 提供 Windows 适配脚本 `scripts/classifier_service.ps1 -Action start|stop|status`。
- **对外接口规范**：
  - `GET /healthz`：探活接口，返回 `{"status": "ok", "model_loaded": true}`。
  - `POST /classify`：
    - 入参：`{"texts": ["买大了想退", "..."]}`
    - 出参：
      ```json
      {
        "results": [
          {
            "text": "买大了想退",
            "labels": ["尺码", "退换货"],
            "scores": {"退换货": 0.94, "尺码": 0.91, "物流": 0.02, ...}
          }
        ]
      }
      ```
    - 判定阈值严格采用 `data/ch10/threshold.json`。

---

## 6. 旁路批量归类流水线 (`scripts/classify_pool.py`)

- **主链路零干扰**：实时对话主链路绝不调用 `:8110`。
- **未归类数据捞取**：
  `SELECT q.id, q.raw_question FROM low_confidence_questions q LEFT JOIN topic_classifications t ON q.id = t.question_id WHERE t.id IS NULL`
- **攒够一批再归类（Batch Gate）**：
  - 默认 `--min-batch 10`：未归类记录不足 10 条时不执行并友好提示；
  - 允许 `--force` 强制立即执行。
- **批量推理与幂等落库**：
  - 分批调用 `http://127.0.0.1:8110/classify`；
  - 批量插入 `topic_classifications` 表；
  - 基于 `question_id` 唯一约束实现幂等。
- **触发入口**：`make classify-pool`（支持手动触发与定时任务）。

---

## 7. 数据库 DDL 与 ORM 实体

### 7.1 DDL 规范 (`sql/ch10-ddl.sql`)
```sql
CREATE TABLE topic_classifications (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  question_id   BIGINT UNSIGNED NOT NULL                COMMENT '归类的问题,指向 low_confidence_questions.id',
  labels        JSON            NOT NULL                COMMENT '多标签,17 类权威类目里命中的若干个',
  classified_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '归类时间',
  PRIMARY KEY (id),
  UNIQUE KEY uk_question_id (question_id),
  CONSTRAINT fk_topic_question FOREIGN KEY (question_id) REFERENCES low_confidence_questions (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='主题分类结果';
```

### 7.2 ORM 映射 (`app/models/topic_classification.py`)
- 定义 `TopicClassification` 模型，与 `LowConfidenceQuestion` 建立 1 对 1 关系。
- 编写 `scripts/init_ch10_db.py`，支持 MySQL 与 SQLite 内存库双模幂等初始化。

---

## 8. 后台看板与九项实证验收前端体系

### 8.1 页面路由与关注点拆分 (共 5 页)

| 路由 | 页面名称 | 核心关注点与展示内容 |
| :--- | :--- | :--- |
| `/topic-distribution` | **主题分布** | 17 类目问题量统计图表（横向柱状/占比）、堆积 Top 3 类目高亮、一键触发批量归类及自动刷新 |
| `/acceptance` | **总览与闸门** | 九项实证各一张闸门卡片、顶部总闸 N/9、分类器在线状态（探 :8110）、每张卡配备就地重跑按钮 |
| `/acceptance/eval` | **评测深度剖析** | Micro vs Macro 对照、每类 P/R/F1/support/红线达成、阈值扫描 9 候选线、17 类混淆矩阵、单句试分类演示（17 类独立过线） |
| `/acceptance/data` | **语料与数据血缘** | 语料血缘四段、文件盘点、三份考卷分布与**数据泄漏自检（硬闸）**、训练三件套状态、ONNX 产物状态 |
| `/acceptance/errors` | **错例与错因账本** | 逐条标准/预测标签对照、漏打/多打/错位三向记账、近邻边界摩擦配对复核 |

- **后台聚合首页 (`/admin`)**：新增一张卡片，显示「实证过闸数 N/9 + 分类器在线状态」，与知识库录入、飞轮待审、观测成本并列。
- **统一外壳与导航**：共享 `app/static/admin.js`，各页头部统一渲染后台主导航与子页 Tab；纯 HTML/CSS 渲染图表与矩阵，不引入庞大外部前端库。

### 8.2 验收 API 设计 (`app/api/acceptance.py`, 前缀 `/api/acceptance`)
- `GET /api/acceptance/overview`：返回九项闸门状态（`pass` / `fail` / `missing`）、总闸通过数、分类器状态。
- `GET /api/acceptance/eval`：读取 `reports/ch10_evaluation_report.json`，返回分档指标、混淆矩阵、9 候选线扫描数据。
- `GET /api/acceptance/data`：返回语料盘点、三份考卷分布、零泄漏自检结果、训练与 ONNX 产物元数据。
- `GET /api/acceptance/errors`：读取 `data/ch10/reports/error_samples.md`，返回结构化错例列表与三向统计。
- `GET /api/acceptance/service`：现场探活 `http://127.0.0.1:8110/healthz`。
- `POST /api/acceptance/classify`：代理 `:8110/classify`，供前端输入单句实时演示「17 类独立过线，过几个打几个」。
- **容错规则**：任何产物缺失时不返回 500，统一返回 `{"present": false, "make_target": "make xxx"}`，前端卡片呈现「未运行」及重跑按钮，坚决避免页面白屏。

### 8.3 白名单作业运行器 (`app/core/jobs.py` & `app/api/jobs.py`, 前缀 `/api/jobs`)
- **零 Shell 注入安全**：所有 `make <target>` 的命令行参数全部在后端模块内硬编码白名单注册，前端只传作业名称（`job_name`），无法传入任何参数、命令片段或路径。
- **命令一致性**：页面按钮与终端敲击执行完全相同的 Makefile 配方，严禁逻辑分叉。
- **进程生命周期与日志**：
  - 日志统一落入 `log/acceptance/<job>.log`，支持前端高频轮询 tail 回显；
  - 启动进程时配置独立会话（Windows 进程树 / Unix `start_new_session=True`），停止时递归终止整条进程链，严禁遗留孤儿进程；
  - 同一作业未结束时拒绝并发重入；分钟级繁重任务（流水线、训练、导出）标记 `heavy=True`，前端触发前弹窗二次确认。

---

## 9. Makefile 目标与调度配方规范

```makefile
.PHONY: classifier-up classifier-down classify-pool eval-ch10 export-onnx train-ch10 data-prep-ch10 replay-threshold

classifier-up:
	python scripts/classifier_service.py start

classifier-down:
	python scripts/classifier_service.py stop

train-ch10:
	python scripts/train_classifier.py

export-onnx:
	python scripts/export_onnx.py

eval-ch10:
	python scripts/evaluate_classifier.py

replay-threshold:
	python scripts/scan_threshold_replay.py

classify-pool:
	python scripts/classify_pool.py --min-batch 10

data-prep-ch10:
	python scripts/data_prep_ch10.py
```

---

## 10. 验收标准与交付清单

1. **九项实证全绿或明确报告**：
   - 17 类分档 F1 红线达标（严 $\ge 0.9$，中 $\ge 0.8$，宽 `—`）；
   - 三份考卷交叉重叠严格为 0（零泄漏硬闸通过）；
   - ONNX 与 PyTorch 输出在容差内，预测标签 100% 一致；
   - 判定阈值复算脚本复现与 `threshold.json` 一致的选线；
   - 错例分析报告包含漏打、多打、错位三向记账与数学闭环。
2. **旁路批量归类真实生效**：
   - `:8110` 推理服务正常运行，`scripts/classify_pool.py` 成功归类低置信度问题并持久化写入 `topic_classifications`；
   - 实时对话主链路绝不调用分类器。
3. **多诉求判定验证**：
   - 「买大了想退」等测试句能同时命中 `["尺码", "退换货"]`。
4. **前端 5 页面与后台整合**：
   - `/topic-distribution`, `/acceptance`, `/acceptance/eval`, `/acceptance/data`, `/acceptance/errors` 均可顺畅访问、交互与重跑。
5. **过程留痕**：
   - `dev-notes/ch10.md` 完整追踪 4 要素（原话、产出、纠偏、返工）。
