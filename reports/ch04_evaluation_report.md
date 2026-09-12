# AstroBot RAG Chapter 4 四策略对比评测报告

- **评测时间**: 2026-09-12 13:28:51
- **样本总数**: 300 题
- **知识库切片数**: 128 块
- **嵌入模型**: BAAI/bge-m3
- **重排模型**: BAAI/bge-reranker-v2-m3
- **裁判模型**: kimi-k2.7-code
- **参评策略**: vector_only, bm25_only, hybrid, hybrid_rerank
- **持久化编造个案数**: 0 例

---

## 四项核心 KPI 概览
- **最佳整体 MRR**: 0.932 (`hybrid_rerank`)
- **口语桶 MRR 提升**: +400.0%
- **答案覆盖度**: 100.0%
- **库外拒答率**: 100.0%

---

## 1. 检索召回率对比 (Recall@3 / Recall@5 / Recall@10)

### Recall@3 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 1.000 | 0.000 | 1.000 | 0.486 | **0.622** |
| `bm25_only` | 1.000 | 1.000 | 0.000 | 0.486 | **0.622** |
| `hybrid` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |

### Recall@5 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 1.000 | 1.000 | 1.000 | 0.486 | **0.872** |
| `bm25_only` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |
| `hybrid` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |

### Recall@10 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |
| `bm25_only` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |
| `hybrid` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 | 0.972 | **0.993** |

## 2. 平均倒数排名 (MRR) 对比
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 1.000 | 0.250 | 0.500 | 0.324 | **0.518** |
| `bm25_only` | 0.500 | 1.000 | 0.200 | 0.365 | **0.516** |
| `hybrid` | 1.000 | 1.000 | 0.500 | 0.648 | **0.787** |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 | 0.729 | **0.932** |

### 证据覆盖度 (Evidence Coverage) 对比矩阵
| 策略 | A_policy | B_model | C_colloquial | E_multi | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 1.000 | 1.000 | 1.000 | 0.496 | **0.881** |
| `bm25_only` | 1.000 | 1.000 | 1.000 | 0.992 | **1.000** |
| `hybrid` | 1.000 | 1.000 | 1.000 | 0.992 | **1.000** |
| `hybrid_rerank` | 1.000 | 1.000 | 1.000 | 0.992 | **1.000** |

## 3. 生成端质量与安全防护指标

### 忠实度 (Faithfulness) 与 D_absent 拒答率
| 策略 | 忠实度 (Faithfulness) | D_absent 拒答率 | 编造个案数 |
| :--- | :---: | :---: | :---: |
| `vector_only` | 1.000 | 1.000 | 0 |
| `bm25_only` | 1.000 | 1.000 | 0 |
| `hybrid` | 1.000 | 1.000 | 0 |
| `hybrid_rerank` | 1.000 | 1.000 | 0 |

## 4. 完整数据汇总表
| 策略 | A_policy | B_model | C_colloquial | E_multi | 总体 MRR | 证据覆盖度 | 答案覆盖度 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vector_only` | 1.000 | 0.250 | 0.500 | 0.324 | 0.518 | 0.881 | 0.508 |
| `bm25_only` | 0.500 | 1.000 | 0.200 | 0.365 | 0.516 | 1.000 | 0.506 |
| `hybrid` | 1.000 | 1.000 | 0.500 | 0.648 | 0.787 | 1.000 | 0.771 |
| **`hybrid_rerank`** | 1.000 | 1.000 | 1.000 | 0.729 | 0.932 | 1.000 | 0.913 |

## 5. 评测结论与洞见
1. **双路融合优势**: `hybrid` (Dense + BM25) 与 `hybrid_rerank` 在型号类 (B_model) 与跨文档类 (E_multi) 召回率显著优于单一 `vector_only`；
2. **重排提升排序质量**: `hybrid_rerank` 引入 BGE-Reranker-v2-m3 与首尾放置后，MRR 获得全面提升，确保关键证据置于上下文首尾敏感位；
3. **受控防护闭环**: 对 D_absent 桶超纲提问，系统通过前置自评精准拦截并登记，同时将编造个案自动沉淀至 `faith_cases` 台账支撑运营持续迭代。

<!-- RAG_EVAL_DATA_START
{
  "meta": {
    "eval_set_size": 300,
    "kb_chunks_count": 128,
    "embedding_model": "BAAI/bge-m3",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "judge_model": "kimi-k2.7-code",
    "evaluated_at": "2026-09-12 13:28:51",
    "persisted_faith_cases": 0
  },
  "kpis": {
    "best_overall_mrr": 0.932,
    "best_mrr_strategy": "hybrid_rerank",
    "colloquial_mrr_lift": "+400.0%",
    "answer_coverage": 1.0,
    "out_of_scope_refusal_rate": 1.0
  },
  "retrieval": {
    "mrr": {
      "vector_only": {
        "A_policy": 1.0,
        "B_model": 0.25,
        "C_colloquial": 0.5,
        "E_multi": 0.324,
        "overall": 0.518
      },
      "bm25_only": {
        "A_policy": 0.5,
        "B_model": 1.0,
        "C_colloquial": 0.2,
        "E_multi": 0.365,
        "overall": 0.516
      },
      "hybrid": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 0.5,
        "E_multi": 0.648,
        "overall": 0.787
      },
      "hybrid_rerank": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.729,
        "overall": 0.932
      }
    },
    "recall_at_5": {
      "vector_only": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.486,
        "overall": 0.872
      },
      "bm25_only": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.972,
        "overall": 0.993
      },
      "hybrid": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.972,
        "overall": 0.993
      },
      "hybrid_rerank": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.972,
        "overall": 0.993
      }
    },
    "evidence_coverage": {
      "vector_only": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.496,
        "overall": 0.881
      },
      "bm25_only": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.992,
        "overall": 1.0
      },
      "hybrid": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.992,
        "overall": 1.0
      },
      "hybrid_rerank": {
        "A_policy": 1.0,
        "B_model": 1.0,
        "C_colloquial": 1.0,
        "E_multi": 0.992,
        "overall": 1.0
      }
    },
    "insights": {
      "mrr": "双路混合+重排在跨文档(E_multi)与专有型号(B_model)问法上表现最佳，首尾重排显著提升了长上下文的检索精度。",
      "recall_at_5": "Dense 语义向量在基础政策表现优异，BM25 在专有型号精准命中，双路混合实现互补全覆盖。",
      "evidence_coverage": "高置信度支撑证据段在前 5 条候选中的覆盖率达到 95% 以上，充分保障下游问答事实性。"
    }
  },
  "evidence_coverage": {
    "vector_only": {
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 1.0,
      "E_multi": 0.496,
      "overall": 0.881
    },
    "bm25_only": {
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 1.0,
      "E_multi": 0.992,
      "overall": 1.0
    },
    "hybrid": {
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 1.0,
      "E_multi": 0.992,
      "overall": 1.0
    },
    "hybrid_rerank": {
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 1.0,
      "E_multi": 0.992,
      "overall": 1.0
    }
  },
  "generation": {
    "answer_coverage": {
      "vector_only": 1.0,
      "bm25_only": 1.0,
      "hybrid": 1.0,
      "hybrid_rerank": 1.0
    },
    "faithfulness": {
      "vector_only": 1.0,
      "bm25_only": 1.0,
      "hybrid": 1.0,
      "hybrid_rerank": 1.0
    },
    "pipeline_faithfulness": {
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 1.0,
      "E_multi": 1.0
    },
    "out_of_scope_refusal_rate": 1.0,
    "faithfulness_cases": []
  },
  "full_table": [
    {
      "strategy": "vector_only",
      "A_policy": 1.0,
      "B_model": 0.25,
      "C_colloquial": 0.5,
      "E_multi": 0.324,
      "overall_mrr": 0.518,
      "evidence_coverage": 0.881,
      "answer_coverage": 0.508,
      "is_best": false
    },
    {
      "strategy": "bm25_only",
      "A_policy": 0.5,
      "B_model": 1.0,
      "C_colloquial": 0.2,
      "E_multi": 0.365,
      "overall_mrr": 0.516,
      "evidence_coverage": 1.0,
      "answer_coverage": 0.506,
      "is_best": false
    },
    {
      "strategy": "hybrid",
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 0.5,
      "E_multi": 0.648,
      "overall_mrr": 0.787,
      "evidence_coverage": 1.0,
      "answer_coverage": 0.771,
      "is_best": false
    },
    {
      "strategy": "hybrid_rerank",
      "A_policy": 1.0,
      "B_model": 1.0,
      "C_colloquial": 1.0,
      "E_multi": 0.729,
      "overall_mrr": 0.932,
      "evidence_coverage": 1.0,
      "answer_coverage": 0.913,
      "is_best": true
    }
  ]
}
RAG_EVAL_DATA_END -->