"""Tomato 数据集的 metrics：整体准确率 + Meta.dimension 三维分层。"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _extract_dimension_slots(row: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = row.get("Meta") or {}
    if not isinstance(meta, dict):
        return "__missing__", "__missing__", "__none__"
    dims = meta.get("dimension", [])
    if not isinstance(dims, list):
        dims = [dims]
    d0 = str(dims[0]) if len(dims) >= 1 and dims[0] not in (None, "") else "__missing__"
    d1 = str(dims[1]) if len(dims) >= 2 and dims[1] not in (None, "") else "__missing__"
    d2 = str(dims[2]) if len(dims) >= 3 and dims[2] not in (None, "") else "__none__"
    return d0, d1, d2


def compute_metrics(
    predictions: List[str],
    gold_letters: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """整体准确率 + 按 Meta.dimension 三维分层的准确率与样本数（对齐 ToMBench：扁平二级指标 + counts）。

    Args:
        predictions: 模型预测答案列表（None 表示 content_none 错误）
        gold_letters: 金标准答案列表（A/B/C/D）
        data: 原始数据列表
        judge_client: 可选的 Judge LLM 客户端，如果提供则使用 LLM judge

    Returns:
        包含基础指标、维度指标和 per_sample_results 的字典
    """
    assert len(predictions) == len(data) == len(gold_letters), "predictions/data/gold 长度须一致"

    # 自定义比较函数
    def is_correct_fn(pred: Any, gold: Any) -> bool:
        return bool(pred and pred == gold)

    # 使用通用函数计算基础指标和每条样本结果
    if judge_client is not None:
        sample_metrics = compute_sample_metrics_with_llm(predictions, gold_letters, judge_client)
    else:
        sample_metrics = compute_sample_metrics(predictions, gold_letters, is_correct_fn)
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    # 维度统计
    dim_total: List[Dict[str, int]] = [defaultdict(int) for _ in range(3)]
    dim_correct: List[Dict[str, int]] = [defaultdict(int) for _ in range(3)]

    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        d0, d1, d2 = _extract_dimension_slots(row)
        for j, d in enumerate((d0, d1, d2)):
            dim_total[j][d] += 1
            if is_correct:
                dim_correct[j][d] += 1

    accuracy = correct / total if total else 0.0

    def _acc_dict(tot: Dict[str, int], cor: Dict[str, int]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in sorted(tot.keys()):
            out[k] = (cor.get(k, 0) / tot[k]) if tot[k] else 0.0
        return out

    dim_acc = [_acc_dict(dim_total[i], dim_correct[i]) for i in range(3)]

    secondary_metrics: Dict[str, float] = {}
    for i in range(3):
        for k, v in dim_acc[i].items():
            secondary_metrics[f"by_dimension_{i+1}.{k}"] = v

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "dimension_1_counts": {k: dim_total[0][k] for k in sorted(dim_total[0].keys())},
        "dimension_2_counts": {k: dim_total[1][k] for k in sorted(dim_total[1].keys())},
        "dimension_3_counts": {k: dim_total[2][k] for k in sorted(dim_total[2].keys())},
        "per_sample_results": per_sample_results,
    }
