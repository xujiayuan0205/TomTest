"""HellaSwag 数据集的 metrics：Overall / In-domain / Zero-shot 三类准确率。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


class _SplitStats(TypedDict):
    correct: int
    total: int


def _safe_div(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取标准答案字母"""
    mcq = row.get("_mcq") or {}
    return mcq.get("gold_letter", "")


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 HellaSwag 的 metrics（直接匹配选项字母）。

    - Overall Accuracy: 全量准确率
    - In-domain Accuracy: Meta.split_type == "indomain"
    - Zero-shot Accuracy: Meta.split_type == "zeroshot"

    Args:
        predictions: 模型预测答案列表（None 表示 content_none 错误）
        gold_answers: 金标准答案列表
        data: 原始数据列表
        judge_client: 可选的 Judge LLM 客户端，如果提供则使用 LLM judge

    Returns:
        包含基础指标、二级指标和 per_sample_results 的字典
    """
    if len(predictions) != len(data):
        raise ValueError(f"predictions/data length mismatch: {len(predictions)} vs {len(data)}")

    # 使用通用函数计算基础指标和每条样本结果
    if judge_client is not None:
        sample_metrics = compute_sample_metrics_with_llm(predictions, gold_answers, judge_client)
    else:
        sample_metrics = compute_sample_metrics(predictions, gold_answers)
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    by_split_type: Dict[str, _SplitStats] = {}

    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        meta = row.get("Meta") if isinstance(row.get("Meta"), dict) else {}
        split_type = (meta.get("split_type") if isinstance(meta, dict) else None) or "unknown"
        split_type = str(split_type)

        if split_type not in by_split_type:
            by_split_type[split_type] = {"correct": 0, "total": 0}
        by_split_type[split_type]["total"] += 1
        if is_correct:
            by_split_type[split_type]["correct"] += 1

    accuracy = _safe_div(correct, total)

    by_split_type_acc = {k: _safe_div(v["correct"], v["total"]) for k, v in by_split_type.items()}
    split_type_counts = {k: v["total"] for k, v in by_split_type.items()}
    secondary_metrics = {f"by_split_type.{k}": v for k, v in by_split_type_acc.items()}

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "by_split_type": by_split_type_acc,
        "split_type_counts": split_type_counts,
        "per_sample_results": per_sample_results,
    }
