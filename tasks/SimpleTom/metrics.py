"""SimpleToM metrics."""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _update_group(stats: Dict[str, Dict[str, int]], key: Any, correct: bool) -> None:
    key_str = str(key) if key not in (None, "") else "unknown"
    if key_str not in stats:
        stats[key_str] = {"correct": 0, "total": 0}
    stats[key_str]["total"] += 1
    if correct:
        stats[key_str]["correct"] += 1


def _flatten(group: Dict[str, Dict[str, int]], prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}.{key}": (value["correct"] / value["total"] if value["total"] else 0.0)
        for key, value in sorted(group.items())
    }


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算整体准确率和按来源、维度、难度分组的准确率。

    Args:
        predictions: 模型预测答案列表（None 表示 content_none 错误）
        gold_answers: 金标准答案列表
        data: 原始数据列表
        judge_client: 可选的 Judge LLM 客户端，如果提供则使用 LLM judge

    Returns:
        包含基础指标、二级指标和 per_sample_results 的字典
    """
    # 使用通用函数计算基础指标和每条样本结果
    if judge_client is not None:
        sample_metrics = compute_sample_metrics_with_llm(predictions, gold_answers, judge_client)
    else:
        sample_metrics = compute_sample_metrics(predictions, gold_answers)
    correct = sample_metrics["correct"]
    total = sample_metrics["total"]
    per_sample_results = sample_metrics["per_sample_results"]

    by_source: Dict[str, Dict[str, int]] = {}
    by_dimension: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}

    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        source = meta.get("dataset_source", "unknown")
        difficulty = meta.get("difficulty", "unknown")
        dims = meta.get("dimension", [])
        if isinstance(dims, list) and dims:
            dimension = dims[0]
        else:
            dimension = dims if dims else "unknown"

        _update_group(by_source, source, is_correct)
        _update_group(by_dimension, dimension, is_correct)
        _update_group(by_difficulty, difficulty, is_correct)

    accuracy = correct / total if total else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **_flatten(by_source, "by_source"),
        **_flatten(by_dimension, "by_dimension"),
        **_flatten(by_difficulty, "by_difficulty"),
        "by_source": {
            key: (value["correct"] / value["total"] if value["total"] else 0.0)
            for key, value in sorted(by_source.items())
        },
        "source_counts": {key: value["total"] for key, value in sorted(by_source.items())},
        "by_dimension": {
            key: (value["correct"] / value["total"] if value["total"] else 0.0)
            for key, value in sorted(by_dimension.items())
        },
        "dimension_counts": {key: value["total"] for key, value in sorted(by_dimension.items())},
        "by_difficulty": {
            key: (value["correct"] / value["total"] if value["total"] else 0.0)
            for key, value in sorted(by_difficulty.items())
        },
        "difficulty_counts": {key: value["total"] for key, value in sorted(by_difficulty.items())},
        "per_sample_results": per_sample_results,
    }