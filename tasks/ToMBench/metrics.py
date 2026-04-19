"""ToMBench 数据集的 metrics 计算"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 ToMBench 的 metrics

    包括整体准确率和按 Meta.ability 分组的二级准确率

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

    # 二级指标：按 Meta.ability 分组
    ability_metrics = {}
    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        # 获取 ability
        ability = "unknown"
        if row.get("Meta") and isinstance(row["Meta"], dict):
            ability_value = row["Meta"].get("ability")
            if ability_value:
                ability = ability_value

        if ability not in ability_metrics:
            ability_metrics[ability] = {"correct": 0, "total": 0}
        ability_metrics[ability]["total"] += 1
        if is_correct:
            ability_metrics[ability]["correct"] += 1

    accuracy = correct / total if total else 0

    # 计算各 ability 准确率
    secondary_metrics = {
        f"by_ability.{ability}": stats["correct"] / stats["total"]
        for ability, stats in ability_metrics.items()
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        **secondary_metrics,
        "by_ability": {ability: stats["correct"] / stats["total"] for ability, stats in ability_metrics.items()},
        "ability_counts": {ability: stats["total"] for ability, stats in ability_metrics.items()},
        "per_sample_results": per_sample_results,
    }
