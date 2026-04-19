"""ToMi 数据集的 metrics 计算"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _normalize_word(text: Any) -> str:
    """归一化为单词比较格式。"""
    if text is None:
        return ""
    return str(text).strip().lower()


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 ToMi 的 metrics（单词答案精确匹配）

    Args:
        predictions: 模型预测答案列表（None 表示 content_none 错误）
        gold_answers: 金标准答案列表
        data: 原始数据列表（未使用，保留兼容性）
        judge_client: 可选的 Judge LLM 客户端，如果提供则使用 LLM judge

    Returns:
        包含基础指标和 per_sample_results 的字典
    """
    pred_answers = [_normalize_word(p) if p is not None else None for p in predictions]
    normalized_gold = [_normalize_word(g) for g in gold_answers]

    # 使用通用函数计算基础指标和每条样本结果
    if judge_client is not None:
        sample_metrics = compute_sample_metrics_with_llm(pred_answers, normalized_gold, judge_client)
    else:
        sample_metrics = compute_sample_metrics(pred_answers, normalized_gold)
    accuracy = sample_metrics["correct"] / sample_metrics["total"] if sample_metrics["total"] else 0

    return {
        "accuracy": accuracy,
        **sample_metrics,
    }