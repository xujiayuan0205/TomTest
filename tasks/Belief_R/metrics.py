"""Belief-R 数据集的 metrics 计算"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _build_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
    """构建选项列表与标准答案字母。

    约定：
    - time_t+1 需要更新信念 => 正确答案放在 c
    - 其他情况 => 正确答案放在 a
    """
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct_list = answer_block.get("Correct_Answer", [])
    wrong_list = answer_block.get("Wrong_Answer", [])

    correct = correct_list[0] if isinstance(correct_list, list) and correct_list else ""
    wrongs: List[str] = []
    if isinstance(wrong_list, list):
        wrongs = [str(w) for w in wrong_list]
    if len(wrongs) < 2:
        wrongs.extend([""] * (2 - len(wrongs)))

    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    step = (meta.get("step", "") or "").lower()
    is_update = step in {"time_t+1", "time_t1", "t+1", "time_t_1"}

    if is_update:
        options = [wrongs[0], wrongs[1], correct]
        gold = "c"
    else:
        options = [correct, wrongs[0], wrongs[1]]
        gold = "a"

    return options, gold


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取该样本的标准答案字母"""
    _, gold = _build_options(row)
    return gold


def compute_metrics(
    predictions: List[str],
    gold_answers: List[str],
    data: List[Dict[str, Any]],
    judge_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """计算 Belief-R 的 metrics

    - Overall Accuracy
    - BU-Acc: time_t+1 且标准答案为 c
    - BM-Acc: 其余样本（无需更新时保持原结论）
    - BREU: (BU-Acc + BM-Acc) / 2

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

    # 二级指标：按 step 分组计算 BU-Acc 和 BM-Acc
    bu_total = 0
    bu_correct = 0
    bm_total = 0
    bm_correct = 0

    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
        step = (meta.get("step", "") or "").lower()
        gold_label = get_gold_label(row)

        is_bu = (step in {"time_t+1", "time_t1", "t+1", "time_t_1"} and gold_label == "c")
        if is_bu:
            bu_total += 1
            if is_correct:
                bu_correct += 1
        else:
            bm_total += 1
            if is_correct:
                bm_correct += 1

    accuracy = correct / total if total else 0
    bu_acc = bu_correct / bu_total if bu_total else 0
    bm_acc = bm_correct / bm_total if bm_total else 0
    breu = (bu_acc + bm_acc) / 2

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "BU-Acc": bu_acc,
        "BM-Acc": bm_acc,
        "BREU": breu,
        "bu_correct": bu_correct,
        "bu_total": bu_total,
        "bm_correct": bm_correct,
        "bm_total": bm_total,
        "per_sample_results": per_sample_results,
    }
