"""FictionalQA 数据集的 metrics 计算"""

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import compute_sample_metrics, compute_sample_metrics_with_llm


def _stable_shuffle(items: List[str], seed: int) -> List[str]:
    import random

    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)
    return items_copy


def _get_ids(row: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    id_str = meta.get("id", "") or ""

    event_id = "unknown"
    doc_id = "unknown"
    style = meta.get("fiction_type", "") or "unknown"

    if "_style_" in id_str:
        event_id = id_str.split("_style_")[0]
        style_part = id_str.split("_style_")[1]
        style = style or style_part.split("_")[0]
    if "_question_" in id_str:
        doc_id = id_str.split("_question_")[0]
    elif id_str:
        doc_id = id_str

    return event_id, doc_id, style


def _build_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
    """构建选项列表与标准答案字母（A/B/C/D）"""
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct_list = answer_block.get("Correct_Answer", [])
    wrong_list = answer_block.get("Wrong_Answer", [])

    correct = correct_list[0] if isinstance(correct_list, list) and correct_list else ""
    wrongs: List[str] = []
    if isinstance(wrong_list, list):
        wrongs = [str(w) for w in wrong_list]

    options = [str(correct)] + wrongs
    if len(options) < 4:
        options.extend([""] * (4 - len(options)))
    options = options[:4]

    event_id, doc_id, style = _get_ids(row)
    seed_src = f"{event_id}|{doc_id}|{style}"
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest(), 16)
    shuffled = _stable_shuffle(options, seed)

    gold_index = shuffled.index(correct) if correct in shuffled else 0
    gold_letter = "ABCD"[gold_index]

    return shuffled, gold_letter


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
    """计算 FictionalQA 的 metrics

    - Overall Accuracy
    - Informed vs Blind Gap
    - Split-based Evaluation: Event / Document / Style

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

    # 二级指标：按 event/document/style 分组
    by_event: Dict[str, Dict[str, int]] = {}
    by_doc: Dict[str, Dict[str, int]] = {}
    by_style: Dict[str, Dict[str, int]] = {}

    blind_values: List[float] = []

    for is_correct, row in zip([r["is_correct"] for r in per_sample_results], data):
        event_id, doc_id, style = _get_ids(row)

        by_event.setdefault(event_id, {"correct": 0, "total": 0})
        by_event[event_id]["total"] += 1
        if is_correct:
            by_event[event_id]["correct"] += 1

        by_doc.setdefault(doc_id, {"correct": 0, "total": 0})
        by_doc[doc_id]["total"] += 1
        if is_correct:
            by_doc[doc_id]["correct"] += 1

        by_style.setdefault(style, {"correct": 0, "total": 0})
        by_style[style]["total"] += 1
        if is_correct:
            by_style[style]["correct"] += 1

        blind_val = None
        if isinstance(row.get("blind_grade_avg"), (int, float)):
            blind_val = float(row.get("blind_grade_avg"))
        else:
            meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
            if isinstance(meta.get("blind_grade_avg"), (int, float)):
                blind_val = float(meta.get("blind_grade_avg"))
        if blind_val is not None:
            blind_values.append(blind_val)

    accuracy = correct / total if total else 0

    def _avg_group(group: Dict[str, Dict[str, int]]) -> float:
        if not group:
            return 0
        vals = []
        for stats in group.values():
            if stats["total"]:
                vals.append(stats["correct"] / stats["total"])
        return sum(vals) / len(vals) if vals else 0

    event_split_acc = _avg_group(by_event)
    doc_split_acc = _avg_group(by_doc)
    style_split_acc = _avg_group(by_style)

    blind_avg = sum(blind_values) / len(blind_values) if blind_values else None
    informed_vs_blind_gap = (accuracy - blind_avg) if blind_avg is not None else None

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "informed_vs_blind_gap": informed_vs_blind_gap,
        "blind_avg": blind_avg,
        "event_split_acc": event_split_acc,
        "document_split_acc": doc_split_acc,
        "style_split_acc": style_split_acc,
        "event_split_details": {
            k: (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in by_event.items()
        },
        "document_split_details": {
            k: (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in by_doc.items()
        },
        "style_split_details": {
            k: (v["correct"] / v["total"] if v["total"] else 0)
            for k, v in by_style.items()
        },
        "event_counts": {k: v["total"] for k, v in by_event.items()},
        "document_counts": {k: v["total"] for k, v in by_doc.items()},
        "style_counts": {k: v["total"] for k, v in by_style.items()},
        "per_sample_results": per_sample_results,
    }
