"""IFEval metrics 计算

调用 instruction_following_eval 的检查器，计算：
- prompt-level 准确率：所有指令都通过才算正确
- instruction-level 准确率：每条指令单独计算
- 按指令类别（tier0）分组的二级指标
支持 strict 和 loose 两种模式。
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

# 将当前目录加入路径，使用本地复制的 instructions_registry
sys.path.insert(0, str(Path(__file__).parent))

from instructions_registry import INSTRUCTION_DICT


def _check_one(row: Dict[str, Any], response: str, loose: bool) -> List[bool]:
    """对单条样本的每条指令做检查，返回每条指令是否通过的列表。"""
    instruction_id_list = row["Meta"]["dimension"]
    # instruction_kwargs 是 JSON 字符串列表，反序列化回字典
    kwargs_list = [json.loads(kw) for kw in row["Meta"]["instruction_kwargs"]]
    prompt = row["Question"]
    is_following_list = []

    if loose:
        r = response.split("\n")
        candidates = [
            response,
            response.replace("*", ""),
            "\n".join(r[1:]).strip(),
            "\n".join(r[:-1]).strip(),
            "\n".join(r[1:-1]).strip(),
            "\n".join(r[1:]).strip().replace("*", ""),
            "\n".join(r[:-1]).strip().replace("*", ""),
            "\n".join(r[1:-1]).strip().replace("*", ""),
        ]
    else:
        candidates = [response]

    for idx, instruction_id in enumerate(instruction_id_list):
        instruction_cls = INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            is_following_list.append(False)
            continue

        instruction = instruction_cls(instruction_id)
        kwargs = {k: v for k, v in kwargs_list[idx].items() if v is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        if loose:
            followed = any(
                c.strip() and instruction.check_following(c)
                for c in candidates
            )
        else:
            followed = bool(response.strip() and instruction.check_following(response))

        is_following_list.append(followed)

    return is_following_list


def compute_metrics(
    predictions: List[str],
    data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """计算 IFEval 的 strict 和 loose 两套指标。

    Args:
        predictions: 模型生成的回答列表
        data: 原始数据列表（每行含 prompt / instruction_id_list / kwargs）

    Returns:
        包含 prompt-level、instruction-level 及按类别二级指标的字典
    """
    prompt_correct_strict = 0
    prompt_correct_loose = 0
    instr_total = 0
    instr_correct_strict = 0
    instr_correct_loose = 0

    tier0_total: Dict[str, int] = defaultdict(int)
    tier0_correct_strict: Dict[str, int] = defaultdict(int)
    tier0_correct_loose: Dict[str, int] = defaultdict(int)

    for pred, row in zip(predictions, data):
        following_strict = _check_one(row, pred, loose=False)
        following_loose = _check_one(row, pred, loose=True)

        if all(following_strict):
            prompt_correct_strict += 1
        if all(following_loose):
            prompt_correct_loose += 1

        for instruction_id, fs, fl in zip(
            row["Meta"]["dimension"], following_strict, following_loose
        ):
            instr_total += 1
            if fs:
                instr_correct_strict += 1
            if fl:
                instr_correct_loose += 1

            tier0 = instruction_id.split(":")[0]
            tier0_total[tier0] += 1
            if fs:
                tier0_correct_strict[tier0] += 1
            if fl:
                tier0_correct_loose[tier0] += 1

    total = len(predictions)

    prompt_acc_strict = prompt_correct_strict / total if total else 0.0
    prompt_acc_loose = prompt_correct_loose / total if total else 0.0
    instr_acc_strict = instr_correct_strict / instr_total if instr_total else 0.0
    instr_acc_loose = instr_correct_loose / instr_total if instr_total else 0.0

    # 添加 per_sample_results
    per_sample_results = []
    for pred, row in zip(predictions, data):
        following_strict = _check_one(row, pred, loose=False)
        following_loose = _check_one(row, pred, loose=True)
        is_correct_strict = all(following_strict)
        is_correct_loose = all(following_loose)
        per_sample_results.append({
            "is_correct": is_correct_strict,
            "error_reason": None if is_correct_strict else "wrong_answer",
        })

    # 二级指标：按指令类别
    by_category_strict = {
        k: tier0_correct_strict[k] / tier0_total[k]
        for k in sorted(tier0_total)
    }
    by_category_loose = {
        k: tier0_correct_loose[k] / tier0_total[k]
        for k in sorted(tier0_total)
    }

    secondary = {}
    for k in sorted(tier0_total):
        secondary[f"by_category_strict.{k}"] = by_category_strict[k]
        secondary[f"by_category_loose.{k}"] = by_category_loose[k]

    return {
        # 主指标（框架用 accuracy / correct / total 做摘要）
        "accuracy": prompt_acc_strict,
        "correct": prompt_correct_strict,
        "total": total,
        # 完整四项指标
        "prompt_accuracy_strict": prompt_acc_strict,
        "prompt_accuracy_loose": prompt_acc_loose,
        "instruction_accuracy_strict": instr_acc_strict,
        "instruction_accuracy_loose": instr_acc_loose,
        # 二级指标
        **secondary,
        "by_category_strict": by_category_strict,
        "by_category_loose": by_category_loose,
        "category_counts": dict(tier0_total),
        "per_sample_results": per_sample_results,
    }
