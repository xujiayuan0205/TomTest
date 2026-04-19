"""HellaSwag prompts"""

from __future__ import annotations

from typing import Any, Dict, List

PROMPTS = {
    "zero-shot": (
        "You are given a short description of a situation. Your task is to choose the most plausible next event.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}\n\n"
        "A. {ending_0}\n"
        "B. {ending_1}\n"
        "C. {ending_2}\n"
        "D. {ending_3}\n\n"
        "Output the answer JSON with exactly one uppercase letter (A, B, C, or D) in the `answer` field."
    )
}


def build_prompt(row: Dict[str, Any], method: str = "zero-shot") -> str:
    """构建 prompt

    Args:
        row: 数据行，包含 _mcq 字段
        method: 方法名 (zero-shot)

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["zero-shot"])
    mcq = row.get("_mcq") or {}
    context = str(mcq.get("context", "")).strip()
    question = str(mcq.get("question") or row.get("Question", "")).strip()
    endings: List[str] = mcq.get("endings", []) or []
    endings = [str(x).strip() for x in endings]
    if len(endings) != 4:
        raise ValueError(f"HellaSwag expects 4 endings, got {len(endings)}")

    return template.format(
        context=context,
        question=question,
        ending_0=endings[0],
        ending_1=endings[1],
        ending_2=endings[2],
        ending_3=endings[3],
    )
