"""IFEval prompts"""
from typing import Any, Dict


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (zero_shot)

    Returns:
        格式化的 prompt（直接返回 Question 字段）
    """
    return row["Question"]