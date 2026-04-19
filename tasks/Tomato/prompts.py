"""Tomato prompts：Transcript/Question/Options 用户块。"""
from typing import Any, Dict

PROMPTS = {
    "v2_generate": (
        "# Transcript\n{story}\n\n"
        "# Question\n{question}\n\n"
        "# Options\n{options}"
    ),
}


def build_prompt(row: Dict[str, Any], method: str = "v2_generate") -> str:
    """拼接用户块（不对正文做 str.format，避免 transcript 中花括号干扰）。

    Args:
        row: 数据行
        method: 方法名

    Returns:
        格式化的 prompt（用户内容）
    """
    mcq = row["_mcq"]
    story_block = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["original_choices"]
    lines_o = [f"[{letter}] {options[letter]}" for letter in sorted(options.keys())]
    options_block = "\n".join(lines_o)

    return (
        f"# Transcript\n{story_block}\n\n"
        f"# Question\n{question}\n\n"
        f"# Options\n{options_block}\n\n"
        f"Output the answer JSON with exactly one uppercase letter (A, B, C, or D)."
    )
