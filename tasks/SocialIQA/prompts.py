"""SOCIALIQA prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": (
        "You are good at commonsense reasoning about social interactions.\n"
        "Read the story and question, then choose the best option.\n"
        "Output the answer JSON with exactly one uppercase letter (A, B, or C)."
    ),
}


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 SOCIALIQA 的选择题 prompt.

    Args:
        row: 数据行，包含 _mcq 字段
        method: 方法名 (zero_shot)

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["zero_shot"])
    mcq = row["_mcq"]
    story = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["choices"]

    option_lines = [f"({letter}) {options[letter]}" for letter in sorted(options.keys())]
    option_block = "\n".join(option_lines)

    return (
        f"{template}\n\n"
        f"Story: {story}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_block}"
    )
