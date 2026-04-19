"""FollowBench prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": (
        "{instruction}\n\n"
        "Output your answer as a JSON object with an `answer` field containing the text."
    ),
}


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (zero_shot)

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["zero_shot"])
    instruction = row.get("Question", "")
    return template.format(instruction=instruction)
