"""ToMQA prompts"""
from typing import Any, Dict

PROMPTS = {
    "zero_shot": (
        "Read the story and answer the question.\n"
        "Output the answer JSON with a single word or short phrase.\n\n"
        "Story: {story}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
}


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["zero_shot"])
    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    story = story_info.get("full_story", "") or ""

    question = row.get("Question", "") or ""

    return template.format(
        story=story,
        question=question,
    )
