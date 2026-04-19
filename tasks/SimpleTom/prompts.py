"""SimpleToM prompts."""
from typing import Any, Dict

SIMPLETOM_MCQA_SYSTEM = (
    "You are an expert at understanding human behavior and theory of mind. "
    "Use the transcript and options to choose the single best answer. "
    "Output the answer JSON with exactly one uppercase letter (A, B, or C) in the `answer` field."
)


def build_prompt(row: Dict[str, Any], method: str = "v2_generate") -> str:
    """构建 prompt

    Args:
        row: 数据行，包含 _mcq 字段
        method: 方法名 (v2_generate)

    Returns:
        格式化的 prompt
    """
    mcq = row["_mcq"]
    story_block = mcq["story"].strip()
    question = mcq["question"].strip()
    options = mcq["original_choices"]
    options_block = "\n".join(
        f"[{letter}] {options[letter]}"
        for letter in sorted(options.keys())
    )

    user = (
        f"# Transcript\n{story_block}\n\n"
        f"# Question\n{question}\n\n"
        f"# Options\n{options_block}"
    )
    return f"{SIMPLETOM_MCQA_SYSTEM}\n\n{user}"
