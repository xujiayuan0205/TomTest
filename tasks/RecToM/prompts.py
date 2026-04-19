"""RecToM prompts."""
from typing import Any, Dict

RECTOM_MULTI_LABEL_SYSTEM = (
    "You are an expert at understanding recommendation dialogues and multiple-choice reasoning. "
    "Read the transcript and question carefully. Some questions have multiple correct options. "
    "Output the answer JSON where `answer` is a list of option labels (e.g., {\"answer\": [\"A\"]} "
    "or {\"answer\": [\"A\", \"B\"]}). Include all correct labels and no incorrect labels."
)


def build_prompt(row: Dict[str, Any], method: str = "multi_label_mcq") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (multi_label_mcq)

    Returns:
        格式化的 prompt
    """
    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    transcript = (story_info.get("full_story", "") or "").strip()
    question = str(row.get("Question", "") or "").strip()

    user = (
        f"# Transcript\n{transcript}\n\n"
        f"# Question\n{question}"
    )
    return f"{RECTOM_MULTI_LABEL_SYSTEM}\n\n{user}"
