"""ToMChallenges prompts"""
from typing import Any, Dict

PROMPTS = {
    "ZS_vanilla": (
        "Choose the correct answer from A or B for the following question:\n"
        "Question:\n"
        "{question_block}\n\n"
        "A. {option_a}\n"
        "B. {option_b}\n\n"
        "Output the answer JSON with exactly one uppercase letter (A or B)."
    ),
}


def build_prompt(row: Dict[str, Any], method: str = "ZS_vanilla") -> str:
    """构建 prompt。"""
    template = PROMPTS.get(method, PROMPTS["ZS_vanilla"])

    story = row.get("Story") if isinstance(row.get("Story"), dict) else {}
    full_story = (story.get("full_story") if isinstance(story, dict) else "") or ""
    question = row.get("Question", "") or ""
    question_block = f"{str(full_story).strip()} {str(question).strip()}".strip()

    mcq = row.get("_mcq") or {}
    choices = mcq.get("choices") if isinstance(mcq.get("choices"), dict) else {}
    option_a = choices.get("A", "")
    option_b = choices.get("B", "")

    return template.format(
        question_block=question_block,
        option_a=option_a,
        option_b=option_b,
    )