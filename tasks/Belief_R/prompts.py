"""Belief-R prompts"""
from typing import Any, Dict, List, Tuple

JSON_OUTPUT_REMINDER = (
    "Output the answer JSON with exactly one lowercase letter (a, b, or c)."
)

PROMPTS = {
    "ZS_vanilla": "",
    "ZS_CoT": "Let's think step by step.",
    "ZS_PS": (
        "Let's first understand the problem and devise a plan to solve the problem.\n"
        "Then, let's carry out the plan and solve the problem step by step."
    ),
    "ZS_RaR": "Rephrase and expand the question, and respond.",
}


def _build_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
    """构建选项列表与标准答案字母。

    约定：
    - time_t+1 需要更新信念 => 正确答案放在 c
    - 其他情况 => 正确答案放在 a
    """
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct_list = answer_block.get("Correct_Answer", [])
    wrong_list = answer_block.get("Wrong_Answer", [])

    correct = correct_list[0] if isinstance(correct_list, list) and correct_list else ""
    wrongs: List[str] = []
    if isinstance(wrong_list, list):
        wrongs = [str(w) for w in wrong_list]
    if len(wrongs) < 2:
        wrongs.extend([""] * (2 - len(wrongs)))

    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    step = (meta.get("step", "") or "").lower()
    is_update = step in {"time_t+1", "time_t1", "t+1", "time_t_1"}

    if is_update:
        options = [wrongs[0], wrongs[1], correct]
        gold = "c"
    else:
        options = [correct, wrongs[0], wrongs[1]]
        gold = "a"

    return options, gold


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取该样本的标准答案字母"""
    _, gold = _build_options(row)
    return gold


def build_prompt(row: Dict[str, Any], method: str = "ZS_vanilla") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (ZS_vanilla/ZS_CoT/ZS_PS/ZS_RaR)

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["ZS_vanilla"])

    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    story = story_info.get("full_story", "") or ""
    question = row.get("Question", "") or ""

    options, _ = _build_options(row)
    options_text = "\n".join(
        [
            f"a. {options[0]}",
            f"b. {options[1]}",
            f"c. {options[2]}",
        ]
    )

    parts = [
        story.strip(),
        question.strip(),
        "Options:\n" + options_text,
        JSON_OUTPUT_REMINDER,
    ]
    if template:
        parts.append(template)

    return "\n\n".join(p for p in parts if p)
