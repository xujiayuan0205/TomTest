"""FictionalQA prompts"""
from typing import Any, Dict, List, Tuple
import hashlib
import random

PROMPTS = {
    "zero_shot": """You are a Theory of Mind expert.

Story: {story}

Question: {question}

Options:
{options}

Output the answer JSON with exactly one uppercase letter (A, B, C, or D) in the `answer` field.""",
}


def _stable_shuffle(items: List[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    items_copy = list(items)
    rng.shuffle(items_copy)
    return items_copy


def _get_ids(row: Dict[str, Any]) -> Tuple[str, str, str]:
    meta = row.get("Meta", {}) if isinstance(row.get("Meta"), dict) else {}
    id_str = meta.get("id", "") or ""

    event_id = "unknown"
    doc_id = "unknown"
    style = meta.get("fiction_type", "") or "unknown"

    if "_style_" in id_str:
        event_id = id_str.split("_style_")[0]
        style_part = id_str.split("_style_")[1]
        style = style or style_part.split("_")[0]
    if "_question_" in id_str:
        doc_id = id_str.split("_question_")[0]
    elif id_str:
        doc_id = id_str

    return event_id, doc_id, style


def _build_options(row: Dict[str, Any]) -> Tuple[List[str], str]:
    """构建选项列表与标准答案字母（A/B/C/D）"""
    answer_block = row.get("Answer", {}) if isinstance(row.get("Answer"), dict) else {}
    correct_list = answer_block.get("Correct_Answer", [])
    wrong_list = answer_block.get("Wrong_Answer", [])

    correct = correct_list[0] if isinstance(correct_list, list) and correct_list else ""
    wrongs: List[str] = []
    if isinstance(wrong_list, list):
        wrongs = [str(w) for w in wrong_list]

    # 组成候选并补齐到 4 个
    options = [str(correct)] + wrongs
    if len(options) < 4:
        options.extend([""] * (4 - len(options)))
    options = options[:4]

    # 稳定洗牌，避免正确答案总在同一位置
    event_id, doc_id, style = _get_ids(row)
    seed_src = f"{event_id}|{doc_id}|{style}"
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest(), 16)
    shuffled = _stable_shuffle(options, seed)

    gold_index = shuffled.index(correct) if correct in shuffled else 0
    gold_letter = "ABCD"[gold_index]

    return shuffled, gold_letter


def get_gold_label(row: Dict[str, Any]) -> str:
    """获取该样本的标准答案字母"""
    _, gold = _build_options(row)
    return gold


def build_prompt(row: Dict[str, Any], method: str = "zero_shot") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (zero_shot)

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["zero_shot"])

    story_info = row.get("Story", {}) if isinstance(row.get("Story"), dict) else {}
    story = story_info.get("full_story", "") or ""
    question = row.get("Question", "") or ""

    options, _ = _build_options(row)
    options_text = "\n".join(
        [
            f"A. {options[0]}",
            f"B. {options[1]}",
            f"C. {options[2]}",
            f"D. {options[3]}",
        ]
    )

    return template.format(
        story=story.strip(),
        question=question.strip(),
        options=options_text,
    )
