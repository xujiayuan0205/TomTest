"""ToMBench prompts"""
from typing import Any, Dict, List

# 1. Vanilla Prompt for Chinese Evaluation
VANILLA_PROMPT_CN = """
请你提供一段故事，一个问题和若干答案选项。
请你根据故事内容和给定的问题，按照常理推测，选择一个最可能的答案选项。
输出答案的json只包含A、B、C、D四个选项中的一个。

【故事】
{story}

【问题】
{question}
"""

# 3. Vanilla Prompt for English Evaluation
VANILLA_PROMPT_EN = """
Below is a multiple-choice question with a story and several answer options. 
Based on the content of the story and the given question, please infer the most likely answer.
Output the answer JSON with exactly one letter (A/B/C/D).

[Story]
{story}

[Question]
{question}
"""

def build_prompt(row: Dict[str, Any], method: str) -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: 方法名 (VANILLA/COT)

    Returns:
        格式化的 prompt
    """
    story = row["Story"]
    question = row["Question"]
    lang = row['Meta']['lang']

    if lang == 'zh':
        if method == 'VANILLA':
            return VANILLA_PROMPT_CN.format(story=story, question=question)
        else:
            raise ValueError(f"Unsupported method={method}")
    elif lang == 'en':
        if method == 'VANILLA':
            return VANILLA_PROMPT_EN.format(story=story, question=question)
        else:
            raise ValueError(f"Unsupported method={method}")
    raise ValueError(f"Unsupported lang={lang}")


