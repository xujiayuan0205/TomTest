"""通用工具函数"""

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.client import LLMClient


def compute_sample_metrics(
    predictions: List[Any],
    gold_answers: List[Any],
    is_correct_fn: Optional[Callable[[Any, Any], bool]] = None,
) -> Dict[str, Any]:
    """计算每条样本的指标，返回整体统计和每条样本的详细结果

    Args:
        predictions: 模型预测答案列表（None 表示 content_none 错误）
        gold_answers: 金标准答案列表
        is_correct_fn: 自定义正确性判断函数，默认为 pred == gold

    Returns:
        包含 correct, total 和 per_sample_results 的字典
    """
    if is_correct_fn is None:
        is_correct_fn = lambda pred, gold: pred == gold

    correct = 0
    total = len(predictions)
    per_sample_results: List[Dict[str, Any]] = []

    for pred, gold in zip(predictions, gold_answers):
        # 检查 predictions 是否为 None（即 content_none 错误）
        if pred is None:
            is_correct = False
            error_reason = "content_none"
        elif is_correct_fn(pred, gold):
            is_correct = True
            error_reason = None
            correct += 1
        else:
            is_correct = False
            error_reason = "wrong_answer"

        per_sample_results.append({
            "is_correct": is_correct,
            "error_reason": error_reason,
        })

    return {
        "correct": correct,
        "total": total,
        "per_sample_results": per_sample_results,
    }


def compute_sample_metrics_with_llm(
    predictions: List[Any],
    gold_answers: List[Any],
    judge_client: Any,
) -> Dict[str, Any]:
    """使用 LLM judge 计算每条样本的指标

    Args:
        predictions: 模型预测答案列表（None 表示 content_none 错误）
        gold_answers: 金标准答案列表
        judge_client: Judge LLM 客户端

    Returns:
        包含 correct, total 和 per_sample_results 的字典
    """
    from src.schemas import JudgeAnswer

    # 构建 judge prompts
    judge_prompts = []
    for pred, gold in zip(predictions, gold_answers):
        pred_str = str(pred) if pred is not None else "(no answer)"
        gold_str = str(gold) if gold is not None else "(no answer)"
        prompt = (
            f"Compare the following model prediction with the gold standard answer.\n"
            f"Model Prediction: {pred_str}\n"
            f"Gold Standard Answer: {gold_str}\n\n"
            f"Determine if the prediction is correct. Respond a JSON with exactly 'True' if correct, 'False' otherwise."
        )
        judge_prompts.append(prompt)

    # 批量调用 LLM judge
    judge_results = judge_client.batch_generate_structure(judge_prompts, JudgeAnswer)

    # 解析结果
    correct = 0
    total = len(predictions)
    per_sample_results: List[Dict[str, Any]] = []

    for pred, judge_result in zip(predictions, judge_results):
        if pred is None:
            is_correct = False
            error_reason = "content_none"
        elif judge_result.content and judge_result.content.answer == "True":
            is_correct = True
            error_reason = None
            correct += 1
        else:
            is_correct = False
            error_reason = "wrong_answer"

        per_sample_results.append({
            "is_correct": is_correct,
            "error_reason": error_reason,
        })

    return {
        "correct": correct,
        "total": total,
        "per_sample_results": per_sample_results,
    }
