# 新增数据集指南

本指南介绍如何在 TomTest 框架中添加一个新的数据集评测。

## 概述

添加新数据集需要创建 4 个文件：
1. `config.yaml` - 数据集配置
2. `prompts.py` - Prompt 模板
3. `metrics.py` - Metrics 计算（包括二级指标）
4. `run.py` - 评测脚本

然后更新顶层 `run_all.py` 注册数据集。

## 步骤 1：创建数据集目录

```bash
cd /path/to/TomTest/tasks
mkdir MyDataset
cd MyDataset
```

## 步骤 2：创建 config.yaml

`config.yaml` 定义数据集的配置：

```yaml
dataset: MyDataset
path: MyDataset/test

# 默认 prompt 方法
method: ZS_vanilla

# 结构化输出的 schema 类名（从 src.schemas 导入）
schema: MCQAnswer

# 系统 prompt（可选，会覆盖 experiment_config.yaml 中的设置）
system_prompt: ""
```

### Schema 选择

在 `config.yaml` 中指定 schema 名称，框架会从 `src/schemas.py` 自动加载。

| Schema | 使用场景 | 字段 |
|---|---|---|
| `MCQAnswer` | 四选一（A/B/C/D） | `answer: Literal["A", "B", "C", "D"]` |
| `MCQAnswer3` | 三选一（A/B/C） | `answer: Literal["A", "B", "C"]` |
| `MCQAnswer3Lower` | 三选一小写（a/b/c） | `answer: Literal["a", "b", "c"]` |
| `OpenAnswer` | 开放式问答 | `answer: str` |
| `OneWordAnswer` | 单词回答 | `answer: str`（无空白） |
| `JudgeAnswer` | LLM Judge 判断 | `answer: Literal["True", "False"]` |
| `MultiLabelAnswer` | 多标签多选 | `answer: List[str]` |

如需自定义 schema，在 `src/schemas.py` 中添加新的 Pydantic 模型，然后在 `config.yaml` 中引用。

## 步骤 3：创建 prompts.py

`prompts.py` 定义 prompt 模板和 `build_prompt` 函数。

```python
"""MyDataset prompts"""
from typing import Any, Dict

PROMPTS = {
    "ZS_vanilla": """You are an expert.

Context: {context}
Question: {question}

Options:
{options}

Answer with exactly one letter (A/B/C/D):""",

    "ZS_cot": """You are an expert.

Context: {context}
Question: {question}

Options:
{options}

Let's think step by step...
Answer with exactly one letter (A/B/C/D):""",
}


def build_prompt(row: Dict[str, Any], method: str = "ZS_vanilla") -> str:
    """构建 prompt

    Args:
        row: 数据行
        method: prompt 方法名称

    Returns:
        格式化的 prompt
    """
    template = PROMPTS.get(method, PROMPTS["ZS_vanilla"])

    # 从数据行提取字段
    context = row.get("context", "")
    question = row.get("question", "")
    options = row.get("options", [])

    # 构建选项文本
    options_text = "\n".join([
        f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)
    ])

    return template.format(context=context, question=question, options=options_text)
```

## 步骤 4：创建 metrics.py

`metrics.py` 定义指标计算函数。

### 模板 1：直接匹配（MCQAnswer/OpenAnswer）

```python
"""MyDataset metrics 计算"""
from typing import Any, Dict, List
from src.utils import compute_sample_metrics


def compute_metrics(predictions: List[str], data: List[Dict[str, Any]], judge_client=None) -> Dict[str, Any]:
    """计算 MyDataset 的 metrics（直接匹配）

    Args:
        predictions: 模型预测答案列表（从结构化输出提取的 answer 字段）
        data: 原始数据列表
        judge_client: LLM judge 客户端（可选）

    Returns:
        包含基础指标和二级指标的字典
    """
    gold_answers = [row.get("answer", "") for row in data]

    # 使用通用函数计算样本级指标
    sample_result = compute_sample_metrics(
        predictions=predictions,
        gold_answers=gold_answers,
        is_correct_fn=lambda p, g: p == g,
    )

    # 二级指标：按类别分组（可选）
    category_metrics = {}
    for pred, gold, row, is_correct in zip(predictions, gold_answers, data, sample_result["per_sample_results"]):
        category = row.get("category", "unknown")

        if category not in category_metrics:
            category_metrics[category] = {"correct": 0, "total": 0}
        category_metrics[category]["total"] += 1
        if is_correct:
            category_metrics[category]["correct"] += 1

    # 计算各类别准确率
    secondary_metrics = {
        f"by_category.{cat}": stats["correct"] / stats["total"]
        for cat, stats in category_metrics.items()
    }

    return {
        "accuracy": sample_result["correct"] / sample_result["total"],
        "correct": sample_result["correct"],
        "total": sample_result["total"],
        **secondary_metrics,
        "by_category": {cat: stats["correct"] / stats["total"] for cat, stats in category_metrics.items()},
        "category_counts": {cat: stats["total"] for cat, stats in category_metrics.items()},
        "per_sample_results": sample_result["per_sample_results"],
    }
```

### 模板 2：LLM Judge

```python
"""MyDataset metrics 计算（LLM Judge 版）"""
from typing import Any, Dict, List
from src.schemas import JudgeAnswer
from src.utils import compute_sample_metrics


def compute_metrics(
    predictions: List[str],
    data: List[Dict[str, Any]],
    judge_client,
) -> Dict[str, Any]:
    """计算 MyDataset 的 metrics（使用 LLM Judge）

    Args:
        predictions: 模型预测答案列表
        data: 原始数据列表
        judge_client: LLM judge 客户端

    Returns:
        包含基础指标的字典
    """
    gold_answers = [row.get("gold_answer", "") for row in data]

    # 批量构建 judge prompts
    judge_prompts = []
    for pred, row in zip(predictions, data):
        gold = row.get("gold_answer", "")
        context = row.get("context", "")
        question = row.get("question", "")

        judge_prompt = f"""Determine if the answer is correct.

Context: {context}
Question: {question}
Ground Truth: {gold}
Model Answer: {pred}

Output True or False:"""
        judge_prompts.append(judge_prompt)

    # 批量结构化输出
    judge_results = judge_client.batch_generate_structure(judge_prompts, JudgeAnswer)

    # 统计正确数
    correct = sum(1 for result in judge_results if result.content and result.content.answer == "True")

    return {
        "accuracy": correct / len(predictions) if predictions else 0,
        "correct": correct,
        "total": len(predictions),
        "per_sample_results": [
            {
                "is_correct": result.content and result.content.answer == "True",
                "error_reason": None,
            }
            for result in judge_results
        ],
    }
```

## 步骤 5：创建 run.py

`run.py` 是评测脚本的主入口。

```python
"""MyDataset 评测脚本"""
import sys
from pathlib import Path

# 添加父目录到路径以导入 src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import runner

from MyDataset.prompts import build_prompt
from MyDataset.metrics import compute_metrics


def main():
    # 加载数据集配置
    dataset_config = runner.load_dataset_config("tasks/MyDataset/config.yaml")

    # 加载实验配置
    experiment_config = runner.load_experiment_config("experiment_config.yaml")

    # 从 config 获取 schema 和方法
    schema = runner.load_schema(dataset_config["schema"])
    prompt_method = dataset_config["method"]

    # 创建 LLM 客户端
    client = runner.create_llm_client(experiment_config["llm_config"], dataset_config)

    # 创建 Judge 客户端（如果配置了）
    judge_client = runner.create_judge_client(experiment_config["judge_config"])

    # 加载数据
    data = runner.load_and_limit_data(
        subset=dataset_config["path"],
        datasets_path=experiment_config["datasets_path"],
        max_samples=experiment_config["max_samples"],
    )

    print(f"Loaded {len(data)} samples from {dataset_config['path']}")
    print(f"Prompt method: {prompt_method}")
    print(f"Schema: {dataset_config['schema']}")
    print(f"Repeats: {experiment_config['repeats']}")

    # 构建 prompts
    prompts = [build_prompt(row, prompt_method) for row in data]
    all_prompts = prompts * experiment_config["repeats"]

    # 批量结构化推理
    print(f"Running inference ({len(all_prompts)} prompts)...")
    results = client.batch_generate_structure(all_prompts, schema)

    # 计算指标
    n = len(data)
    all_predictions = []
    all_metrics = []

    for i in range(experiment_config["repeats"]):
        start = i * n
        end = start + n
        repeat_results = results[start:end]
        predictions = [r.content.answer if r.content else None for r in repeat_results]
        all_predictions.append(predictions)

        # 调用数据集的 metrics 函数
        metrics = compute_metrics(predictions, data, judge_client)
        all_metrics.append(metrics)
        print(f"Run {i+1}: Accuracy={metrics['accuracy']:.4f}, Correct={metrics['correct']}/{metrics['total']}")

    # 保存结果
    gold_answers = [row.get("answer", "") for row in data]
    runner.save_common_results(
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        all_results=[[r for r in results[i*n:(i+1)*n]] for i in range(experiment_config["repeats"])],
        all_prompts=[all_prompts[i*n:(i+1)*n] for i in range(experiment_config["repeats"])],
        gold_answers=gold_answers,
        all_metrics=all_metrics,
    )

    # 打印统计摘要
    runner.print_summary_stats(all_metrics, experiment_config["repeats"], len(gold_answers))


if __name__ == "__main__":
    main()
```

## 步骤 6：注册数据集

编辑顶层 `run_all.py`，在 `DATASETS` 列表中添加：

```python
DATASETS = [
    "Belief_R",
    "FictionalQA",
    "MyDataset",  # 新增
    "SocialIQA",
    ...
]
```

## 步骤 7：测试

```bash
# 配置实验参数（编辑 experiment_config.yaml）
# 然后运行
python tasks/MyDataset/run.py

# 或运行所有数据集
python run_all.py
```

## 常见问题

### Q: 如何添加自定义 schema？

编辑 `src/schemas.py`，添加新的 Pydantic 模型：

```python
class CustomAnswer(BaseModel):
    """自定义答案 schema"""
    answer: Literal["OPTION_1", "OPTION_2", "OPTION_3"]
    confidence: float  # 可选字段
```

然后在数据集的 `config.yaml` 中引用：

```yaml
schema: CustomAnswer
```

### Q: 如何处理不同的数据格式？

数据集加载器返回的是数据列表，每个元素是一个字典。根据实际数据格式调整 `build_prompt` 和 `extract_gold_answers` 函数。

### Q: 如何支持多个 prompt 方法？

在 `PROMPTS` 字典中添加新方法，然后在 `config.yaml` 中修改 `method`。

### Q: 如何调试？

使用 `experiment_config.yaml` 中的 `max_samples` 参数限制样本数进行快速测试：

```yaml
max_samples: 3  # 只测试前 3 条样本
```

### Q: 如何使用 LLM Judge？

1. 在 `experiment_config.yaml` 中配置 judge 并设置 `use_llm_judge: true`
2. 在 `run.py` 中调用 `runner.create_judge_client()` 获取 judge 客户端
3. 在 `metrics.py` 中使用 judge 客户端进行判断

### Q: 如何覆盖系统 prompt？

在数据集的 `config.yaml` 中设置 `system_prompt`：

```yaml
system_prompt: "You are a helpful assistant specialized in theory of mind tasks."
```