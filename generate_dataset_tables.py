"""
Dataset Tables Generator

从 results/ 目录读取 metrics.json 文件，为每个数据集生成详细表格：
1. 基础指标.md：accuracy, correct, total
2. 其他指标.md：其他所有指标（包括字典类型的子指标）
3. 复制 config.json 到输出目录
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set


def collect_metrics(results_dir: str, exp_suffix: str = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """收集所有数据集和模型的 metrics

    Args:
        results_dir: results 目录路径
        exp_suffix: 实验时间后缀，用于筛选特定实验结果。如果为 None，自动选择每个数据集最新的实验

    Returns:
        {dataset_name: {model_name: metrics_data}}
    """
    results_path = Path(results_dir)
    metrics_data = {}

    for dataset_dir in results_path.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
            continue

        dataset_name = dataset_dir.name
        metrics_data[dataset_name] = {}

        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue

            model_name = model_dir.name

            # 查找实验目录
            if exp_suffix:
                # 指定后缀
                exp_dir = model_dir / f"exp_{exp_suffix}"
                if not exp_dir.exists():
                    continue
            else:
                # 自动选择最新的实验目录
                exp_dirs = sorted(model_dir.glob("exp_*"))
                if not exp_dirs:
                    continue
                exp_dir = exp_dirs[-1]  # 最新的

            metrics_file = exp_dir / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics_data[dataset_name][model_name] = json.load(f)

    return metrics_data


def get_all_metrics_names(metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    """获取所有出现的指标名称（标量类型）

    Args:
        metrics_data: 收集的 metrics 数据

    Returns:
        所有标量指标名称的列表
    """
    metric_names = set()

    for dataset_metrics in metrics_data.values():
        for model_metrics in dataset_metrics.values():
            if "avg_metrics" in model_metrics:
                for key, value in model_metrics["avg_metrics"].items():
                    if not isinstance(value, dict):
                        metric_names.add(key)

    return sorted(metric_names)


def get_dict_metrics(metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    """获取所有字典类型的指标名称

    Args:
        metrics_data: 收集的 metrics 数据

    Returns:
        所有字典类型指标名称的列表
    """
    dict_metrics = set()

    for dataset_metrics in metrics_data.values():
        for model_metrics in dataset_metrics.values():
            if "avg_metrics" in model_metrics:
                for key, value in model_metrics["avg_metrics"].items():
                    if isinstance(value, dict):
                        dict_metrics.add(key)

    return sorted(dict_metrics)


def format_value(value: Any) -> str:
    """格式化指标值

    Args:
        value: 指标值

    Returns:
        格式化后的字符串
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    elif isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    else:
        return str(value)


def generate_basic_metrics_table(dataset_name: str, models: List[str], metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """生成基础指标表格：accuracy, correct, total

    Args:
        dataset_name: 数据集名称
        models: 模型列表
        metrics_data: 收集的 metrics 数据

    Returns:
        Markdown 表格字符串
    """
    # 构建 Markdown 表格
    header = "| 指标 \\ 模型 | " + " | ".join(models) + " |"
    separator = "|" + "|".join(["---"] * (len(models) + 1)) + "|"

    lines = [
        f"# {dataset_name} - 基础指标",
        "",
        header,
        separator,
    ]

    basic_metrics = ["accuracy", "correct", "total"]
    for metric in basic_metrics:
        row = [metric]
        for model in models:
            if model in metrics_data[dataset_name]:
                value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(metric, "")
                row.append(format_value(value))
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_other_metrics_table(dataset_name: str, models: List[str], metrics_data: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """生成其他指标表格

    Args:
        dataset_name: 数据集名称
        models: 模型列表
        metrics_data: 收集的 metrics 数据

    Returns:
        Markdown 表格字符串
    """
    metric_names = get_all_metrics_names({dataset_name: metrics_data[dataset_name]})
    basic_metrics = ["accuracy", "correct", "total"]
    other_metrics = [m for m in metric_names if m not in basic_metrics]

    lines = [
        f"# {dataset_name} - 其他指标",
        "",
    ]

    if other_metrics:
        lines.extend([
            "## 标量指标",
            "",
            "| 指标 \\ 模型 | " + " | ".join(models) + " |",
            "|" + "|".join(["---"] * (len(models) + 1)) + "|",
        ])

        for metric in other_metrics:
            row = [metric]
            for model in models:
                if model in metrics_data[dataset_name]:
                    value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(metric, "")
                    row.append(format_value(value))
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    # 处理字典类型的指标
    dict_metrics = get_dict_metrics({dataset_name: metrics_data[dataset_name]})

    for dict_metric in dict_metrics:
        lines.append(f"## {dict_metric}")
        lines.append("")

        # 获取所有子键
        all_sub_keys = set()
        for model in models:
            if model in metrics_data[dataset_name]:
                dict_value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(dict_metric, {})
                if isinstance(dict_value, dict):
                    all_sub_keys.update(dict_value.keys())

        if all_sub_keys:
            sub_keys = sorted(all_sub_keys)
            lines.append("| 子指标 \\ 模型 | " + " | ".join(models) + " |")
            lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")

            for sub_key in sub_keys:
                row = [sub_key]
                for model in models:
                    if model in metrics_data[dataset_name]:
                        dict_value = metrics_data[dataset_name][model].get("avg_metrics", {}).get(dict_metric, {})
                        if isinstance(dict_value, dict) and sub_key in dict_value:
                            row.append(format_value(dict_value[sub_key]))
                        else:
                            row.append("-")
                    else:
                        row.append("-")
                lines.append("| " + " | ".join(row) + " |")

            lines.append("")

    if not other_metrics and not dict_metrics:
        lines.append("该数据集没有其他指标。\n")

    return "\n".join(lines)


def generate_dataset_tables(results_dir: str = "results", output_dir: str = "tables", exp_suffix: str = None) -> None:
    """生成所有数据集的指标表格

    Args:
        results_dir: results 目录路径
        output_dir: 输出目录路径
        exp_suffix: 实验时间后缀，用于筛选特定实验结果（必填）
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)

    metrics_data = collect_metrics(results_dir, exp_suffix)

    if not metrics_data:
        print("没有找到任何评测结果。")
        return

    for dataset_name in sorted(metrics_data.keys()):
        models = sorted(metrics_data[dataset_name].keys())

        dataset_report_dir = output_path / dataset_name
        dataset_report_dir.mkdir(parents=True, exist_ok=True)

        basic_metrics = generate_basic_metrics_table(dataset_name, models, metrics_data)
        basic_path = dataset_report_dir / "基础指标.md"
        basic_path.write_text(basic_metrics, encoding='utf-8')

        other_metrics = generate_other_metrics_table(dataset_name, models, metrics_data)
        other_path = dataset_report_dir / "其他指标.md"
        other_path.write_text(other_metrics, encoding='utf-8')

        for model_name in models:
            model_dir = results_path / dataset_name / model_name

            # 确定 exp_dir，与 collect_metrics 逻辑一致
            if exp_suffix:
                exp_dir = model_dir / f"exp_{exp_suffix}"
            else:
                exp_dirs = sorted(model_dir.glob("exp_*"))
                exp_dir = exp_dirs[-1] if exp_dirs else None

            if exp_dir:
                config_file = exp_dir / "config.json"
            else:
                config_file = model_dir / "config.json"

            if config_file.exists():
                output_model_dir = dataset_report_dir / model_name
                output_model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, output_model_dir / "config.json")

        print(f"数据集 [{dataset_name}] 报告已保存到: {dataset_report_dir}/")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="从 results/ 目录生成各数据集的指标表格",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="results 目录路径（默认: results）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tables",
        help="输出目录路径（默认: tables）"
    )
    parser.add_argument(
        "--exp-suffix",
        type=str,
        default=None,
        help="实验时间后缀（可选），如: 20240101_120000。不传则自动选择每个数据集最新的实验"
    )

    args = parser.parse_args()

    generate_dataset_tables(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        exp_suffix=args.exp_suffix,
    )


if __name__ == "__main__":
    main()