#!/usr/bin/env python3
"""提取推理块模式：三组配置连续运行并输出 3xN 对比表。"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _bootstrap_run_all_source(run_all_path: Path) -> str:
    """在子进程最先执行：先配置 logging，再跑 run_all；避免 structure_client 的 INFO 刷屏。"""
    p = str(run_all_path.resolve())
    return f"""import logging, runpy
_r = logging.getLogger()
for _h in _r.handlers[:]:
    _r.removeHandler(_h)
try:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
except TypeError:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
for _n in ("httpx", "httpcore", "openai"):
    logging.getLogger(_n).setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
runpy.run_path({p!r}, run_name="__main__")
"""


def _is_run_all_command(cmd: Any) -> bool:
    if not isinstance(cmd, (list, tuple)) or len(cmd) < 2:
        return False
    return str(cmd[1]).endswith("run_all.py")


def _install_quiet_run_all_subprocess() -> None:
    """拦截 mode_runner 对 run_all.py 的 subprocess.run，压低 StructureClient 等 INFO，保留 retry/warning/error。"""

    _orig = subprocess.run

    def _wrapped(*args: Any, **kwargs: Any):
        cmd = args[0] if args else kwargs.get("args")
        if not _is_run_all_command(cmd):
            if args:
                return _orig(*args, **kwargs)
            return _orig(**kwargs)

        cwd = kwargs.get("cwd")
        if cwd is None:
            cwd = Path.cwd()
        else:
            cwd = Path(cwd)
        run_all = cwd / "run_all.py"
        if not run_all.is_file():
            run_all = Path(cmd[1]).resolve()
        bootstrap = _bootstrap_run_all_source(run_all)
        new_cmd = [sys.executable, "-c", bootstrap]
        if args:
            return _orig(new_cmd, *args[1:], **kwargs)
        kwargs = dict(kwargs)
        kwargs["args"] = new_cmd
        return _orig(**kwargs)

    subprocess.run = _wrapped  # type: ignore[assignment]


_install_quiet_run_all_subprocess()

from mode_runner import build_parser, run_mode


def _latest_dataset_metrics(results_path: Path, model_name: str) -> Dict[str, Dict[str, Any]]:
    """读取每个数据集该模型的最新 metrics。"""
    metrics: Dict[str, Dict[str, Any]] = {}
    if not results_path.exists():
        return metrics

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        model_dir = dataset_dir / model_name
        if not model_dir.is_dir():
            continue
        exp_dirs = sorted([p for p in model_dir.glob("exp_*") if p.is_dir()], key=lambda p: p.name)
        if not exp_dirs:
            continue
        latest_exp = exp_dirs[-1]
        metrics_path = latest_exp / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, encoding="utf-8") as f:
            payload = json.load(f)
        metrics[dataset_dir.name] = payload.get("avg_metrics", {})
    return metrics


def _format_acc_cell(metric: Dict[str, Any]) -> str:
    if not metric:
        return "-"
    acc = metric.get("accuracy")
    correct = metric.get("correct")
    total = metric.get("total")
    if isinstance(acc, (int, float)):
        if isinstance(correct, (int, float)) and isinstance(total, (int, float)):
            return f"{acc:.4f} ({int(correct)}/{int(total)})"
        return f"{acc:.4f}"
    return "-"


def _write_comparison_table(
    output_path: Path,
    run_rows: List[Tuple[str, Dict[str, Dict[str, Any]]]],
) -> None:
    all_datasets = sorted({ds for _, m in run_rows for ds in m.keys()})
    lines = [
        "# Reasoning Extract 3xN 对比",
        "",
        "| 运行模式 \\ 数据集 | " + " | ".join(all_datasets) + " |",
        "|" + "|".join(["---"] * (len(all_datasets) + 1)) + "|",
    ]
    for run_name, metrics in run_rows:
        row = [run_name]
        for ds in all_datasets:
            row.append(_format_acc_cell(metrics.get(ds, {})))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = build_parser("reasoning_extract_think_few")
    parser.add_argument(
        "--shared-vllm",
        action="store_true",
        help="三段评测共用外部 vLLM（不在每段之间强制启停）。默认关闭：每段都会尝试启停 vLLM 以串行切换模型。",
    )
    parser.add_argument(
        "--few-samples",
        type=int,
        default=1000,
        help="少样本模式每个数据集抽样数量（默认 1000）",
    )
    parser.add_argument(
        "--few-repeats",
        type=int,
        default=1,
        help="少样本模式默认 repeats=1（可改）",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help="基线模型名；不传则读取 YAML 的 baseline_model / reasoning_extract.baseline_model，否则默认 Qwen3-8B",
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help="目标模型名；不传则读取 YAML 的 target_model / reasoning_extract.target_model，否则默认使用 llm.model_name",
    )
    args = parser.parse_args()

    # vLLM 子进程继承环境变量；降低 APIServer 周期性 INFO（仍可能有关键 WARNING/ERROR）
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    with open(args.config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    reasoning_cfg = (
        base_cfg.get("reasoning_extract", {})
        if isinstance(base_cfg.get("reasoning_extract"), dict)
        else {}
    )

    target_model = (
        args.target_model
        or reasoning_cfg.get("target_model")
        or base_cfg.get("target_model")
        or (base_cfg.get("llm", {}) or {}).get("model_name")
    )
    if not target_model or not isinstance(target_model, str):
        raise SystemExit(
            "未能推断目标模型名：请在 YAML 设置 llm.model_name，或添加 target_model / reasoning_extract.target_model，"
            "或使用命令行 --target-model"
        )

    baseline_model = (
        args.baseline_model
        or reasoning_cfg.get("baseline_model")
        or base_cfg.get("baseline_model")
        or "Qwen3-8B"
    )
    if not isinstance(baseline_model, str) or not baseline_model:
        raise SystemExit("baseline_model 无效：请检查 YAML 或命令行 --baseline-model")

    results_path = Path(base_cfg.get("results_path", "results"))
    if not results_path.is_absolute():
        results_path = Path(__file__).resolve().parent.parent / results_path

    repeats = args.repeats if args.repeats is not None else args.few_repeats
    plans = [
        ("baseline_qwen3_8b_think_on", baseline_model, True),
        ("target_no_think", target_model, False),
        ("target_think_on", target_model, True),
    ]
    collected_rows: List[Tuple[str, Dict[str, Dict[str, Any]]]] = []

    for mode_name, model_name, thinking in plans:
        # 默认：每段评测都串行管理 vLLM（开启 -> 跑 run_all -> 关闭），避免不同权重/模型名复用同一端口时互相污染。
        # 如需三段共用同一个外部 vLLM 实例：传 --shared-vllm（并自行保证模型切换策略正确）。
        if args.shared_vllm:
            manage_vllm = args.manage_vllm
            stop_after_run = not args.no_vllm_stop_after_run
        else:
            if args.manage_vllm is False:
                raise SystemExit("已启用每段串行 vLLM 管理：请不要使用 --no-manage-vllm；如需外部常驻服务请改用 --shared-vllm")
            manage_vllm = True
            stop_after_run = not args.no_vllm_stop_after_run
            if not stop_after_run:
                raise SystemExit("串行 vLLM 模式下不建议使用 --no-vllm-stop-after-run（会导致下一段切换模型时端口仍被占用）")

        code = run_mode(
            mode_name=mode_name,
            base_config_path=args.config,
            enable_thinking=thinking,
            max_samples=args.few_samples,
            repeats=repeats,
            model_name=model_name,
            manage_vllm=manage_vllm,
            vllm_start_command=args.vllm_start_command,
            vllm_stop_command=args.vllm_stop_command,
            vllm_models_url=args.vllm_models_url,
            vllm_start_timeout_sec=args.vllm_start_timeout_sec,
            vllm_stop_after_run=stop_after_run,
        )
        if code != 0:
            print(f"[{mode_name}] 运行失败，返回码: {code}")
            return code
        metrics = _latest_dataset_metrics(results_path, model_name)
        tag = f"{model_name} | think={'on' if thinking else 'off'}"
        collected_rows.append((tag, metrics))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = Path(__file__).resolve().parent.parent / "tables" / f"reasoning_3xN_compare_{ts}.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    _write_comparison_table(table_path, collected_rows)
    print(f"\n3xN 对比表已生成: {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
