#!/usr/bin/env python3
"""Baseline 模式：开 thinking，跑全量样本。"""

from __future__ import annotations

import sys

from mode_runner import build_parser, run_mode


def main() -> int:
    parser = build_parser("baseline_think_full")
    args = parser.parse_args()
    return run_mode(
        mode_name="baseline_think_full",
        base_config_path=args.config,
        enable_thinking=True,
        max_samples=0,
        repeats=args.repeats,
        manage_vllm=args.manage_vllm,
        vllm_start_command=args.vllm_start_command,
        vllm_stop_command=args.vllm_stop_command,
        vllm_models_url=args.vllm_models_url,
        vllm_start_timeout_sec=args.vllm_start_timeout_sec,
        vllm_stop_after_run=not args.no_vllm_stop_after_run,
    )


if __name__ == "__main__":
    sys.exit(main())
