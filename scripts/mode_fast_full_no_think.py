#!/usr/bin/env python3
"""快速模式：关 thinking，跑全量样本。"""

from __future__ import annotations

import argparse
import sys

from mode_runner import build_parser, run_mode


def main() -> int:
    parser = build_parser("fast_full_no_think")
    parser.add_argument(
        "--fast-repeats",
        type=int,
        default=1,
        help="快速模式默认 repeats=1（可改）",
    )
    args = parser.parse_args()

    repeats = args.repeats if args.repeats is not None else args.fast_repeats
    return run_mode(
        mode_name="fast_full_no_think",
        base_config_path=args.config,
        enable_thinking=False,
        max_samples=0,
        repeats=repeats,
        manage_vllm=args.manage_vllm,
        vllm_start_command=args.vllm_start_command,
        vllm_stop_command=args.vllm_stop_command,
        vllm_models_url=args.vllm_models_url,
        vllm_start_timeout_sec=args.vllm_start_timeout_sec,
        vllm_stop_after_run=not args.no_vllm_stop_after_run,
    )


if __name__ == "__main__":
    sys.exit(main())
