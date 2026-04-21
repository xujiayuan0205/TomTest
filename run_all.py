"""统一评测入口

运行所有数据集的评测脚本。
所有配置通过 experiment_config.yaml 管理，无需命令行参数。
"""
import subprocess
import sys
import os
from pathlib import Path


# 数据集列表
DATASETS = [
    
    "Tomato",
    "ToMBench",
    
    # "ToMi", 指标异常低
    "ToMQA",
    # "ToMChallenges",
    # "Belief_R",
    # "FictionalQA",
    # # "FollowBench", 指标异常低
    # "HellaSwag",
    # # "IFEval", 报错
    # "RecToM",
    # "SimpleTom",
    # "SocialIQA",
]


def run_dataset(dataset: str) -> bool:
    """运行指定数据集的评测

    Args:
        dataset: 数据集名称

    Returns:
        是否成功
    """
    project_root = Path(__file__).resolve().parent
    run_script = project_root / "tasks" / dataset / "run.py"
    if not run_script.exists():
        print(f"[{dataset}] run.py not found, skipping.")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {dataset}")
    print(f"{'='*60}")

    try:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{project_root}:{existing_pythonpath}"
            if existing_pythonpath
            else str(project_root)
        )
        result = subprocess.run(
            [sys.executable, str(run_script)],
            check=True,
            capture_output=False,
            cwd=project_root,
            env=env,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{dataset}] Error: {e}")
        print(f"Return code: {e.returncode}")
        return False
    except Exception as e:
        print(f"[{dataset}] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    for dataset in DATASETS:
        run_dataset(dataset)

    print(f"\n{'='*60}")
    print("All datasets completed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()