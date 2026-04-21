#!/usr/bin/env python3
"""运行不同评测模式的公共工具。"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CONFIG_PATH = ROOT / "experiment_config.yaml"
RUN_ALL_PATH = ROOT / "run_all.py"

# 原版 Qwen3-8B 权重路径（用于自动生成 vLLM 启动命令）
QWEN3_8B_ORIGINAL_MODEL_DIR = "/storage/tangxr/home/data/model/Qwen3-8B"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件不是字典结构: {path}")
    return data


def _ensure_dict(parent: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"配置字段 `{key}` 必须是字典，当前是: {type(value).__name__}")
    return value


def _build_mode_config(
    base_config: Dict[str, Any],
    *,
    enable_thinking: bool,
    max_samples: int,
    repeats: int | None,
    model_name: str | None = None,
) -> Dict[str, Any]:
    cfg = dict(base_config)
    llm = _ensure_dict(cfg, "llm")
    llm["enable_thinking"] = bool(enable_thinking)
    if model_name:
        llm["model_name"] = model_name
    cfg["max_samples"] = int(max_samples)
    if repeats is not None:
        cfg["repeats"] = int(repeats)
    return cfg


def _render_command_template(command: str, mode_config: Dict[str, Any]) -> str:
    llm_cfg = mode_config.get("llm", {}) if isinstance(mode_config.get("llm"), dict) else {}
    return command.format(
        model_name=llm_cfg.get("model_name", ""),
        api_url=llm_cfg.get("api_url", ""),
    )


def _rewrite_local_bind_hostname_for_client(parsed) -> str:
    """把 bind 地址（0.0.0.0 / ::）改写成本机回环，便于 urllib/httpx 做健康检查。

    说明：服务端监听 0.0.0.0 表示监听所有网卡，但客户端连接 http://0.0.0.0:port 在 Linux 上常常失败（Connection refused）。
    """
    host = parsed.hostname
    if host in {"0.0.0.0", "::"}:
        new_host = "127.0.0.1"
        if parsed.port:
            new_netloc = f"{new_host}:{parsed.port}"
        else:
            new_netloc = new_host
        parsed = parsed._replace(netloc=new_netloc)
    return urlunparse(parsed)


def _client_models_url_from_api_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    base = urlunparse(parsed)
    base = _rewrite_local_bind_hostname_for_client(urlparse(base))
    return base.rstrip("/") + "/models"


def _default_models_url(mode_config: Dict[str, Any]) -> Optional[str]:
    llm_cfg = mode_config.get("llm", {}) if isinstance(mode_config.get("llm"), dict) else {}
    api_url = llm_cfg.get("api_url")
    if not api_url or not isinstance(api_url, str):
        return None
    return _client_models_url_from_api_url(api_url)


def _api_listen_host_port(mode_config: Dict[str, Any]) -> tuple[str, int] | None:
    """从 llm.api_url 解析 vLLM OpenAI server 监听 host/port（用于默认启动命令）。"""
    llm_cfg = mode_config.get("llm", {}) if isinstance(mode_config.get("llm"), dict) else {}
    api_url = llm_cfg.get("api_url")
    if not api_url or not isinstance(api_url, str):
        return None
    parsed = urlparse(api_url)
    if not parsed.hostname or not parsed.port:
        return None
    return parsed.hostname, int(parsed.port)


def _default_vllm_start_command_for_qwen3_8b_original(mode_config: Dict[str, Any]) -> str | None:
    """当本轮评测模型为 Qwen3-8B（原版）时，生成默认 vLLM 启动命令。"""
    llm_cfg = mode_config.get("llm", {}) if isinstance(mode_config.get("llm"), dict) else {}
    model_name = llm_cfg.get("model_name")
    if model_name != "Qwen3-8B":
        return None

    host_port = _api_listen_host_port(mode_config)
    if not host_port:
        return None
    host, port = host_port

    return (
        f"{sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {QWEN3_8B_ORIGINAL_MODEL_DIR} "
        "--trust-remote-code "
        "--reasoning-parser qwen3 "
        f"--host {host} --port {port} "
        "--served-model-name Qwen3-8B "
        # 降噪：默认关闭 uvicorn access log（否则会刷屏 POST /v1/chat/completions）
        "--disable-uvicorn-access-log "
        "--no-enable-log-requests"
    )


def _default_stop_command(mode_config: Dict[str, Any]) -> Optional[str]:
    models_url = _default_models_url(mode_config)
    if not models_url:
        return None
    parsed = urlparse(models_url)
    if not parsed.port:
        return None
    return f"fuser -k {parsed.port}/tcp || true"


def _wait_for_vllm_ready(
    *,
    models_url: str,
    expected_model_name: str,
    timeout_sec: int,
) -> None:
    deadline = time.time() + timeout_sec
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            with urlopen(models_url, timeout=5) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
            if expected_model_name in payload:
                return
            last_err = f"模型 {expected_model_name} 尚未出现在 {models_url}"
        except Exception as e:  # pragma: no cover
            last_err = str(e)
        time.sleep(2)
    raise TimeoutError(f"等待 vLLM 就绪超时（{timeout_sec}s）: {last_err}")


def _start_vllm(command: str) -> subprocess.Popen[str]:
    print(f"[vllm] 启动命令: {command}")
    # 新会话便于在关闭时对整组发 SIGTERM/SIGKILL；仅 terminate() bash 无法带走 EngineCore 等多进程子树
    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=ROOT,
        start_new_session=True,
    )
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(f"vLLM 启动进程提前退出，exit_code={proc.returncode}")
    return proc


def _stop_vllm(command: str) -> None:
    print(f"[vllm] 关闭命令: {command}")
    subprocess.run(["bash", "-lc", command], cwd=ROOT, check=False)


def _extra_stop_vllm(command: str | None) -> None:
    """可选：在 fuser 之后执行（例如 pkill 残留的 vLLM/EngineCore 子进程）。"""
    if not command or not str(command).strip():
        return
    print(f"[vllm] 额外清理: {command.strip()}")
    subprocess.run(["bash", "-lc", str(command)], cwd=ROOT, check=False)


def _read_proc_cmdline(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()


def _is_probably_vllm_compute_cmdline(cmd: str) -> bool:
    """EngineCore 等子进程 cmdline 往往不含 openai.api_server，但会包含 vllm 包路径或关键字。"""
    if not cmd:
        return False
    lower = cmd.lower()
    if "vllm" in lower:
        return True
    if "openai.api_server" in lower:
        return True
    # site-packages/vllm/... 被截断时可能只有路径片段
    if "/vllm/" in cmd or r"\vllm\\" in cmd:
        return True
    return False


def _nvidia_compute_pids(gpu_index: int) -> list[int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
            "-i",
            str(gpu_index),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        return []
    out: list[int] = []
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        first = line.split(",")[0].strip()
        try:
            out.append(int(first))
        except ValueError:
            continue
    return out


def _cleanup_vllm_processes_on_gpu(gpu_index: int) -> None:
    """根据 nvidia-smi 占用 GPU 的 PID，结束命令行看起来像 vLLM 的进程（补 pkill 漏掉的 EngineCore 等）。"""
    pids = _nvidia_compute_pids(gpu_index)
    if not pids:
        return
    targets: list[int] = []
    for pid in pids:
        cmd = _read_proc_cmdline(pid)
        if cmd and _is_probably_vllm_compute_cmdline(cmd):
            targets.append(pid)
    if not targets:
        return
    print(f"[vllm] 根据 nvidia-smi 结束 GPU:{gpu_index} 上 vLLM 相关 PID: {targets}")
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(2)
    for pid in targets:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            cmd = _read_proc_cmdline(pid)
            if cmd and _is_probably_vllm_compute_cmdline(cmd):
                os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _kill_vllm_launcher_process(proc: subprocess.Popen[str] | None) -> None:
    """结束本脚本通过 _start_vllm 拉起的 bash + 子进程（优先整进程组）。"""
    if proc is None or proc.poll() is not None:
        return
    pid = proc.pid
    pgid: int | None
    try:
        pgid = os.getpgid(pid)
    except OSError:
        pgid = None
    print(f"[vllm] 结束本机启动的 vLLM 进程树 (pid={pid}, pgid={pgid})")
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _gpu_free_mib(gpu_index: int = 0) -> float | None:
    """当前 GPU 空闲显存（MiB）；失败时返回 None。"""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            cwd=ROOT,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        line = proc.stdout.strip().splitlines()[0].strip()
        return float(line)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _wait_for_min_gpu_free_mib(
    *,
    gpu_index: int,
    min_free_mib: float,
    timeout_sec: int,
    poll_sec: float = 2.0,
) -> None:
    """在启动新 vLLM 前轮询，直到空闲显存 >= 阈值或超时。"""
    deadline = time.time() + max(1, int(timeout_sec))
    last: float | None = None
    while time.time() < deadline:
        last = _gpu_free_mib(gpu_index)
        if last is not None and last >= min_free_mib:
            print(
                f"[vllm] GPU:{gpu_index} 空闲显存 {last:.0f} MiB >= 阈值 {min_free_mib:.0f} MiB，继续启动"
            )
            return
        if last is not None:
            print(
                f"[vllm] GPU:{gpu_index} 当前空闲 {last:.0f} MiB，低于阈值 {min_free_mib:.0f} MiB，"
                f"{poll_sec:.0f}s 后重试（最长 {timeout_sec}s）…"
            )
        else:
            print(f"[vllm] 无法读取 nvidia-smi，{poll_sec:.0f}s 后重试…")
        time.sleep(poll_sec)
    raise TimeoutError(
        f"等待 GPU:{gpu_index} 释放显存超时（{timeout_sec}s）："
        f"最后读数={last} MiB，需要 >= {min_free_mib} MiB。"
        "请检查是否仍有 python/vLLM 进程占用 GPU，或增大 vllm.wait_gpu_timeout_sec。"
    )


def _vllm_cooldown_after_stop(vllm_opts: Dict[str, Any]) -> None:
    """stop 之后的固定等待 + 可选按显存阈值轮询。"""
    sleep_sec = int(vllm_opts.get("sleep_sec_after_stop", 10))
    if sleep_sec > 0:
        print(f"[vllm] 等待 {sleep_sec}s 以便释放 GPU 显存（可在 YAML 设置 vllm.sleep_sec_after_stop）")
        time.sleep(sleep_sec)
    min_mib = vllm_opts.get("wait_min_free_gpu_mib")
    if min_mib is None:
        return
    try:
        thr = float(min_mib)
    except (TypeError, ValueError):
        return
    if thr <= 0:
        return
    gpu_index = int(vllm_opts.get("gpu_index", 0))
    timeout_sec = int(vllm_opts.get("wait_gpu_timeout_sec", 180))
    poll_sec = float(vllm_opts.get("wait_gpu_poll_sec", 2.0))
    print(
        f"[vllm] 轮询 GPU:{gpu_index} 空闲显存，需 >= {thr:.0f} MiB（vllm.wait_min_free_gpu_mib）"
    )
    _wait_for_min_gpu_free_mib(
        gpu_index=gpu_index,
        min_free_mib=thr,
        timeout_sec=timeout_sec,
        poll_sec=poll_sec,
    )


def _resolve_vllm_options(
    *,
    base_config: Dict[str, Any],
    mode_config: Dict[str, Any],
    manage_vllm: bool | None,
    vllm_start_command: str | None,
    vllm_stop_command: str | None,
    vllm_models_url: str | None,
    vllm_start_timeout_sec: int,
    vllm_stop_after_run: bool,
) -> Dict[str, Any]:
    vllm_cfg = base_config.get("vllm", {}) if isinstance(base_config.get("vllm"), dict) else {}
    llm_cfg = mode_config.get("llm", {}) if isinstance(mode_config.get("llm"), dict) else {}
    model_name = llm_cfg.get("model_name")

    enabled = False
    if manage_vllm is True:
        enabled = True
    elif manage_vllm is False:
        enabled = False
    else:
        # manage_vllm is None：未显式指定 --manage-vllm / --no-manage-vllm
        enabled = bool(vllm_cfg.get("auto_manage", False))
        if not enabled and model_name == "Qwen3-8B":
            # 默认：需要原版 Qwen3-8B 时自动管理 vLLM（可用 YAML `vllm.disable_auto_manage_original_qwen3_8b: true` 关闭）
            enabled = not bool(vllm_cfg.get("disable_auto_manage_original_qwen3_8b", False))

    # 启动命令优先级：CLI > vllm.start_command > vllm.start_commands[model_name] > 内置原版 Qwen3-8B
    start_cmd = vllm_start_command or vllm_cfg.get("start_command")
    if not start_cmd and model_name:
        by_model = vllm_cfg.get("start_commands")
        if isinstance(by_model, dict) and model_name in by_model:
            val = by_model.get(model_name)
            if isinstance(val, str) and val.strip():
                start_cmd = val
    if enabled and not start_cmd:
        start_cmd = _default_vllm_start_command_for_qwen3_8b_original(mode_config)
    stop_cmd = vllm_stop_command or vllm_cfg.get("stop_command") or _default_stop_command(mode_config)
    models_url = vllm_models_url or vllm_cfg.get("models_url") or _default_models_url(mode_config)
    if isinstance(models_url, str):
        models_url = _rewrite_local_bind_hostname_for_client(urlparse(models_url))
    timeout = int(vllm_cfg.get("start_timeout_sec", vllm_start_timeout_sec))
    stop_after = bool(vllm_cfg.get("stop_after_run", vllm_stop_after_run))
    if isinstance(start_cmd, str):
        start_cmd = _render_command_template(start_cmd, mode_config)
    if isinstance(stop_cmd, str):
        stop_cmd = _render_command_template(stop_cmd, mode_config)
    extra_stop = vllm_cfg.get("extra_stop_command")
    if isinstance(extra_stop, str):
        extra_stop = _render_command_template(extra_stop, mode_config)
    else:
        extra_stop = None
    # 杀掉旧 vLLM 后 CUDA 显存往往不能立刻归还，默认多等几秒再启新进程（串行切换模型时尤其重要）
    sleep_after_stop = int(vllm_cfg.get("sleep_sec_after_stop", 10))
    wait_min_mib = vllm_cfg.get("wait_min_free_gpu_mib")
    gpu_index = int(vllm_cfg.get("gpu_index", 0))
    wait_timeout = int(vllm_cfg.get("wait_gpu_timeout_sec", 180))
    try:
        wait_poll = float(vllm_cfg.get("wait_gpu_poll_sec", 2.0))
    except (TypeError, ValueError):
        wait_poll = 2.0
    sigterm_gpu = bool(vllm_cfg.get("sigterm_gpu_vllm_processes", True))
    return {
        "enabled": enabled,
        "start_command": start_cmd,
        "stop_command": stop_cmd,
        "extra_stop_command": extra_stop,
        "models_url": models_url,
        "timeout_sec": timeout,
        "stop_after_run": stop_after,
        "sleep_sec_after_stop": max(0, sleep_after_stop),
        "wait_min_free_gpu_mib": wait_min_mib,
        "gpu_index": gpu_index,
        "wait_gpu_timeout_sec": max(1, wait_timeout),
        "wait_gpu_poll_sec": max(0.5, wait_poll),
        "sigterm_gpu_vllm_processes": sigterm_gpu,
    }


def run_mode(
    *,
    mode_name: str,
    base_config_path: Path,
    enable_thinking: bool,
    max_samples: int,
    repeats: int | None,
    model_name: str | None = None,
    manage_vllm: bool | None = None,
    vllm_start_command: str | None = None,
    vllm_stop_command: str | None = None,
    vllm_models_url: str | None = None,
    vllm_start_timeout_sec: int = 180,
    vllm_stop_after_run: bool = True,
) -> int:
    if not base_config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {base_config_path}")
    if not RUN_ALL_PATH.exists():
        raise FileNotFoundError(f"未找到入口脚本: {RUN_ALL_PATH}")

    base_config = _load_yaml(base_config_path)
    mode_config = _build_mode_config(
        base_config,
        enable_thinking=enable_thinking,
        max_samples=max_samples,
        repeats=repeats,
        model_name=model_name,
    )
    vllm_opts = _resolve_vllm_options(
        base_config=base_config,
        mode_config=mode_config,
        manage_vllm=manage_vllm,
        vllm_start_command=vllm_start_command,
        vllm_stop_command=vllm_stop_command,
        vllm_models_url=vllm_models_url,
        vllm_start_timeout_sec=vllm_start_timeout_sec,
        vllm_stop_after_run=vllm_stop_after_run,
    )

    temp_config_path = ROOT / f".tmp_experiment_config_{mode_name}.yaml"
    temp_config_path.write_text(
        yaml.safe_dump(mode_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    backup_path = ROOT / ".tmp_experiment_config_backup.yaml"
    had_original = EXPERIMENT_CONFIG_PATH.exists()

    started_proc: Optional[subprocess.Popen[str]] = None
    try:
        if vllm_opts["enabled"]:
            if not vllm_opts["start_command"]:
                mn = mode_config.get("llm", {}).get("model_name", "")
                raise ValueError(
                    f"已启用 vLLM 管理，但未找到模型 `{mn}` 的启动命令。"
                    "原版 Qwen3-8B 可省略（会自动生成）。"
                    "其余模型请在 YAML 中配置 `vllm.start_commands`（按模型名映射到启动命令）、"
                    "或全局 `vllm.start_command`、或命令行 `--vllm-start-command`。"
                )
            if vllm_opts["stop_command"]:
                _stop_vllm(vllm_opts["stop_command"])
            _extra_stop_vllm(vllm_opts.get("extra_stop_command"))
            if vllm_opts.get("sigterm_gpu_vllm_processes", True):
                _cleanup_vllm_processes_on_gpu(int(vllm_opts.get("gpu_index", 0)))
            _vllm_cooldown_after_stop(vllm_opts)
            started_proc = _start_vllm(vllm_opts["start_command"])
            if vllm_opts["models_url"]:
                _wait_for_vllm_ready(
                    models_url=vllm_opts["models_url"],
                    expected_model_name=mode_config.get("llm", {}).get("model_name", ""),
                    timeout_sec=vllm_opts["timeout_sec"],
                )
                print(f"[vllm] 服务就绪: {vllm_opts['models_url']}")

        if had_original:
            shutil.copy2(EXPERIMENT_CONFIG_PATH, backup_path)

        # 先把目标模型 YAML 落到 experiment_config.yaml，满足“按模型配置切换”的可见流程。
        shutil.copy2(base_config_path, EXPERIMENT_CONFIG_PATH)
        print(f"[mode={mode_name}] 已复制目标 YAML: {base_config_path} -> {EXPERIMENT_CONFIG_PATH}")

        # 再写入模式级覆盖（thinking / max_samples / repeats / model_name）。
        shutil.copy2(temp_config_path, EXPERIMENT_CONFIG_PATH)
        print(f"[mode={mode_name}] 已应用模式覆盖参数到: {EXPERIMENT_CONFIG_PATH}")

        print(f"[mode={mode_name}] 使用基础配置: {base_config_path}")
        print(
            f"[mode={mode_name}] model={mode_config.get('llm', {}).get('model_name')}, "
            f"thinking={enable_thinking}, max_samples={max_samples}, repeats={mode_config.get('repeats')}"
        )
        print(f"[mode={mode_name}] 开始运行 run_all.py ...")

        result = subprocess.run([sys.executable, str(RUN_ALL_PATH)], cwd=ROOT)
        return result.returncode
    finally:
        if vllm_opts["enabled"] and vllm_opts["stop_after_run"]:
            # 先结束本脚本拉起的进程树，再 fuser/pkill；顺序反了会导致子进程仍占满 GPU
            _kill_vllm_launcher_process(started_proc)
            if vllm_opts.get("sigterm_gpu_vllm_processes", True):
                _cleanup_vllm_processes_on_gpu(int(vllm_opts.get("gpu_index", 0)))
            if vllm_opts.get("stop_command"):
                _stop_vllm(vllm_opts["stop_command"])
            _extra_stop_vllm(vllm_opts.get("extra_stop_command"))
        if backup_path.exists():
            shutil.copy2(backup_path, EXPERIMENT_CONFIG_PATH)
            backup_path.unlink(missing_ok=True)
        elif not had_original and EXPERIMENT_CONFIG_PATH.exists():
            EXPERIMENT_CONFIG_PATH.unlink()
        temp_config_path.unlink(missing_ok=True)


def build_parser(default_mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{default_mode} 模式运行器")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_CONFIG_PATH,
        help="基础 YAML 配置路径（默认: experiment_config.yaml）",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="可选覆盖 repeats（不传则保留基础配置）",
    )
    vllm_manage = parser.add_mutually_exclusive_group()
    vllm_manage.add_argument(
        "--manage-vllm",
        dest="manage_vllm",
        action="store_true",
        default=None,
        help="强制开启 vLLM 自动管理（启动前清理端口/进程，运行后关闭）",
    )
    vllm_manage.add_argument(
        "--no-manage-vllm",
        dest="manage_vllm",
        action="store_false",
        default=None,
        help="强制关闭 vLLM 自动管理（假设服务已在外部启动）",
    )
    parser.add_argument(
        "--vllm-start-command",
        type=str,
        default=None,
        help="可选覆盖 vllm 启动命令，支持模板 {model_name} {api_url}",
    )
    parser.add_argument(
        "--vllm-stop-command",
        type=str,
        default=None,
        help="可选覆盖 vllm 关闭命令，支持模板 {model_name} {api_url}",
    )
    parser.add_argument(
        "--vllm-models-url",
        type=str,
        default=None,
        help="可选覆盖就绪检查地址（默认从 llm.api_url 推导 /models）",
    )
    parser.add_argument(
        "--vllm-start-timeout-sec",
        type=int,
        default=180,
        help="等待 vLLM 就绪超时时间（秒）",
    )
    parser.add_argument(
        "--no-vllm-stop-after-run",
        action="store_true",
        help="运行后不自动关闭 vLLM",
    )
    return parser
