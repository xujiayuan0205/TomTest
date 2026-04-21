#!/usr/bin/env bash
# 串行跑多份 experiment_config；后台 nohup，stdout/stderr 写入单一日志文件。
# 用法：
#   ./scripts/run_batch_nohup.sh
#   LOGFILE=/path/to/my.log ./scripts/run_batch_nohup.sh   # 指定固定日志路径（默认带时间戳）

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/logs"

LOGFILE="${LOGFILE:-$ROOT/logs/tomtest_batch_$(date +%Y%m%d_%H%M%S).log}"
export PYTHONUNBUFFERED=1
export ROOT

nohup bash <<'BATCH' >"$LOGFILE" 2>&1 &
cd "$ROOT" || exit 1
for cfg in \
  Qwen3-8B-LoRASelfCoT.yaml \
  Qwen3-8B-LoRASelfCoTFull.yaml \
  Qwen3-8B-LoRASIPReSelfCoT.yaml \
  Qwen3-8B-LoRASIPReSelfCoTFull.yaml \
  Qwen3-8B-SelfCoT.yaml \
  Qwen3-8B-SelfCoTFull.yaml \
  Qwen3-8B-SIPColdStart.yaml \
  Qwen3-8B-ToMRL.yaml
do
  echo "========== $cfg =========="
  date -Is
  cp "experiment_configs/$cfg" experiment_config.yaml && python run_all.py || break
done
echo "========== batch finished =========="
date -Is
BATCH

echo "已在后台启动 (PID=$!)"
echo "日志: $LOGFILE"
echo "查看: tail -f $LOGFILE"
