#!/usr/bin/env bash
# 一键全量验证：在一次性容器里把代码拷进容器内文件系统再跑 pytest。
#
# 为什么要拷贝：直接在 /mnt/e（drvfs/9p）上跑 sqlite 密集的 tests/agent
# 会慢一个数量级（实测 50 分钟跑不完 vs 容器内 16 分钟）。
#
# 用法（在 WSL 里执行，或从 Git Bash 用 wsl 调用）：
#   wsl -e bash scripts/run_tests.sh                       # 全量
#   wsl -e bash scripts/run_tests.sh tests/core           # 指定目录
#   wsl -e bash scripts/run_tests.sh tests/core/test_x.py::test_y
#
# 镜像默认 xiaomachi-bot:local，可用 XIAOMACHI_IMAGE 覆盖。
set -euo pipefail

IMAGE="${XIAOMACHI_IMAGE:-xiaomachi-bot:local}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=(tests/agent tests/core tests/adapters)
fi

echo "full verification: image=$IMAGE targets=${TARGETS[*]}"

docker run --rm -v "$ROOT:/src:ro" "$IMAGE" sh -c "
  set -e
  mkdir -p /work
  cp -r /src/app /src/configs /src/scripts /src/tests /src/pyproject.toml /work/
  cd /work
  python -m pytest ${TARGETS[*]} -q -p no:cacheprovider
"
