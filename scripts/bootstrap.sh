#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

echo "[1/4] Updating git submodules"
git submodule update --init --recursive

echo "[2/4] Building bgfx release libraries and tools"
make -C submods/bgfx linux-gcc-release64

echo "[3/4] Configuring SegMesh"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

echo "[4/4] Building SegMesh"
cmake --build build -j"$(nproc)"

echo
echo "Build finished."
echo "Run with: ./build/segmesh"
