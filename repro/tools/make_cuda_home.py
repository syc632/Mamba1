"""Assemble a usable CUDA_HOME from the *extracted* (not installed) CUDA Windows installer.

The CUDA Windows installer normally requires Administrator (UAC). Instead we:
  1) download cuda_<ver>_windows.exe (local installer)
  2) extract the needed components with 7-Zip (no admin)
  3) merge them into the standard CUDA_HOME layout:  bin/ include/ lib/x64/ nvvm/

Usage:
    python repro/tools/make_cuda_home.py --raw D:/cuda/raw --out D:/cuda/v13.2
"""
import argparse
import os
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path, label=""):
    if not src.exists():
        print(f"  [skip] missing {src}")
        return 0
    n = 0
    dst.mkdir(parents=True, exist_ok=True)
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        target = dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            t = target / f
            if not t.exists() or s.stat().st_size != t.stat().st_size:
                shutil.copy2(s, t)
            n += 1
    print(f"  [ok] {label or src.name}: {n} files -> {dst}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="D:/cuda/raw")
    ap.add_argument("--out", default="D:/cuda/v13.2")
    args = ap.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[cuda] assembling {out} from {raw}")
    total = 0

    # --- bin: nvcc toolchain + cicc ---
    total += copy_tree(raw / "cuda_nvcc/nvcc/bin", out / "bin", "nvcc/bin")
    cicc = raw / "libnvvm/nvvm/nvvm/bin/cicc.exe"
    if cicc.exists():
        shutil.copy2(cicc, out / "bin" / "cicc.exe")
        print("  [ok] cicc.exe -> bin/")
    # runtime DLLs must be findable next to nvcc (and on PATH)
    total += copy_tree(raw / "cuda_cudart/cudart/bin/x64", out / "bin", "cudart/bin/x64")

    # --- include: host + device headers ---
    total += copy_tree(raw / "cuda_cudart/cudart/include", out / "include", "cudart/include")
    total += copy_tree(raw / "cuda_nvcc/nvcc/include", out / "include", "nvcc/include")
    total += copy_tree(raw / "cuda_crt/crt/include/crt", out / "include/crt", "crt headers")
    cccl = raw / "cuda_cccl/thrust/include/cccl"
    for sub in ("cub", "thrust", "cuda", "nv"):
        total += copy_tree(cccl / sub, out / "include" / sub, f"cccl/{sub}")

    # --- lib: import libraries for MSVC ---
    total += copy_tree(raw / "cuda_cudart/cudart/lib/x64", out / "lib/x64", "cudart/lib/x64")
    total += copy_tree(raw / "libnvvm/nvvm/nvvm/lib/x64", out / "lib/x64", "nvvm/lib/x64")

    # --- nvvm: cicc/nvvm dll + libdevice (needed by nvcc device codegen) ---
    total += copy_tree(raw / "libnvvm/nvvm/nvvm/bin/x64", out / "nvvm/bin/x64", "nvvm/bin/x64")
    total += copy_tree(raw / "libnvvm/nvvm/nvvm/libdevice", out / "nvvm/libdevice", "libdevice")
    total += copy_tree(raw / "libnvvm/nvvm/nvvm/include", out / "nvvm/include", "nvvm/include")

    print(f"[cuda] done, {total} files under {out}")
    for probe in ["bin/nvcc.exe", "bin/ptxas.exe", "lib/x64/cudart.lib",
                  "include/cuda_runtime.h", "include/cub/cub.cuh",
                  "include/cuda/std/type_traits", "nvvm/libdevice/libdevice.10.bc"]:
        p = out / probe
        print(f"  {'OK ' if p.exists() else 'MISSING'} {probe}")


if __name__ == "__main__":
    main()
