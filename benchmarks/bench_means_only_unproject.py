#!/usr/bin/env python3
"""Wall-clock: full covariance SVD unproject vs means-only affine.

Mirrors ``sharp.utils.gaussians.apply_transform`` (compose → map → CPU fp64
SVD) against the fork's means-only path. Synthetic Gaussians at SHARP density.
No checkpoint, plyfile, or scipy required.

Example:
  python benchmarks/bench_means_only_unproject.py --n 1180000 --device cpu
  python benchmarks/bench_means_only_unproject.py --n 1180000 --device cuda
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import torch


def compose_covariance_matrices(
    quaternions: torch.Tensor, singular_values: torch.Tensor
) -> torch.Tensor:
    """Same construction as upstream ``compose_covariance_matrices``."""
    # Rotation matrices from quaternions (wxyz).
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    rotations = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
    scales = singular_values
    # R @ diag(s^2) @ R^T
    scaled = rotations * scales.unsqueeze(-2)
    return scaled @ rotations.transpose(-1, -2)


def decompose_covariance_svd(covariance_matrices: torch.Tensor) -> torch.Tensor:
    """Upstream bottleneck: detach → CPU float64 → batched 3x3 SVD."""
    cov = covariance_matrices.detach().cpu().to(torch.float64)
    rotations, singular_values_2, _ = torch.linalg.svd(cov)
    # Return singular values only for timing parity (quat rebuild is extra).
    return singular_values_2.sqrt()


def apply_transform_full(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    linear = transform[..., :3, :3]
    offset = transform[..., :3, 3]
    mean_out = means @ linear.T + offset
    cov = compose_covariance_matrices(quats, scales)
    cov = linear @ cov @ linear.transpose(-1, -2)
    _ = decompose_covariance_svd(cov)
    return mean_out


def apply_transform_means_only(
    means: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    linear = transform[..., :3, :3]
    offset = transform[..., :3, 3]
    return means @ linear.T + offset


def _synthetic(n: int, device: torch.device):
    means = torch.randn(n, 3, device=device, dtype=torch.float32) * 0.3
    quats = torch.randn(n, 4, device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 0.01 + 1e-4
    transform = torch.eye(4, device=device, dtype=torch.float32)[:3].clone()
    transform[0, 0] = 2.1
    transform[1, 1] = 2.1
    transform[2, 2] = 1.7
    transform[:, 3] = torch.tensor([0.01, -0.02, 0.5], device=device)
    return means, quats, scales, transform


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, device: torch.device, repeats: int) -> float:
    fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / repeats


def _run_one(n: int, device: torch.device, repeats: int) -> tuple[float, float, float]:
    means, quats, scales, transform = _synthetic(n, device)
    full_s = _time(
        lambda: apply_transform_full(means, quats, scales, transform),
        device,
        repeats,
    )
    means_s = _time(
        lambda: apply_transform_means_only(means, transform),
        device,
        repeats,
    )
    full_means = apply_transform_full(means, quats, scales, transform)
    fast_means = apply_transform_means_only(means, transform)
    max_abs = (full_means - fast_means).abs().max().item()
    return full_s, means_s, max_abs


def _write_plot(
    sizes: list[int],
    full_times: list[float],
    means_times: list[float],
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [f"N={n:,}" for n in sizes]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(
        x - w / 2,
        full_times,
        w,
        label="Upstream full unproject (CPU fp64 SVD)",
        color="#c44e52",
    )
    b2 = ax.bar(x + w / 2, means_times, w, label="means-only affine", color="#4c72b0")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall time (s, log scale)")
    ax.set_title("SHARP metric unproject: full SVD vs means-only")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(1e-3, max(full_times) * 4)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    speed = full_times[-1] / max(means_times[-1], 1e-9)
    ax.annotate(
        f"{speed:.0f}x at N={sizes[-1]:,}",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#bbbbbb"),
    )
    fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote={path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[300_000, 1_180_000])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--plot",
        default=str(Path(__file__).resolve().parents[1] / "docs" / "bench_means_only.png"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    full_times: list[float] = []
    means_times: list[float] = []
    print(f"host={platform.node()}")
    print(f"platform={platform.platform()}")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}")
    print(f"device={device}")
    for n in args.n:
        full_s, means_s, max_abs = _run_one(n, device, args.repeats)
        full_times.append(full_s)
        means_times.append(means_s)
        print(f"n={n}")
        print(f"full_unproject_s={full_s:.4f}")
        print(f"means_only_s={means_s:.4f}")
        print(f"speedup_vs_full={full_s / max(means_s, 1e-9):.2f}x")
        print(f"mean_max_abs_err={max_abs:.3e}")
    _write_plot(list(args.n), full_times, means_times, args.plot)


if __name__ == "__main__":
    main()
