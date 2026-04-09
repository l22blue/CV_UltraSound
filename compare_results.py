"""
전처리 방법별 실험 결과 비교 스크립트

outputs/ 폴더의 result_*.json 파일들을 자동 수집하여
전처리 방법별 Accuracy, AUC, Sensitivity, Specificity를
Grouped Bar Chart로 비교합니다.

사용법:
    python compare_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

METRIC_KEYS = ["accuracy", "auc", "sensitivity", "specificity"]
METRIC_LABELS = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
PREPROCESS_ORDER = ["none", "gaussian", "median", "bilateral", "non_local_means"]
PREPROCESS_DISPLAY = {
    "none": "Baseline\n(None)",
    "gaussian": "Gaussian",
    "median": "Median",
    "bilateral": "Bilateral",
    "non_local_means": "Non-Local\nMeans",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def load_results() -> dict[str, dict]:
    """outputs/ 디렉토리에서 result_*.json을 읽어 전처리별 최신 결과를 반환."""
    results: dict[str, tuple[str, dict]] = {}  # preprocessing -> (filename, data)

    for f in sorted(OUTPUT_DIR.glob("result_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        prep = data.get("config", {}).get("preprocessing", "unknown")
        # 파일명 timestamp 순으로 정렬하므로 마지막 것이 최신
        results[prep] = (f.name, data)

    return {k: v[1] for k, v in results.items()}


def print_table(results: dict[str, dict]) -> None:
    """콘솔에 비교 테이블 출력."""
    header = f"{'Method':<20}" + "".join(f"{m:<14}" for m in METRIC_LABELS)
    print("\n" + "=" * len(header))
    print("  전처리 방법별 성능 비교 (Test Set, Patient-Level)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for prep in PREPROCESS_ORDER:
        if prep not in results:
            continue
        metrics = results[prep].get("test_metrics_patient_level", {})
        row = f"{prep:<20}"
        for key in METRIC_KEYS:
            val = metrics.get(key, float("nan"))
            row += f"{val:<14.4f}"
        print(row)

    print("=" * len(header) + "\n")


def plot_comparison(results: dict[str, dict]) -> Path:
    """Grouped Bar Chart를 생성하고 PNG로 저장."""
    present = [p for p in PREPROCESS_ORDER if p in results]
    if not present:
        print("⚠ 비교할 결과가 없습니다.")
        sys.exit(1)

    n_methods = len(present)
    n_metrics = len(METRIC_KEYS)
    x = np.arange(n_methods)
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(10, n_methods * 2.2), 6))

    for i, (key, label, color) in enumerate(zip(METRIC_KEYS, METRIC_LABELS, COLORS)):
        values = []
        for prep in present:
            m = results[prep].get("test_metrics_patient_level", {})
            values.append(m.get(key, 0.0))
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)

        # 바 위에 값 표시
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("전처리 방법별 성능 비교 (Patient-Level Test Metrics)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([PREPROCESS_DISPLAY.get(p, p) for p in present], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # 하단에 실험 조건 표시
    sample_cfg = next(iter(results.values())).get("config", {})
    info_text = f"Epochs: {sample_cfg.get('epochs', '?')}  |  Batch: {sample_cfg.get('batch_size', '?')}  |  Seed: {sample_cfg.get('seed', '?')}  |  Model: ResNet-18"
    ax.text(0.5, -0.12, info_text, transform=ax.transAxes, ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "comparison_chart.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    if not OUTPUT_DIR.exists():
        print(f"⚠ outputs 폴더를 찾을 수 없습니다: {OUTPUT_DIR}")
        sys.exit(1)

    results = load_results()
    if not results:
        print("⚠ outputs/ 폴더에 result_*.json 파일이 없습니다. 먼저 실험을 실행하세요.")
        sys.exit(1)

    print(f"\n📊 {len(results)}개 전처리 방법 결과 발견: {', '.join(results.keys())}")
    print_table(results)

    chart_path = plot_comparison(results)
    print(f"✅ 비교 차트 저장 완료: {chart_path}")


if __name__ == "__main__":
    main()
