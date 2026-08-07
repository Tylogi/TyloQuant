#!/usr/bin/env python3
"""Render the released DeepSeek-V4-Flash-0731 MFQ/UD KLD comparison.

Run from the repository root:

    uv run --with matplotlib python \
      docs/figures/plot_deepseek_v4_flash_0731_kld.py

The values below mirror the complete ctx=512 table in
docs/deepseek-v4-flash-0731-results.md (updated 2026-08-06).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Result:
    label: str
    size_gib: float
    mean_kld: float


MFQ = (
    Result("MFQ S", 77.519, 0.313576),
    Result("MFQ M", 88.007, 0.244488),
    Result("MFQ L", 98.007, 0.201444),
)

UD = (
    Result("IQ1_S", 76.871, 0.645514),
    Result("IQ1_M", 80.933, 0.581024),
    Result("IQ2_XXS", 84.621, 0.478268),
    Result("IQ2_M", 84.682, 0.478002),
    Result("Q2_K_XL", 90.182, 0.403276),
    Result("IQ3_XXS", 97.051, 0.306343),
    Result("IQ3_S", 108.098, 0.310893),
    Result("Q3_K_M", 119.282, 0.215570),
    Result("Q3_K_XL", 119.402, 0.215313),
    Result("IQ4_NL", 127.277, 0.180695),
    Result("Q4_K_XL", 144.444, 0.149590),
    Result("Q8_K_XL", 150.753, 0.149420),
)

MATCHED_PAIRS = (
    (MFQ[0], UD[0], 51.422),
    (MFQ[1], UD[4], 39.374),
    (MFQ[2], UD[5], 34.242),
)

UD_LABEL_OFFSETS = {
    "IQ1_S": (8, -2),
    "IQ1_M": (8, 2),
    "IQ2_XXS": (-10, 11),
    "IQ2_M": (8, -14),
    "Q2_K_XL": (8, 2),
    "IQ3_XXS": (-8, -17),
    "IQ3_S": (8, 8),
    "Q3_K_M": (-10, 11),
    "Q3_K_XL": (8, -15),
    "IQ4_NL": (8, 8),
    "Q4_K_XL": (-8, 10),
    "Q8_K_XL": (-8, -15),
}


def render(output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cfd7e3",
            "axes.labelcolor": "#46546a",
            "xtick.color": "#637086",
            "ytick.color": "#637086",
            "svg.fonttype": "none",
        }
    )

    mfq_color = "#2477d4"
    ud_color = "#e07a2e"
    pair_color = "#9aa7ba"

    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    ax.plot(
        [result.size_gib for result in UD],
        [result.mean_kld for result in UD],
        color=ud_color,
        linewidth=2.2,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=2,
        label="Unsloth Dynamic (UD)",
        zorder=3,
    )
    ax.plot(
        [result.size_gib for result in MFQ],
        [result.mean_kld for result in MFQ],
        color=mfq_color,
        linewidth=3,
        marker="D",
        markersize=7.5,
        markeredgecolor="white",
        markeredgewidth=1.2,
        label="MFQ",
        zorder=5,
    )

    for mfq, ud, reduction in MATCHED_PAIRS:
        ax.plot(
            [mfq.size_gib, ud.size_gib],
            [mfq.mean_kld, ud.mean_kld],
            color=pair_color,
            linewidth=1.35,
            linestyle=(0, (4, 4)),
            zorder=1,
        )
        x_mid = (mfq.size_gib + ud.size_gib) / 2
        y_mid = (mfq.mean_kld + ud.mean_kld) / 2
        ax.annotate(
            f"{reduction:.1f}% lower",
            (x_mid, y_mid),
            xytext=(7, 0),
            textcoords="offset points",
            color="#435169",
            fontsize=9.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.24",
                "fc": "white",
                "ec": "#d7dde7",
                "alpha": 0.94,
            },
            zorder=6,
        )

    for result in MFQ:
        ax.annotate(
            f"{result.label}  {result.mean_kld:.3f}",
            (result.size_gib, result.mean_kld),
            xytext=(8, -18 if result.label == "MFQ L" else 8),
            textcoords="offset points",
            color="#155ca9",
            fontsize=10,
            fontweight="bold",
            zorder=7,
        )

    for result in UD:
        x_offset, y_offset = UD_LABEL_OFFSETS[result.label]
        ax.annotate(
            result.label,
            (result.size_gib, result.mean_kld),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="right" if x_offset < 0 else "left",
            color="#a9551a",
            fontsize=8.7,
            zorder=6,
        )

    ax.set_xlim(73.5, 154.5)
    ax.set_ylim(0.12, 0.69)
    ax.set_xticks([75, 90, 105, 120, 135, 150])
    ax.set_yticks([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
    ax.set_xlabel("Model size (GiB)", labelpad=10, fontweight="bold")
    ax.set_ylabel("Mean KLD", labelpad=10, fontweight="bold")
    ax.grid(True, color="#dfe5ed", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d7dde7",
        framealpha=0.96,
        ncols=2,
        columnspacing=1.4,
        handlelength=2.5,
    )
    for text in legend.get_texts():
        text.set_color("#38465a")
        text.set_fontweight("bold")

    fig.suptitle(
        "DeepSeek-V4-Flash-0731: Size vs. Mean KLD",
        x=0.085,
        y=0.965,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.085,
        0.895,
        "Full WikiText-2 · ctx=512 · 146,115 scored tokens · lower and farther left is better",
        ha="left",
        fontsize=11.5,
        color="#687386",
    )
    fig.text(
        0.5,
        0.035,
        "Closest-size Mean KLD reduction: S 51.42%  ·  M 39.37%  ·  L 34.24%",
        ha="center",
        fontsize=10.8,
        fontweight="bold",
        color="#435169",
    )

    fig.subplots_adjust(left=0.085, right=0.97, top=0.83, bottom=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format=output.suffix.removeprefix("."), metadata={"Date": None})
    plt.close(fig)

    if output.suffix.lower() == ".svg":
        svg = output.read_text(encoding="utf-8")
        title = (
            "\n <title>DeepSeek-V4-Flash-0731 MFQ versus Unsloth Dynamic "
            "Mean KLD</title>\n"
            " <desc>Full WikiText-2 ctx-512 results compare model size and Mean "
            "KLD. MFQ has lower KLD in all three closest-size comparisons.</desc>"
        )
        svg_start = svg.index("<svg")
        svg_open_end = svg.index(">", svg_start) + 1
        svg = svg[:svg_open_end] + title + svg[svg_open_end:]
        svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
        output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("deepseek-v4-flash-mfq-vs-ud-kld.svg"),
    )
    render(parser.parse_args().output)
