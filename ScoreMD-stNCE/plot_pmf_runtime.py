#!/usr/bin/env python3
"""Plot supplied ALDP/Chignolin PMF results against training seconds per epoch.

Requires matplotlib. Run: python plot_pmf_runtime.py
Writes PDF to the parent directory's outputs folder, replacing an existing PDF.
Use --output-dir to override the destination or --show to open an interactive window.

stNCE-s denotes the supplied \\tstnceSelf row (and likewise for +DSM).
Error bars reproduce the supplied +/- values; their statistical definition was
not specified. Runtime is per training epoch, not total training or sampling time.
Standalone DSM has runtime data but no PMF data, so it is not plotted.
"""

import argparse
from pathlib import Path


METHODS = (r"tNCE", r"tNCE$+$DSM", r"stNCE-s", r"stNCE$+$DSM", r"FPE")
DATASET_COLORS = {"ALDP": "#0072B2", "Chignolin": "#D55E00"}
MARKERS = ("o", "s", "^", "D", "P")
# All arrays follow METHODS order. PMF entries are (mean, reported uncertainty).
DATA = {
    "ALDP": {
        "runtime": (0.610, 0.780, 0.575, 0.648, 1.608),
        "IID": ((0.095, 0.001), (0.063, 0.002), (0.106, 0.004),
                (0.076, 0.002), (0.098, 0.003)),
        "Langevin": ((0.119, 0.003), (0.066, 0.003), (0.139, 0.001),
                     (0.088, 0.005), (0.104, 0.004)),
    },
    "Chignolin": {
        "runtime": (2.343, 2.866, 2.103, 2.349, 6.462),
        "IID": ((0.654, 0.001), (0.060, 0.015), (0.083, 0.000),
                (0.059, 0.001), (0.037, 0.000)),
        "Langevin": ((3.143, 0.031), (0.162, 0.053), (0.170, 0.014),
                     (0.065, 0.005), (0.039, 0.001)),
    },
}
DSM_RUNTIME = {"ALDP": 0.281, "Chignolin": 0.935}


def make_figure():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.73, bottom=0.22,
                        wspace=0.16)

    for ax, sampler in zip(axes, ("IID", "Langevin")):
        title = "Diffusion sampling" if sampler == "IID" else "Langevin sampling"
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Time per epoch (s, log scale)")
        # Axes coordinates keep the better-direction arrows independent of scale.
        arrow_style = dict(arrowstyle="-|>", color="#222222", lw=1.8,
                           mutation_scale=13)
        ax.annotate("", xy=(0.25, -0.28), xytext=(0.75, -0.28),
                    xycoords="axes fraction", arrowprops=arrow_style,
                    annotation_clip=False)
        ax.tick_params(axis="x", labelbottom=True)
        if ax is axes[0]:
            ax.set_ylabel("PMF error (log scale)")
            ax.annotate("", xy=(-0.23, 0.25), xytext=(-0.23, 0.75),
                        xycoords="axes fraction", arrowprops=arrow_style,
                        annotation_clip=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.43, 9)
        ax.set_ylim(0.025, 5)
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.set_yticks([0.03, 0.1, 0.3, 1, 3])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=0.18)
        for dataset, data in DATA.items():
            color = DATASET_COLORS[dataset]
            for i, marker in enumerate(MARKERS):
                mean, uncertainty = data[sampler][i]
                runtime = data["runtime"][i]
                ax.errorbar(runtime, mean, yerr=uncertainty, fmt=marker,
                            color=color, markersize=8, capsize=3,
                            elinewidth=1.2, markeredgecolor="white",
                            markeredgewidth=0.7, zorder=3)
    dataset_handles = [Line2D([], [], color=color, linewidth=3, label=name)
                       for name, color in DATASET_COLORS.items()]
    fig.legend(handles=dataset_handles, loc="upper center",
               bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False,
               columnspacing=2)
    method_handles = [Line2D([], [], marker=m, color="#555555", linestyle="none",
                             markersize=8, label=name)
                      for name, m in zip(METHODS, MARKERS)]
    fig.legend(handles=method_handles, loc="upper center",
               bbox_to_anchor=(0.55, 0.91), ncol=5, frameon=False, columnspacing=0.8,
               handletextpad=0.4, handlelength=1.1, fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "outputs",
                        help="Output directory (default: outputs beside ScoreMD-with-STNCE)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = make_figure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "pmf_vs_runtime.pdf"
    path.unlink(missing_ok=True)
    fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    print(path.resolve())
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
