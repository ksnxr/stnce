import pathlib
import pickle
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Polygon
from scipy.ndimage import gaussian_filter

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ebmlearn.default import IMAGE_FOLDER, RESULTS_FOLDER

METHODS = ["tnce", "tcnce", "stnce_white"]
METHOD_TITLES = {
    "tnce": "Temporal",
    "tcnce": "Spatial",
    "stnce_white": "Spatiotemporal",
}
PANEL_METHODS = [
    ("Temporal", "tnce"),
    ("Spatial", "tcnce"),
    ("Spatiotemporal", "stnce_white"),
]
LABEL_FONTSIZE = 27
ERROR_COLOR = "#f51e0e"
SPATIOTEMPORAL_COLOR = "#a71664"
PER_PAIR_RESULTS_PATTERN = "run_synthetic_many_results_*_pair_*.pkl"
REQUIRED_COLUMNS = {
    "loss_name",
    "data_multimodality",
    "data_support_mismatch",
}
METRIC_COLUMNS = ("test_density_r2", "evaluate_density_r2")


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    z = (x - mean) / std
    return np.exp(-0.5 * z**2) / (std * np.sqrt(2.0 * np.pi))


def teaser_interpolant_density(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    std_t = np.sqrt((1.0 - t) ** 2 + t**2)
    component_1 = normal_pdf(x, mean=-2.0 * t, std=std_t)
    component_2 = normal_pdf(x, mean=2.0 * t, std=std_t)
    return 0.5 * component_1 + 0.5 * component_2


def load_result_pickle(pickle_path: pathlib.Path) -> dict:
    with pickle_path.open("rb") as f:
        return pickle.load(f)


def select_metric_row(
    result: dict, step_selection: str
) -> dict[str, float | int | str]:
    history = result.get("history") or []
    if step_selection == "best":
        metrics = (
            max(history, key=lambda item: float(item["val_density_r2"]))
            if history
            else result.get("best_metrics")
        )
    elif step_selection == "last":
        metrics = history[-1] if history else result.get("last_metrics")
    else:
        raise ValueError(
            f"Unknown step selection {step_selection!r}; expected 'best' or 'last'."
        )

    if metrics is None:
        raise ValueError(
            f"Result for loss={result.get('loss_name')!r}, "
            f"mu_pair_index={result.get('mu_pair_index')!r} has no {step_selection} "
            "metrics."
        )

    return {
        "loss_name": result["loss_name"],
        "mu_pair_index": result["mu_pair_index"],
        "mu_1": result["mu_1"],
        "mu_2": result["mu_2"],
        "data_multimodality": result["data_multimodality"],
        "data_support_mismatch": result["data_support_mismatch"],
        "epoch": metrics["epoch"],
        "train_loss": metrics["train_loss"],
        "val_loss": metrics["val_loss"],
        "test_loss": metrics["test_loss"],
        "val_density_r2": metrics["val_density_r2"],
        "test_density_r2": metrics["test_density_r2"],
        "evaluate_density_r2": metrics.get(
            "evaluate_density_r2", metrics["test_density_r2"]
        ),
    }


def load_result_files(
    result_files: list[pathlib.Path],
    step_selection: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for pickle_path in result_files:
        rows.append(select_metric_row(load_result_pickle(pickle_path), step_selection))
    return rows


def resolve_result_files() -> list[pathlib.Path]:
    return sorted(RESULTS_FOLDER.glob(PER_PAIR_RESULTS_PATTERN))


def get_density_r2(row: dict[str, float | int | str]) -> float:
    for metric_column in METRIC_COLUMNS:
        value = row.get(metric_column)
        if value not in (None, ""):
            return float(value)
    raise ValueError(
        "Result row is missing a density R2 column. Expected one of: "
        + ", ".join(METRIC_COLUMNS)
    )


def validate_results(rows: list[dict[str, float | int | str]]) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(rows[0])
    if missing_columns:
        raise ValueError(
            "Pickle result is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )
    if not any(metric_column in rows[0] for metric_column in METRIC_COLUMNS):
        raise ValueError(
            "Pickle result is missing a density R2 column. Expected one of: "
            + ", ".join(METRIC_COLUMNS)
        )

    missing_methods = [
        method
        for method in METHODS
        if not any(row["loss_name"] == method for row in rows)
    ]
    if missing_methods:
        raise ValueError(
            "Missing result rows for method(s): " + ", ".join(missing_methods)
        )


def build_interpolated_grid(rows: list[dict[str, float | int | str]], method: str):
    method_rows = [row for row in rows if row["loss_name"] == method]
    if not method_rows:
        raise ValueError(f"No rows found for method '{method}' in the result pickles.")

    x_points = np.array(
        [float(row["data_multimodality"]) for row in method_rows], dtype=float
    )
    y_points = np.array(
        [float(row["data_support_mismatch"]) for row in method_rows], dtype=float
    )
    z_points = np.array([1.0 - get_density_r2(row) for row in method_rows], dtype=float)

    x_grid = np.linspace(0.0, 1.0, 500)
    y_grid = np.linspace(0.0, 1.0, 500)
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.column_stack([x_points, y_points])

    try:
        from scipy.interpolate import griddata
    except ImportError as exc:
        raise ImportError("scipy is required to interpolate the teaser plot.") from exc

    zz = griddata(points, z_points, (xx, yy), method="linear")
    nearest = griddata(points, z_points, (xx, yy), method="nearest")
    zz = np.where(np.isnan(zz), nearest, zz)
    zz = gaussian_filter(zz, sigma=10.0)

    return zz


def draw_annotation_arrow(ax, start, end, color, *, double_headed=False):
    """Place solid arrows in panel-relative coordinates, above the densities."""
    if double_headed:
        # The density panel is square, so axes coordinates preserve these shapes.
        start, end = np.asarray(start), np.asarray(end)
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([-direction[1], direction[0]])
        head_length, head_half_width, shaft_half_width = 0.105, 0.052, 0.013
        near = start + head_length * direction
        far = end - head_length * direction
        vertices = [
            start,
            near + head_half_width * normal,
            near + shaft_half_width * normal,
            far + shaft_half_width * normal,
            far + head_half_width * normal,
            end,
            far - head_half_width * normal,
            far - shaft_half_width * normal,
            near - shaft_half_width * normal,
            near - head_half_width * normal,
        ]
        ax.add_patch(Polygon(
            vertices, closed=True, transform=ax.transAxes,
            facecolor=color, edgecolor="none", zorder=5,
        ))
        return

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="simple,head_length=1,head_width=1,tail_width=0.26",
            mutation_scale=30,
            linewidth=0,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
    )


def draw_error_annotations(ax: plt.Axes, method: str) -> None:
    ax.text(
        0.5, 0.08,
        "No error" if method == "stnce_white" else "Error increases",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=26, fontweight="bold", color=ERROR_COLOR,
        zorder=6,
    )
    if method == "tnce":
        draw_annotation_arrow(ax, (0.51, 0.28), (0.51, 0.76), ERROR_COLOR)
    elif method == "tcnce":
        draw_annotation_arrow(ax, (0.25, 0.47), (0.75, 0.47), ERROR_COLOR)


def draw_teaser_panel(ax: plt.Axes) -> None:
    num_t = 240
    num_x_bins = 320
    x_limits = (-3.3, 3.3)

    t_edges = np.linspace(0.0, 1.0, num_t + 1)
    x_edges = np.linspace(x_limits[0], x_limits[1], num_x_bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
    t_grid, x_grid = np.meshgrid(t_centers, x_centers, indexing="ij")
    density = teaser_interpolant_density(x_grid, t_grid)

    teaser_cmap = LinearSegmentedColormap.from_list(
        "white_to_blue_violet",
        [
            (0.0, "#ffffff"),
            (0.3, "#ffffff"),
            (0.4, "#d7def7"),
            (1.0, "#4b5fb8"),
        ],
    )
    ax.contourf(t_grid, x_grid, density, levels=100, cmap=teaser_cmap)

    ax.set_title("Density", fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("t", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("x", labelpad=18, fontsize=LABEL_FONTSIZE)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-4.5, 3.3)
    ax.set_box_aspect(1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_yticks([])
    ax.tick_params(length=4, width=1)

    draw_annotation_arrow(ax, (0.25, 0.23), (0.25, 0.62), "black", double_headed=True)
    draw_annotation_arrow(ax, (0.32, 0.29), (0.76, 0.29), "black", double_headed=True)
    draw_annotation_arrow(
        ax, (0.32, 0.59), (0.81, 0.38), SPATIOTEMPORAL_COLOR, double_headed=True
    )
    for x, y, label, color, align in [
        (0.02, 0.82, "Spatial", "black", "left"),
        (0.57, 0.10, "Temporal", "black", "center"),
        (0.71, 0.69, "Spatio-\ntemporal", SPATIOTEMPORAL_COLOR, "center"),
    ]:
        ax.text(
            x, y, label, transform=ax.transAxes,
            ha=align, va="center", fontsize=LABEL_FONTSIZE,
            color=color, linespacing=0.9, zorder=6,
        )


def save_teaser_plot(
    result_files: list[pathlib.Path],
    step_selection: str,
    output_path: pathlib.Path,
) -> None:
    rows = load_result_files(result_files, step_selection)
    if not rows:
        raise ValueError(f"No data found in {len(result_files)} result pickle file(s).")
    validate_results(rows)

    plt.rcParams.update(
        {
            "font.size": 21,
            "axes.labelsize": LABEL_FONTSIZE,
            "axes.titlesize": LABEL_FONTSIZE,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    fig.subplots_adjust(left=0.055, right=0.88, bottom=0.2, top=0.83, wspace=0.45)
    draw_teaser_panel(axes[0])

    vmin = 0.0
    vmax = 1.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "white_to_red", ["#ffffff", "#b30000"]
    )
    cmap.set_bad(color="#bdbdbd")
    image = None

    for ax, (title, method) in zip(axes[1:], PANEL_METHODS):
        zz = build_interpolated_grid(rows, method)
        image = ax.imshow(
            zz,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=[0.0, 1.0, 0.0, 1.0],
            aspect="auto",
            interpolation="bicubic",
        )
        ax.set_title(title, fontsize=LABEL_FONTSIZE)
        ax.set_xlabel("multimodality", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("support mismatch", fontsize=LABEL_FONTSIZE)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_box_aspect(1.0)
        ax.set_xticks([0.0, 1.0])
        ax.set_yticks([0.0, 1.0])
        draw_error_annotations(ax, method)

    colorbar_ax = fig.add_axes([0.925, 0.22, 0.022, 0.50])
    colorbar = fig.colorbar(
        ScalarMappable(norm=image.norm, cmap=cmap),
        cax=colorbar_ax,
    )
    colorbar.set_ticks([0.0, 1.0])
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.set_title("error", pad=8, fontsize=LABEL_FONTSIZE)

    output_paths = [
        output_path.with_suffix(".png"),
        output_path.with_suffix(".pdf"),
    ]
    for path in output_paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {', '.join(str(path) for path in output_paths)}")


def main() -> None:
    result_files = resolve_result_files()
    if not result_files:
        raise ValueError(
            "No result pickles found. Expected files matching "
            f"{PER_PAIR_RESULTS_PATTERN!r} in {RESULTS_FOLDER}."
        )

    save_teaser_plot(
        result_files=result_files,
        step_selection="best",
        output_path=IMAGE_FOLDER / "teaser_best.pdf",
    )
    save_teaser_plot(
        result_files=result_files,
        step_selection="last",
        output_path=IMAGE_FOLDER / "teaser_last.pdf",
    )


if __name__ == "__main__":
    main()
