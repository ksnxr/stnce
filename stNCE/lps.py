import os
import shutil

from PIL import Image, ImageDraw
import torch

from data.data import ImageNet64

NUM_EXAMPLES = 10
# Replace with the saved test_log_densities.pt file to visualize.
LOG_DENSITY_PATHS = {"eval": "TODO-0"}


def _load_imagenet64_val_data():
    dataset = ImageNet64(
        "../imagenet64",
        train=False,
        transform=None,
        download=False,
    )
    return dataset.data


def _topk_extremes(indices, lps, k):
    lowest_positions = torch.topk(lps, k=k, largest=False).indices
    highest_positions = torch.topk(lps, k=k, largest=True).indices

    return {
        "Highest log densities": (indices[highest_positions], lps[highest_positions]),
        "Lowest log densities": (indices[lowest_positions], lps[lowest_positions]),
    }


def _plot_extremes(extremes, images, output_path):
    image_size = 128
    cols = 10
    rows = 2
    group_cols = 5
    gap = 6
    border_width = 4
    cell_size = image_size + 2 * border_width
    width = cols * cell_size + (cols - 1) * gap
    height = rows * cell_size + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    sections = (
        ("Highest log densities", "red", 0),
        ("Lowest log densities", "blue", group_cols),
    )

    for section_title, border_color, col_offset in sections:
        section_indices, _ = extremes[section_title]
        for position, image_index in enumerate(section_indices.tolist()):
            row = position // group_cols
            col = col_offset + position % group_cols
            x = col * (cell_size + gap)
            y = row * (cell_size + gap)
            image = Image.fromarray(images[int(image_index)]).resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            )
            draw.rectangle(
                (x, y, x + cell_size - 1, y + cell_size - 1),
                outline=border_color,
                width=border_width,
            )
            canvas.paste(image, (x + border_width, y + border_width))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def _process_eval(output_dir, eval_name, images):
    log_density_path = LOG_DENSITY_PATHS[eval_name]
    output_path = os.path.join(output_dir, f"{eval_name}_extreme_log_densities.png")
    density_payload = torch.load(log_density_path, map_location="cpu")
    indices = density_payload["indices"].long()
    lps = density_payload["log_densities"].double()

    if indices.numel() != lps.numel():
        raise ValueError(
            f"indices and log_densities must have the same length, got "
            f"{indices.numel()} and {lps.numel()}."
        )
    if torch.any(indices < 0) or torch.any(indices >= len(images)):
        raise ValueError("At least one logged index is outside imagenet64/val_data.")

    extremes = _topk_extremes(indices, lps, NUM_EXAMPLES)
    _plot_extremes(extremes, images, output_path)

    print(
        f"\n{eval_name}: mean={torch.mean(lps):.4f} "
        f"min={torch.min(lps):.4f} max={torch.max(lps):.4f}"
    )
    for section_title, (section_indices, section_lps) in extremes.items():
        print(f"\n{section_title}:")
        for rank, (image_index, log_density) in enumerate(
            zip(section_indices.tolist(), section_lps.tolist()), start=1
        ):
            print(
                f"{rank:2d}. index={int(image_index):5d} log_density={float(log_density):.4f}"
            )
    print(f"\nSaved visualization to {output_path}")


def main():
    images = _load_imagenet64_val_data()

    output_dir = "extreme_figs"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print("Removed extreme_figs")
    os.mkdir(output_dir)
    print("Created extreme_figs")

    for eval_name in LOG_DENSITY_PATHS:
        _process_eval(output_dir, eval_name, images)


if __name__ == "__main__":
    main()
