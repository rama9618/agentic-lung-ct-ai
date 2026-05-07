from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


DEFAULT_DATA_DIR = Path("data/Data")
DEFAULT_REPORT_PATH = Path("reports/dataset_summary.txt")
SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jfif",
    ".jpe",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def image_paths(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def percentile_from_histogram(histogram: list[int], percentile: float) -> int | None:
    total = sum(histogram)
    if total == 0:
        return None

    threshold = (percentile / 100) * (total - 1)
    running = 0
    for value, count in enumerate(histogram):
        running += count
        if running > threshold:
            return value
    return len(histogram) - 1


def intensity_stats(histogram: list[int]) -> dict[str, float | int | None]:
    total = sum(histogram)
    if total == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
        }

    values = [value for value, count in enumerate(histogram) if count]
    pixel_sum = sum(value * count for value, count in enumerate(histogram))
    square_sum = sum((value**2) * count for value, count in enumerate(histogram))
    mean = pixel_sum / total
    variance = max((square_sum / total) - (mean**2), 0)

    return {
        "count": total,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "p01": percentile_from_histogram(histogram, 1),
        "p50": percentile_from_histogram(histogram, 50),
        "p99": percentile_from_histogram(histogram, 99),
    }


def format_stats(stats: dict[str, float | int | None]) -> str:
    if stats["count"] == 0:
        return "No valid pixels found."

    return (
        f"count={stats['count']}, min={stats['min']}, max={stats['max']}, "
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"p01={stats['p01']}, p50={stats['p50']}, p99={stats['p99']}"
    )


def top_counter(counter: Counter, limit: int = 20) -> list[tuple[object, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:limit]


def analyze_dataset(data_dir: Path) -> dict:
    summary = {
        "data_dir": data_dir,
        "splits": {},
        "overall": {
            "valid_images": 0,
            "corrupt_files": [],
            "sizes": Counter(),
            "modes": Counter(),
            "histogram": [0] * 256,
            "non_image_files": [],
        },
    }

    for split in SPLITS:
        split_dir = data_dir / split
        split_summary = {
            "exists": split_dir.exists(),
            "classes": {},
            "valid_images": 0,
            "corrupt_files": [],
            "sizes": Counter(),
            "modes": Counter(),
            "histogram": [0] * 256,
            "non_image_files": [],
        }

        if not split_dir.exists():
            summary["splits"][split] = split_summary
            continue

        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        for class_dir in class_dirs:
            class_summary = {
                "valid_images": 0,
                "corrupt_files": [],
                "sizes": Counter(),
                "modes": Counter(),
                "histogram": [0] * 256,
            }

            for path in image_paths(class_dir):
                try:
                    with Image.open(path) as image:
                        image.verify()

                    with Image.open(path) as image:
                        mode = image.mode
                        size = image.size
                        rgb_histogram = image.convert("RGB").histogram()
                except (OSError, UnidentifiedImageError) as error:
                    relative_path = path.relative_to(data_dir)
                    corrupt_record = f"{relative_path} ({error})"
                    class_summary["corrupt_files"].append(corrupt_record)
                    split_summary["corrupt_files"].append(corrupt_record)
                    summary["overall"]["corrupt_files"].append(corrupt_record)
                    continue

                intensity_histogram = [
                    rgb_histogram[index]
                    + rgb_histogram[index + 256]
                    + rgb_histogram[index + 512]
                    for index in range(256)
                ]

                class_summary["valid_images"] += 1
                class_summary["sizes"][size] += 1
                class_summary["modes"][mode] += 1
                class_summary["histogram"] = [
                    old + new
                    for old, new in zip(class_summary["histogram"], intensity_histogram)
                ]

                split_summary["valid_images"] += 1
                split_summary["sizes"][size] += 1
                split_summary["modes"][mode] += 1
                split_summary["histogram"] = [
                    old + new
                    for old, new in zip(split_summary["histogram"], intensity_histogram)
                ]

                summary["overall"]["valid_images"] += 1
                summary["overall"]["sizes"][size] += 1
                summary["overall"]["modes"][mode] += 1
                summary["overall"]["histogram"] = [
                    old + new
                    for old, new in zip(summary["overall"]["histogram"], intensity_histogram)
                ]

            split_summary["classes"][class_dir.name] = class_summary

        image_parent_dirs = {path.parent for class_dir in class_dirs for path in image_paths(class_dir)}
        for path in sorted(split_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS:
                if path.parent in image_parent_dirs or path.parent == split_dir:
                    relative_path = path.relative_to(data_dir)
                    split_summary["non_image_files"].append(str(relative_path))
                    summary["overall"]["non_image_files"].append(str(relative_path))

        summary["splits"][split] = split_summary

    return summary


def render_counter(counter: Counter, item_label: str, limit: int = 20) -> list[str]:
    if not counter:
        return [f"  No {item_label} found."]

    lines = []
    for item, count in top_counter(counter, limit=limit):
        if isinstance(item, tuple) and len(item) == 2:
            label = f"{item[0]}x{item[1]}"
        else:
            label = str(item)
        lines.append(f"  {label}: {count}")

    remaining = len(counter) - limit
    if remaining > 0:
        lines.append(f"  ... {remaining} more {item_label} omitted")
    return lines


def render_report(summary: dict) -> str:
    lines = [
        "Dataset EDA Summary",
        "===================",
        f"Data directory: {summary['data_dir']}",
        "",
    ]

    overall = summary["overall"]
    lines.extend(
        [
            "Overall",
            "-------",
            f"Valid images: {overall['valid_images']}",
            f"Corrupt files: {len(overall['corrupt_files'])}",
            f"Non-image files skipped: {len(overall['non_image_files'])}",
            f"Pixel intensity stats (RGB channels, 0-255): {format_stats(intensity_stats(overall['histogram']))}",
            "Image modes:",
            *render_counter(overall["modes"], "modes"),
            "Image sizes:",
            *render_counter(overall["sizes"], "sizes"),
            "",
        ]
    )

    for split, split_summary in summary["splits"].items():
        lines.extend(
            [
                f"{split}",
                "-" * len(split),
            ]
        )

        if not split_summary["exists"]:
            lines.extend(["  Split directory missing.", ""])
            continue

        lines.extend(
            [
                f"Valid images: {split_summary['valid_images']}",
                f"Corrupt files: {len(split_summary['corrupt_files'])}",
                f"Non-image files skipped: {len(split_summary['non_image_files'])}",
                f"Pixel intensity stats (RGB channels, 0-255): {format_stats(intensity_stats(split_summary['histogram']))}",
                "",
                "Images per class:",
            ]
        )

        for class_name, class_summary in split_summary["classes"].items():
            lines.append(f"  {class_name}: {class_summary['valid_images']}")

        lines.extend(["", "Image modes:", *render_counter(split_summary["modes"], "modes")])
        lines.extend(["Image sizes:", *render_counter(split_summary["sizes"], "sizes"), ""])

        lines.append("Class details:")
        for class_name, class_summary in split_summary["classes"].items():
            lines.extend(
                [
                    f"  {class_name}",
                    f"    Valid images: {class_summary['valid_images']}",
                    f"    Corrupt files: {len(class_summary['corrupt_files'])}",
                    f"    Pixel intensity stats: {format_stats(intensity_stats(class_summary['histogram']))}",
                    "    Modes:",
                ]
            )
            lines.extend(f"    {line.strip()}" for line in render_counter(class_summary["modes"], "modes"))
            lines.append("    Sizes:")
            lines.extend(f"    {line.strip()}" for line in render_counter(class_summary["sizes"], "sizes"))
        lines.append("")

    if overall["corrupt_files"]:
        lines.extend(["Corrupt Files", "-------------"])
        lines.extend(f"- {path}" for path in overall["corrupt_files"])
        lines.append("")

    if overall["non_image_files"]:
        lines.extend(["Non-Image Files Skipped", "-----------------------"])
        lines.extend(f"- {path}" for path in overall["non_image_files"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze image dataset structure and quality.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_dataset(args.data_dir)
    report = render_report(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote dataset summary to {args.output}")


if __name__ == "__main__":
    main()
