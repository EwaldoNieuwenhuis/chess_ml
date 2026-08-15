"""
Visual Quality Assurance & Diagnostic Overlay Tool for the Hybrid Chess Dataset.

Renders normalized bounding boxes, canonical class labels, and bottom-center base
contact points on sample images across train, validation, and test splits to visually
verify label accuracy, class mappings, and parallax footprint alignment (US-2.3.2 / ADR-008).

Usage:
    uv run python scripts/visualize_hybrid_dataset.py --help
    uv run python scripts/visualize_hybrid_dataset.py --dataset-dir data/hybrid_chess --samples 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Ensure UTF-8 output encoding across Windows shells/consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel

from src.dataset.normalizer import CanonicalClassMapper

console = Console(force_terminal=True, legacy_windows=False)

# Color palettes (BGR format for OpenCV)
# White pieces: Gold/Cyan tones; Black pieces: Red/Magenta/Crimson tones
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    # White pieces (0..5)
    0: (0, 215, 255),    # Pawn - Gold
    1: (255, 191, 0),    # Knight - Deep Sky Blue
    2: (0, 255, 255),    # Bishop - Yellow
    3: (255, 144, 30),   # Rook - Dodger Blue
    4: (0, 165, 255),    # Queen - Orange
    5: (0, 255, 127),    # King - Spring Green
    # Black pieces (6..11)
    6: (0, 0, 220),      # Pawn - Crimson
    7: (128, 0, 128),    # Knight - Purple
    8: (34, 34, 178),    # Bishop - Firebrick
    9: (180, 105, 255),  # Rook - Hot Pink
    10: (0, 0, 139),     # Queen - Dark Red
    11: (130, 0, 75),    # King - Indigo
}


def draw_bounding_box(
    img: np.ndarray,
    class_id: int,
    xc: float,
    yc: float,
    w: float,
    h: float,
    class_name: str,
    draw_contact_point: bool = True,
) -> None:
    """Draws a single normalized bounding box and contact anchor onto an image."""
    img_h, img_w = img.shape[:2]

    # Convert normalized (xc, yc, w, h) to pixel (xmin, ymin, xmax, ymax)
    x_min = int(round((xc - w / 2.0) * img_w))
    y_min = int(round((yc - h / 2.0) * img_h))
    x_max = int(round((xc + w / 2.0) * img_w))
    y_max = int(round((yc + h / 2.0) * img_h))

    # Clamp to image boundaries
    x_min = max(0, min(img_w - 1, x_min))
    y_min = max(0, min(img_h - 1, y_min))
    x_max = max(0, min(img_w - 1, x_max))
    y_max = max(0, min(img_h - 1, y_max))

    color = CLASS_COLORS_BGR.get(class_id, (0, 255, 0))
    thickness = max(2, int(round(min(img_w, img_h) / 300.0)))

    # Draw box outline
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

    # Draw label badge
    label_text = f"{class_id}:{class_name}"
    font_scale = max(0.4, min(img_w, img_h) / 1200.0)
    font_thick = max(1, int(round(font_scale * 2.0)))
    (lbl_w, lbl_h), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
    )

    badge_ymin = max(0, y_min - lbl_h - baseline - 4)
    badge_ymax = y_min
    badge_xmax = min(img_w, x_min + lbl_w + 6)

    # Background filled rectangle for label text
    cv2.rectangle(
        img,
        (x_min, badge_ymin),
        (badge_xmax, badge_ymax),
        color,
        -1,
    )
    # White text on dark/colored badge
    text_color = (0, 0, 0) if class_id < 6 else (255, 255, 255)
    cv2.putText(
        img,
        label_text,
        (x_min + 3, badge_ymax - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thick,
        cv2.LINE_AA,
    )

    # Draw bottom-center base contact anchor (xc, ymax)
    if draw_contact_point:
        base_x = int(round(xc * img_w))
        base_y = y_max
        radius = max(3, int(round(thickness * 1.5)))
        # Outer black ring
        cv2.circle(img, (base_x, base_y), radius + 1, (0, 0, 0), -1)
        # Inner colored dot
        cv2.circle(img, (base_x, base_y), radius, (0, 255, 255), -1)


def render_annotated_sample(
    img_path: Path,
    lbl_path: Path | None,
    mapper: CanonicalClassMapper,
    draw_contact_point: bool = True,
) -> np.ndarray:
    """Loads an image, parses its YOLO label, and returns an annotated RGB/BGR numpy array."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image at: {img_path}")

    # Check if empty/negative sample
    if lbl_path is None or not lbl_path.exists() or lbl_path.stat().st_size == 0:
        # Overlay negative badge
        cv2.putText(
            img,
            "[NEGATIVE SAMPLE: EMPTY BACKGROUND]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cid = int(parts[0])
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    cname = mapper.get_class_info(cid).name
                    draw_bounding_box(img, cid, xc, yc, w, h, cname, draw_contact_point=draw_contact_point)
                except Exception:
                    continue

    return img


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visual quality assurance tool for hybrid chess dataset annotations."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/hybrid_chess",
        help="Path to hybrid dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/samples/hybrid_dataset_qa",
        help="Output directory to save annotated QA images.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random samples to render per split (train, val, test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for reproducible sample selection.",
    )
    parser.add_argument(
        "--no-contact-points",
        action="store_true",
        help="Disable drawing bottom-center base contact points.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = Panel(
        f"[bold white]♟️ Chess ML - Hybrid Dataset Visual QA Inspector[/bold white]\n"
        f"[dim]Renders Bounding Boxes, Class Badges, and Parallax Contact Anchors[/dim]\n\n"
        f"[cyan]Dataset Dir :[/cyan] [yellow]{dataset_path}[/yellow]\n"
        f"[cyan]Output Dir  :[/cyan] [green]{out_dir}[/green]\n"
        f"[cyan]Samples/Split:[/cyan] [magenta]{args.samples}[/magenta]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    console.print()

    if not dataset_path.exists():
        console.print(f"[bold red]❌ Dataset directory not found:[/bold red] {dataset_path}")
        return 1

    mapper = CanonicalClassMapper()
    rng = random.Random(args.seed)

    total_rendered = 0
    for split_name in ("train", "val", "test"):
        img_dir = dataset_path / "images" / split_name
        lbl_dir = dataset_path / "labels" / split_name

        if not img_dir.exists():
            console.print(f"[yellow]Skipping split '{split_name}' (directory not found: {img_dir})[/yellow]")
            continue

        images = [f for f in img_dir.glob("*") if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        if not images:
            console.print(f"[yellow]No images in '{split_name}' split.[/yellow]")
            continue

        sample_subset = rng.sample(images, min(len(images), args.samples))
        split_out_dir = out_dir / split_name
        split_out_dir.mkdir(parents=True, exist_ok=True)

        for img_file in sample_subset:
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            annotated = render_annotated_sample(
                img_file,
                lbl_file if lbl_file.exists() else None,
                mapper=mapper,
                draw_contact_point=not args.no_contact_points,
            )
            dst_file = split_out_dir / f"qa_{img_file.name}"
            cv2.imwrite(str(dst_file), annotated)
            total_rendered += 1

        console.print(
            f"[green]Rendered {len(sample_subset)} QA samples for split '{split_name}' to:[/green] {split_out_dir}"
        )

    console.print(
        f"\n[bold green]✅ Visual QA rendering complete! Total rendered images: {total_rendered}[/bold green]\n"
        f"[dim]Inspect annotated images in: {out_dir}[/dim]\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
