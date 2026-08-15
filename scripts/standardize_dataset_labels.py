"""
CLI Utility to inspect, standardize, and sanitize chess piece annotations across datasets.

Usage:
    uv run python scripts/standardize_dataset_labels.py --help
    uv run python scripts/standardize_dataset_labels.py --dataset all --verify-only
    uv run python scripts/standardize_dataset_labels.py --dataset roboflow_staunton --dry-run
"""

from __future__ import annotations

import argparse
import sys

# Ensure UTF-8 output encoding across Windows shells/consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.dataset.normalizer import (
    DEFAULT_CANONICAL_CONFIG_PATH,
    AnnotationStandardizer,
    CanonicalClassMapper,
    DatasetLabelNormalizer,
    SanitizationStats,
)

console = Console(force_terminal=True, legacy_windows=False)


def display_header(target_dataset: str, config_path: str) -> None:
    panel = Panel(
        f"[bold white]♟️ Chess ML - Label Standardization & Coordinate Sanitizer[/bold white]\n"
        f"[dim]Canonical 12-Class Schema & Normalized YOLO Formatter (ADR-008)[/dim]\n\n"
        f"[cyan]Target Dataset :[/cyan] [yellow]{target_dataset}[/yellow]\n"
        f"[cyan]Classes Config :[/cyan] [green]{config_path}[/green]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    console.print()


def display_canonical_classes(mapper: CanonicalClassMapper) -> None:
    table = Table(
        title="Canonical 12-Class Schema Definition",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Class ID", justify="center", style="bold yellow")
    table.add_column("Canonical Name", style="cyan")
    table.add_column("Piece Type", justify="center", style="green")
    table.add_column("Color", justify="center")
    table.add_column("FEN Char", justify="center", style="bold white")

    for cid in range(12):
        info = mapper.get_class_info(cid)
        color_styled = f"[white]{info.color.value}[/white]" if info.is_white else f"[bold black on white]{info.color.value}[/bold black on white]"
        table.add_row(
            str(cid),
            info.name,
            info.piece_type.value,
            color_styled,
            info.fen_char,
        )

    console.print(table)
    console.print()


def display_sanitization_stats(stats: SanitizationStats, mapper: CanonicalClassMapper, dataset_name: str) -> None:
    table = Table(
        title=f"Sanitization Statistics: {dataset_name}",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Percentage", justify="right", style="yellow")

    total = stats.total_annotations or 1
    table.add_row("Total Parsed Annotations", str(stats.total_annotations), "100.0%")
    table.add_row(
        "Valid Standardized Boxes",
        str(stats.valid_annotations),
        f"{(stats.valid_annotations / total) * 100:.1f}%",
    )
    table.add_row(
        "Epsilon Clamped Boxes",
        str(stats.clamped_annotations),
        f"{(stats.clamped_annotations / total) * 100:.1f}%",
    )
    table.add_row(
        "Discarded Degenerate (<0.005)",
        str(stats.discarded_degenerate),
        f"{(stats.discarded_degenerate / total) * 100:.1f}%",
    )
    table.add_row(
        "Discarded Out-of-Bounds (<40% vis)",
        str(stats.discarded_out_of_bounds),
        f"{(stats.discarded_out_of_bounds / total) * 100:.1f}%",
    )
    table.add_row(
        "Discarded Unknown / Non-Piece Classes",
        str(stats.discarded_unknown_class),
        f"{(stats.discarded_unknown_class / total) * 100:.1f}%",
    )

    console.print(table)
    console.print()

    # Class distribution
    dist_table = Table(
        title=f"Piece Distribution: {dataset_name}",
        show_header=True,
        header_style="bold green",
    )
    dist_table.add_column("ID", justify="center", style="bold yellow")
    dist_table.add_column("Class Name", style="cyan")
    dist_table.add_column("Count", justify="right", style="bold white")
    dist_table.add_column("Share", justify="right", style="yellow")

    valid_total = stats.valid_annotations or 1
    for cid in range(12):
        name = mapper.get_class_info(cid).name
        cnt = stats.class_counts.get(cid, 0)
        dist_table.add_row(
            str(cid),
            name,
            str(cnt),
            f"{(cnt / valid_total) * 100:.1f}%",
        )

    console.print(dist_table)
    console.print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standardize and sanitize heterogeneous chess dataset annotations into canonical 12-class YOLO format."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Target dataset key (e.g. 'roboflow_staunton', 'chessred', 'all').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CANONICAL_CONFIG_PATH),
        help="Path to canonical_classes.yaml configuration file.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Display canonical schema and verify configuration without processing files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate annotations in memory without writing output files.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Custom source directory containing annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory for normalized annotations.",
    )

    args = parser.parse_args()

    display_header(args.dataset, args.config)

    mapper = CanonicalClassMapper(args.config)
    display_canonical_classes(mapper)

    if args.verify_only:
        console.print("[bold green]✅ Canonical class schema and configuration verified successfully.[/bold green]")
        return 0

    standardizer = AnnotationStandardizer(mapper)
    normalizer = DatasetLabelNormalizer(standardizer)

    if args.source_dir:
        src_path = Path(args.source_dir)
        dst_path = Path(args.output_dir or f"{args.source_dir}_standardized")
        if not src_path.exists():
            console.print(f"[bold red]❌ Source directory not found: {src_path}[/bold red]")
            return 1

        console.print(f"🔄 Normalizing directory: [cyan]{src_path}[/cyan] -> [green]{dst_path}[/green]")
        stats = normalizer.normalize_yolo_directory(
            source_labels_dir=src_path,
            target_labels_dir=dst_path,
            dataset_source=args.dataset if args.dataset != "all" else None,
        )
        display_sanitization_stats(stats, mapper, args.dataset)

    console.print("[bold green]✨ Standardization completed successfully.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
