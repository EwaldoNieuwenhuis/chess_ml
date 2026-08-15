#!/usr/bin/env python3
"""
CLI Tool: Download and Ingest Physical Chess Datasets.

Usage:
    uv run python scripts/download_physical_datasets.py --dataset chessred
    uv run python scripts/download_physical_datasets.py --dataset all
    uv run python scripts/download_physical_datasets.py --dataset all --force
    uv run python scripts/download_physical_datasets.py --verify-only
"""

from __future__ import annotations

import argparse
import logging
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

from src.dataset import DatasetRegistry

# Setup Rich console and logging with legacy Windows safe fallback
console = Console(force_terminal=True, legacy_windows=False)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chess_ml.cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="♟️ Ingest physical chess datasets (ChessReD, Roboflow, Kaggle) into data/raw/physical/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["all", "chessred", "roboflow_staunton", "kaggle_tripod"],
        default="all",
        help="Target dataset to download or 'all' for complete ingestion.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/raw/physical"),
        help="Destination directory for raw physical datasets.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force redownload even if dataset files already exist and verify.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing downloaded dataset files without downloading new data.",
    )
    return parser.parse_args()


def display_banner(output_dir: Path, target: str) -> None:
    banner_text = (
        f"[bold cyan]♟️ Chess ML - Physical Dataset Ingestion Engine[/bold cyan]\n"
        f"[white]Target Dataset :[/white] [bold yellow]{target}[/bold yellow]\n"
        f"[white]Output Path    :[/white] [green]{output_dir.resolve()}[/green]"
    )
    console.print(Panel(banner_text, border_style="cyan"))


def verify_dataset_directory(target_dir: Path) -> dict[str, int]:
    """Inspect contents of a dataset folder and return summary stats."""
    if not target_dir.exists():
        return {"images": 0, "annotations": 0, "total_files": 0}

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    annot_exts = {".json", ".xml", ".txt"}

    images = 0
    annotations = 0
    total = 0

    for path in target_dir.rglob("*"):
        if path.is_file():
            total += 1
            if path.suffix.lower() in image_exts:
                images += 1
            elif path.suffix.lower() in annot_exts:
                annotations += 1

    return {"images": images, "annotations": annotations, "total_files": total}


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    display_banner(output_dir, args.dataset)

    available = DatasetRegistry.list_available()
    targets = available if args.dataset == "all" else [args.dataset]

    if args.verify_only:
        console.print("\n[bold blue]🔍 Verifying existing physical datasets...[/bold blue]\n")
        table = Table(title="Physical Dataset Local Verification Status")
        table.add_column("Dataset Key", style="cyan", no_wrap=True)
        table.add_column("Local Path", style="dim")
        table.add_column("Images", justify="right", style="green")
        table.add_column("Annotations", justify="right", style="yellow")
        table.add_column("Status", style="bold")

        for key in available:
            subdir = output_dir / key
            stats = verify_dataset_directory(subdir)
            status = "[green]Ready[/green]" if stats["total_files"] > 0 else "[red]Missing[/red]"
            table.add_row(
                key,
                str(subdir.relative_to(Path.cwd()) if subdir.is_relative_to(Path.cwd()) else subdir),
                str(stats["images"]),
                str(stats["annotations"]),
                status,
            )

        console.print(table)
        return 0

    console.print(f"\n[bold green]🚀 Initiating ingestion for {len(targets)} dataset(s)...[/bold green]\n")
    results = {}

    for key in targets:
        console.print(f"[bold cyan]>>> Processing dataset: [yellow]{key}[/yellow] <<<[/bold cyan]")
        try:
            downloader = DatasetRegistry.get_downloader(key)
            result_path = downloader.download(base_output_dir=output_dir, force=args.force)
            stats = verify_dataset_directory(result_path)
            results[key] = {
                "success": True,
                "path": result_path,
                "stats": stats,
            }
            console.print(f"[bold green]✓ Completed {key} -> {result_path}[/bold green]\n")
        except Exception as e:
            logger.error(f"Failed processing {key}: {e}", exc_info=True)
            results[key] = {
                "success": False,
                "error": str(e),
            }
            console.print(f"[bold red]✗ Failed {key}: {e}[/bold red]\n")

    # Final summary table
    summary_table = Table(title="Physical Dataset Ingestion Summary")
    summary_table.add_column("Dataset", style="cyan")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Total Files", justify="right")
    summary_table.add_column("Details", style="dim")

    for key, res in results.items():
        if res["success"]:
            total_files = str(res["stats"]["total_files"])
            summary_table.add_row(key, "[green]SUCCESS[/green]", total_files, str(res["path"]))
        else:
            summary_table.add_row(key, "[red]FAILED[/red]", "0", res.get("error", "Error"))

    console.print(summary_table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
