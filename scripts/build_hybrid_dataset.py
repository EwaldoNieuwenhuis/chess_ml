"""
CLI Script to build a balanced, deduplicated hybrid chess dataset for YOLO training.

Merges physical 3D and digital 2D chess datasets into data/hybrid_chess/ with stratified
splits (70% train / 15% val / 15% test), generates canonical data.yaml, and exports manifest.json.

Usage:
    uv run python scripts/build_hybrid_dataset.py --help
    uv run python scripts/build_hybrid_dataset.py --dry-run
    uv run python scripts/build_hybrid_dataset.py --output-dir data/hybrid_chess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding across Windows shells/consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.dataset.builder import (
    DatasetManifest,
    DatasetSplitRatio,
    HybridDatasetBuilder,
    ImageSample,
)
from src.dataset.normalizer import CanonicalClassMapper
from src.schemas.contracts import DomainType

console = Console(force_terminal=True, legacy_windows=False)


def display_header(output_dir: str, ratios: DatasetSplitRatio, seed: int, dry_run: bool) -> None:
    status = "[bold yellow]DRY-RUN (No files written)[/bold yellow]" if dry_run else "[bold green]PRODUCTION BUILD[/bold green]"
    panel = Panel(
        f"[bold white]♟️ Chess ML - Hybrid Dataset Builder & YOLO Splitter[/bold white]\n"
        f"[dim]Merges Physical 3D & Digital 2D Subsets with SHA-256 Deduplication (US-2.3.2 / ADR-008)[/dim]\n\n"
        f"[cyan]Output Directory :[/cyan] [yellow]{output_dir}[/yellow]\n"
        f"[cyan]Split Ratios     :[/cyan] [magenta]Train: {ratios.train:.0%} | Val: {ratios.val:.0%} | Test: {ratios.test:.0%}[/magenta]\n"
        f"[cyan]Random Seed      :[/cyan] [white]{seed}[/white]\n"
        f"[cyan]Execution Mode   :[/cyan] {status}",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    console.print()


def display_manifest_report(manifest: DatasetManifest, mapper: CanonicalClassMapper) -> None:
    # 1. Overall Dataset Summary
    summary_table = Table(
        title="Hybrid Dataset Overall Summary",
        show_header=True,
        header_style="bold blue",
    )
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right", style="bold white")

    summary_table.add_row("Total Scanned Images", str(manifest.total_images))
    summary_table.add_row("Cryptographic Duplicates Filtered", f"[red]{manifest.total_duplicates_removed}[/red]")
    summary_table.add_row("Unique Images Retained", f"[green]{manifest.total_unique_images}[/green]")
    summary_table.add_row("Total Annotated Pieces", f"[yellow]{manifest.total_pieces}[/yellow]")
    summary_table.add_row("Physical 3D Samples", str(manifest.domain_totals.get("physical_3d", 0)))
    summary_table.add_row("Digital 2D Samples", str(manifest.domain_totals.get("digital_2d", 0)))

    console.print(summary_table)
    console.print()

    # 2. Partition Splits Table
    splits_table = Table(
        title="Stratified Partition Allocation",
        show_header=True,
        header_style="bold magenta",
    )
    splits_table.add_column("Partition", justify="center", style="bold yellow")
    splits_table.add_column("Total Images", justify="right", style="white")
    splits_table.add_column("Physical 3D", justify="right", style="cyan")
    splits_table.add_column("Digital 2D", justify="right", style="green")
    splits_table.add_column("Negative (0-byte)", justify="right", style="magenta")
    splits_table.add_column("Total Pieces", justify="right", style="yellow")
    splits_table.add_column("Share", justify="right", style="dim white")

    tot_imgs = manifest.total_unique_images or 1
    for s_name in ("train", "val", "test"):
        if s_name in manifest.splits:
            s = manifest.splits[s_name]
            share = (s.total_images / tot_imgs) * 100
            splits_table.add_row(
                s_name.upper(),
                str(s.total_images),
                str(s.physical_count),
                str(s.digital_count),
                str(s.negative_count),
                str(s.total_pieces),
                f"{share:.1f}%",
            )

    console.print(splits_table)
    console.print()

    # 3. Class Distribution Across Splits
    dist_table = Table(
        title="Class Distribution Across Splits",
        show_header=True,
        header_style="bold green",
    )
    dist_table.add_column("ID", justify="center", style="bold yellow")
    dist_table.add_column("Class Name", style="cyan")
    dist_table.add_column("Train", justify="right", style="white")
    dist_table.add_column("Val", justify="right", style="white")
    dist_table.add_column("Test", justify="right", style="white")
    dist_table.add_column("Total", justify="right", style="bold yellow")

    for cid in range(12):
        name = mapper.get_class_info(cid).name
        tr_cnt = manifest.splits.get("train", {}).class_distribution.get(cid, 0) if "train" in manifest.splits else 0
        va_cnt = manifest.splits.get("val", {}).class_distribution.get(cid, 0) if "val" in manifest.splits else 0
        te_cnt = manifest.splits.get("test", {}).class_distribution.get(cid, 0) if "test" in manifest.splits else 0
        tot_cnt = tr_cnt + va_cnt + te_cnt
        dist_table.add_row(
            str(cid),
            name,
            str(tr_cnt),
            str(va_cnt),
            str(te_cnt),
            str(tot_cnt),
        )

    console.print(dist_table)
    console.print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge and split physical and digital chess datasets into a balanced YOLO hybrid dataset."
    )
    parser.add_argument(
        "--physical-dir",
        type=str,
        default="data/standardized/physical",
        help="Source directory for standardized physical 3D datasets.",
    )
    parser.add_argument(
        "--digital-dir",
        type=str,
        default="data/standardized/digital",
        help="Source directory for standardized digital 2D datasets.",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="data/standardized/negative",
        help="Optional directory containing negative background/empty board images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/hybrid_chess",
        help="Destination directory for the compiled hybrid dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of samples for training split (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of samples for validation split (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of samples for test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for deterministic splitting.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="File transfer mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute splits and show distribution metrics without copying files.",
    )
    parser.add_argument(
        "--max-samples-per-domain",
        type=int,
        default=None,
        help="Optional cap on sample count per domain (for rapid testing).",
    )

    args = parser.parse_args()

    split_ratios = DatasetSplitRatio(
        train=args.train_ratio,
        val=args.val_ratio,
        test=args.test_ratio,
    )
    mapper = CanonicalClassMapper()

    display_header(args.output_dir, split_ratios, args.seed, args.dry_run)

    builder = HybridDatasetBuilder(
        output_dir=args.output_dir,
        split_ratios=split_ratios,
        class_mapper=mapper,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )

    # 1. Scan source pools
    scanned_samples: list[ImageSample] = []

    # Physical pool
    phys_dir = Path(args.physical_dir)
    if phys_dir.exists():
        console.print(f"[cyan]Scanning physical datasets in:[/cyan] {phys_dir}")
        phys_samples = builder.scan_directory(phys_dir, domain=DomainType.PHYSICAL_3D, source_name="phys")
        if args.max_samples_per_domain:
            phys_samples = phys_samples[: args.max_samples_per_domain]
        scanned_samples.extend(phys_samples)
    else:
        console.print(f"[yellow]Physical directory not found (skipping):[/yellow] {phys_dir}")

    # Digital pool
    dig_dir = Path(args.digital_dir)
    if dig_dir.exists():
        console.print(f"[cyan]Scanning digital datasets in:[/cyan] {dig_dir}")
        dig_samples = builder.scan_directory(dig_dir, domain=DomainType.DIGITAL_2D, source_name="dig")
        if args.max_samples_per_domain:
            dig_samples = dig_samples[: args.max_samples_per_domain]
        scanned_samples.extend(dig_samples)
    else:
        console.print(f"[yellow]Digital directory not found (skipping):[/yellow] {dig_dir}")

    # Negative pool
    neg_dir = Path(args.negative_dir)
    if neg_dir.exists():
        console.print(f"[cyan]Scanning negative background samples in:[/cyan] {neg_dir}")
        neg_samples = builder.scan_directory(
            neg_dir, domain=DomainType.PHYSICAL_3D, source_name="neg", is_negative=True
        )
        scanned_samples.extend(neg_samples)

    if not scanned_samples:
        console.print(
            "[bold red]❌ No image samples discovered in the provided source paths.[/bold red]\n"
            "[yellow]Hint: Run dataset downloaders and normalizers first, or provide custom --physical-dir / --digital-dir.[/yellow]"
        )
        return 1

    console.print(f"\n[bold green]Found {len(scanned_samples)} total candidate images across all sources.[/bold green]\n")

    # 2. Build dataset
    manifest = builder.build_dataset(scanned_samples, dry_run=args.dry_run)

    # 3. Display summary
    display_manifest_report(manifest, mapper)

    if not args.dry_run:
        console.print(
            f"[bold green]✅ Hybrid dataset successfully generated at:[/bold green] [yellow]{args.output_dir}[/yellow]\n"
            f"[dim]├── data.yaml\n"
            f"├── manifest.json\n"
            f"├── images/ (train, val, test)\n"
            f"└── labels/ (train, val, test)[/dim]\n"
        )
    else:
        console.print("[bold yellow]Dry-run complete. No files were written to disk.[/bold yellow]\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
