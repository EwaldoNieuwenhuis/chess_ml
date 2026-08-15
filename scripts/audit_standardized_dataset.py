"""
Automated Dataset Integrity & Corruption Audit CLI Tool (US-2.3.3 / ADR-008).

Performs exhaustive static and statistical auditing on 100% of YOLO annotation files (.txt)
and corresponding images in standardized and hybrid dataset directories prior to training.

Usage:
    uv run python scripts/audit_standardized_dataset.py --help
    uv run python scripts/audit_standardized_dataset.py --target-dir data/hybrid_chess
    uv run python scripts/audit_standardized_dataset.py --target-dir data/standardized/roboflow_staunton
    uv run python scripts/audit_standardized_dataset.py --all --report-json data/diagnostics/audit_report.json
"""

from __future__ import annotations

import argparse
import json
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
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.dataset.auditor import (
    AuditConfig,
    DatasetAuditReport,
    DatasetIntegrityAuditor,
    ViolationSeverity,
)
from src.dataset.normalizer import (
    DEFAULT_CANONICAL_CONFIG_PATH,
    CanonicalClassMapper,
)

console = Console(force_terminal=True, legacy_windows=False)


def display_header(target_paths: list[Path], strict: bool) -> None:
    path_strs = ", ".join(f"[yellow]{p}[/yellow]" for p in target_paths)
    strict_str = "[bold red]STRICT (Degenerate boxes treated as errors)[/bold red]" if strict else "[dim]STANDARD[/dim]"
    panel = Panel(
        f"[bold white]♟️ Chess ML - Automated Dataset Integrity & Corruption Auditor[/bold white]\n"
        f"[dim]Exhaustive YOLO Coordinate, 0-Byte Negative & Class Balance Verifier (US-2.3.3 / ADR-008)[/dim]\n\n"
        f"[cyan]Target Paths :[/cyan] {path_strs}\n"
        f"[cyan]Audit Mode   :[/cyan] {strict_str}",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    console.print()


def display_pairing_summary(report: DatasetAuditReport) -> None:
    table = Table(
        title=f"Image & Label Synchronicity: {report.dataset_path.name}",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Status / Share", justify="right")

    tot_imgs = report.total_images_scanned or 1
    table.add_row("Total Images Scanned", str(report.total_images_scanned), "100.0%")
    table.add_row("Total Labels Scanned", str(report.total_labels_scanned), "-")
    table.add_row(
        "Matched Image-Label Pairs",
        str(report.matched_pairs_count),
        f"[green]{(report.matched_pairs_count / tot_imgs) * 100:.1f}%[/green]",
    )

    orphan_img_style = "[green]0[/green]" if report.orphaned_images_count == 0 else f"[bold red]{report.orphaned_images_count}[/bold red]"
    table.add_row("Orphaned Images (Missing Label)", orphan_img_style, "[dim]Fatal Error[/dim]" if report.orphaned_images_count > 0 else "[green]OK[/green]")

    orphan_lbl_style = "[green]0[/green]" if report.orphaned_labels_count == 0 else f"[bold red]{report.orphaned_labels_count}[/bold red]"
    table.add_row("Orphaned Labels (Missing Image)", orphan_lbl_style, "[dim]Fatal Error[/dim]" if report.orphaned_labels_count > 0 else "[green]OK[/green]")

    neg_style = "[magenta]" + str(report.negative_samples_count) + "[/magenta]"
    table.add_row("0-Byte Negative Background Samples", neg_style, f"[magenta]{report.negative_sample_ratio:.2f}%[/magenta]")

    corrupt_img_style = "[green]0[/green]" if report.corrupted_images_count == 0 else f"[bold red]{report.corrupted_images_count}[/bold red]"
    table.add_row("Corrupted / Unreadable Images", corrupt_img_style, "[dim]Fatal Error[/dim]" if report.corrupted_images_count > 0 else "[green]OK[/green]")

    console.print(table)
    console.print()


def display_annotation_summary(report: DatasetAuditReport) -> None:
    table = Table(
        title=f"Bounding Box Validity & Geometry: {report.dataset_path.name}",
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Rate / Percentage", justify="right")

    tot_boxes = report.total_boxes_scanned or 1
    table.add_row("Total Scanned Bounding Boxes", str(report.total_boxes_scanned), "100.0%")
    
    valid_pct = (report.valid_boxes_count / tot_boxes) * 100.0
    valid_style = "[green]" if report.corrupted_boxes_count == 0 else "[yellow]"
    table.add_row("Valid Canonical Annotations", f"{valid_style}{report.valid_boxes_count}[/{valid_style}]", f"{valid_pct:.2f}%")

    corrupt_style = "[green]0[/green]" if report.corrupted_boxes_count == 0 else f"[bold red]{report.corrupted_boxes_count}[/bold red]"
    table.add_row("Corrupted / Invalid Bounding Boxes", corrupt_style, f"{(report.corrupted_boxes_count / tot_boxes) * 100:.2f}%")

    degen_style = "[dim white]0[/dim white]" if report.degenerate_boxes_count == 0 else f"[yellow]{report.degenerate_boxes_count}[/yellow]"
    table.add_row("Degenerate Boxes (w, h < 0.005)", degen_style, f"[yellow]{report.degenerate_box_rejection_rate:.4f}%[/yellow]")

    clamp_style = "[dim white]0[/dim white]" if report.clamped_boxes_count == 0 else f"[cyan]{report.clamped_boxes_count}[/cyan]"
    table.add_row("Epsilon Clamped Boxes (Floating Drift)", clamp_style, f"{(report.clamped_boxes_count / tot_boxes) * 100:.2f}%")

    console.print(table)
    console.print()


def display_class_histogram(report: DatasetAuditReport, mapper: CanonicalClassMapper) -> None:
    table = Table(
        title=f"Canonical Class Distribution & Balance Histogram: {report.dataset_path.name}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", justify="center", style="bold yellow")
    table.add_column("Class Name", style="cyan")
    table.add_column("Color", justify="center")
    table.add_column("Type", justify="center", style="green")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Share (%)", justify="right", style="yellow")
    table.add_column("Histogram Visual", style="bold white")

    total_valid = sum(report.class_counts.values()) or 1
    max_count = max(report.class_counts.values()) if report.class_counts else 1
    bar_max_width = 24

    for cid in range(12):
        info = mapper.get_class_info(cid)
        cnt = report.class_counts.get(cid, 0)
        pct = (cnt / total_valid) * 100.0
        
        # Calculate ASCII bar length
        bar_len = int(round((cnt / max(1, max_count)) * bar_max_width))
        bar_char = "█" * bar_len
        bar_styled = f"[bright_yellow]{bar_char}[/bright_yellow]" if info.is_white else f"[magenta]{bar_char}[/magenta]"

        color_styled = "[white]White[/white]" if info.is_white else "[bold black on white]Black[/bold black on white]"
        table.add_row(
            str(cid),
            info.name,
            color_styled,
            info.piece_type.value,
            str(cnt),
            f"{pct:.2f}%",
            bar_styled,
        )

    console.print(table)

    # Color balance summary
    white_pct = (report.white_piece_count / total_valid) * 100.0
    black_pct = (report.black_piece_count / total_valid) * 100.0
    balance_text = (
        f"[bold cyan]⚖️ Piece Color Distribution:[/bold cyan] "
        f"[white]White: {report.white_piece_count} ({white_pct:.1f}%)[/white] | "
        f"[magenta]Black: {report.black_piece_count} ({black_pct:.1f}%)[/magenta]"
    )
    console.print(balance_text)
    console.print()


def display_violations(report: DatasetAuditReport, max_display: int = 25) -> None:
    if not report.violations:
        return

    table = Table(
        title=f"Audit Violations & Diagnostics (Showing top {min(len(report.violations), max_display)} of {len(report.violations)})",
        show_header=True,
        header_style="bold red",
    )
    table.add_column("Severity", justify="center")
    table.add_column("Violation Type", style="cyan")
    table.add_column("File", style="yellow")
    table.add_column("Line", justify="center", style="white")
    table.add_column("Description", style="white")

    for v in report.violations[:max_display]:
        sev_style = "[bold red]ERROR[/bold red]" if v.severity == ViolationSeverity.ERROR else "[bold yellow]WARN[/bold yellow]"
        line_str = str(v.line_number) if v.line_number is not None else "-"
        table.add_row(
            sev_style,
            v.violation_type.value,
            v.file_path.name,
            line_str,
            v.description,
        )

    console.print(table)
    if len(report.violations) > max_display:
        console.print(f"[dim]...and {len(report.violations) - max_display} additional violations omitted from console preview.[/dim]")
    console.print()


def display_final_status(report: DatasetAuditReport) -> None:
    if report.passed:
        panel = Panel(
            f"[bold green]✅ AUDIT PASSED: 100% DATASET INTEGRITY VERIFIED[/bold green]\n\n"
            f"[white]Target: {report.dataset_path}[/white]\n"
            f"[dim]Scanned: {report.total_images_scanned} images | {report.total_boxes_scanned} boxes | "
            f"Errors: {report.error_count} | Warnings: {report.warning_count}[/dim]",
            border_style="green",
            expand=False,
        )
    else:
        panel = Panel(
            f"[bold red]❌ AUDIT FAILED: DATASET CORRUPTION DETECTED ({report.error_count} ERRORS)[/bold red]\n\n"
            f"[white]Target: {report.dataset_path}[/white]\n"
            f"[red]Please review the violation details above and sanitize annotations before training.[/red]",
            border_style="red",
            expand=False,
        )
    console.print(panel)
    console.print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Dataset Integrity & Corruption Audit Tool (US-2.3.3 / ADR-008)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-dir",
        "-t",
        nargs="+",
        type=str,
        default=None,
        help="One or more dataset root directories to audit (e.g. data/hybrid_chess or data/standardized).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Shorthand dataset name under data/standardized/ or data/ (e.g. roboflow_staunton, hybrid_chess).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit both data/standardized/ and data/hybrid_chess/ if they exist.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(DEFAULT_CANONICAL_CONFIG_PATH),
        help="Path to canonical classes YAML configuration.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict audit mode: treats degenerate boxes (w, h < 0.005) as fatal errors.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Path to save structured JSON audit report.",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        default=None,
        help="Path to save Markdown audit report.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode: suppress detailed tables and print minimal summary.",
    )
    return parser.parse_args()


def resolve_target_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

    if args.all:
        std_root = Path("data/standardized")
        if std_root.exists():
            # Add all subdirectories in standardized
            for sub in std_root.iterdir():
                if sub.is_dir():
                    paths.append(sub)
        hybrid_root = Path("data/hybrid_chess")
        if hybrid_root.exists():
            paths.append(hybrid_root)
        if not paths:
            paths.append(Path("data/hybrid_chess"))
        return paths

    if args.dataset:
        cand1 = Path("data/standardized") / args.dataset
        cand2 = Path("data") / args.dataset
        cand3 = Path(args.dataset)
        if cand1.exists():
            paths.append(cand1)
        elif cand2.exists():
            paths.append(cand2)
        elif cand3.exists():
            paths.append(cand3)
        else:
            paths.append(cand1)
        return paths

    if args.target_dir:
        for t in args.target_dir:
            paths.append(Path(t))
        return paths

    # Default fallback: check data/hybrid_chess, then data/standardized
    hybrid_path = Path("data/hybrid_chess")
    if hybrid_path.exists():
        paths.append(hybrid_path)
    else:
        paths.append(Path("data/standardized"))

    return paths


def main() -> int:
    args = parse_args()
    target_paths = resolve_target_paths(args)

    if not args.quiet:
        display_header(target_paths, args.strict)

    mapper = CanonicalClassMapper(args.config)
    config = AuditConfig(
        strict_degenerate_as_error=args.strict,
        config_path=args.config,
    )
    auditor = DatasetIntegrityAuditor(config=config, class_mapper=mapper)

    all_passed = True
    combined_reports: list[DatasetAuditReport] = []

    for target_path in target_paths:
        if not args.quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"[cyan]Auditing dataset at: {target_path}...", total=None)
                report = auditor.audit_dataset(target_path)
                progress.remove_task(task)
        else:
            report = auditor.audit_dataset(target_path)

        combined_reports.append(report)
        if not report.passed:
            all_passed = False

        if not args.quiet:
            display_pairing_summary(report)
            display_annotation_summary(report)
            display_class_histogram(report, mapper)
            display_violations(report)
            display_final_status(report)

    # Optional report exports
    if args.report_json:
        out_json = Path(args.report_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        export_data = [r.to_dict() for r in combined_reports] if len(combined_reports) > 1 else combined_reports[0].to_dict()
        out_json.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        if not args.quiet:
            console.print(f"[green]Audit JSON report exported to:[/green] [yellow]{out_json}[/yellow]")

    if args.report_md:
        out_md = Path(args.report_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md_text = "\n\n---\n\n".join(r.to_markdown(mapper) for r in combined_reports)
        out_md.write_text(md_text, encoding="utf-8")
        if not args.quiet:
            console.print(f"[green]Audit Markdown report exported to:[/green] [yellow]{out_md}[/yellow]")

    # Return exit code 0 on clean dataset, 1 on corruption (for CI automation)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
