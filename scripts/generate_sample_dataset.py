"""
Synthetic Chessboard Sample Dataset Generator.

Generates realistic annotated chess board images (both digital 2D and synthetic perspective physical 3D)
with exact ground-truth canonical YOLO 12-class bounding boxes and 0-byte negative sample files.

Usage:
    uv run python scripts/generate_sample_dataset.py
    uv run python scripts/generate_sample_dataset.py --num-physical 30 --num-digital 30 --num-negative 10
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

import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.panel import Panel

console = Console(force_terminal=True, legacy_windows=False)

# Canonical mapping
# 0..5: White (P, N, B, R, Q, K)
# 6..11: Black (p, n, b, r, q, k)
PIECE_TO_CLASS_ID = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}

UNICODE_PIECES = {
    0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K",
    6: "p", 7: "n", 8: "b", 9: "r", 10: "q", 11: "k",
}

THEMES = [
    {"light": (240, 217, 181), "dark": (181, 136, 99), "name": "classic_wood"},
    {"light": (238, 238, 210), "dark": (118, 150, 86), "name": "chess_com_green"},
    {"light": (240, 240, 240), "dark": (130, 160, 190), "name": "ocean_blue"},
    {"light": (230, 230, 230), "dark": (110, 110, 110), "name": "grayscale"},
]


def create_board_image(
    board: chess.Board,
    img_size: int = 640,
    theme: dict | None = None,
    is_physical: bool = False,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Renders a chessboard with pieces and computes normalized YOLO bounding boxes."""
    if theme is None:
        theme = random.choice(THEMES)

    tile_size = img_size // 8
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # 1. Draw chessboard tiles
    for r in range(8):
        for c in range(8):
            color = theme["light"] if (r + c) % 2 == 0 else theme["dark"]
            y1, y2 = r * tile_size, (r + 1) * tile_size
            x1, x2 = c * tile_size, (c + 1) * tile_size
            # Convert RGB to BGR for OpenCV
            img[y1:y2, x1:x2] = (color[2], color[1], color[0])

    annotations: list[tuple[int, float, float, float, float]] = []

    # Convert to PIL for anti-aliased piece drawing
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for square, piece in board.piece_map().items():
        cid = PIECE_TO_CLASS_ID[(piece.piece_type, piece.color)]
        col_idx = chess.square_file(square)
        row_idx = 7 - chess.square_rank(square)

        # Center in square
        xc_pix = (col_idx + 0.5) * tile_size
        yc_pix = (row_idx + 0.5) * tile_size

        # Piece bounding box dimension (relative to square)
        box_w_pix = tile_size * 0.75
        box_h_pix = tile_size * 0.85
        
        # Bottom-aligned in tile for realistic footprint
        box_xc_pix = xc_pix
        box_yc_pix = (row_idx + 0.5) * tile_size

        # Normalized coordinates
        norm_xc = box_xc_pix / img_size
        norm_yc = box_yc_pix / img_size
        norm_w = box_w_pix / img_size
        norm_h = box_h_pix / img_size

        annotations.append((cid, norm_xc, norm_yc, norm_w, norm_h))

        # Draw piece representation
        piece_color = (255, 255, 255) if piece.color == chess.WHITE else (25, 25, 25)
        outline_color = (0, 0, 0) if piece.color == chess.WHITE else (220, 220, 220)

        # Draw piece circle & text label
        radius = int(tile_size * 0.32)
        circle_bbox = [
            int(xc_pix - radius),
            int(yc_pix - radius),
            int(xc_pix + radius),
            int(yc_pix + radius),
        ]
        draw.ellipse(circle_bbox, fill=piece_color, outline=outline_color, width=2)
        symbol = UNICODE_PIECES[cid]
        # Text symbol
        draw.text(
            (xc_pix - 6, yc_pix - 8),
            symbol,
            fill=outline_color,
        )

    result_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if is_physical:
        # Add subtle perspective tilt and lighting gradient to simulate physical 3D camera
        src_pts = np.float32([[0, 0], [img_size, 0], [img_size, img_size], [0, img_size]])
        inset = random.randint(30, 70)
        dst_pts = np.float32([[inset, inset // 2], [img_size - inset, inset // 2], [img_size, img_size], [0, img_size]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result_bgr = cv2.warpPerspective(result_bgr, M, (img_size, img_size), borderValue=(40, 40, 40))

        # Update annotation bounding box centers via homography
        warped_annots: list[tuple[int, float, float, float, float]] = []
        for cid, xc, yc, w, h in annotations:
            pt = np.array([[[xc * img_size, yc * img_size]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, M)[0][0]
            new_xc = transformed[0] / img_size
            new_yc = transformed[1] / img_size
            # Scale height slightly with perspective
            new_w = w * (0.8 + 0.4 * yc)
            new_h = h * (0.8 + 0.4 * yc)
            if 0.0 <= new_xc <= 1.0 and 0.0 <= new_yc <= 1.0:
                warped_annots.append((cid, new_xc, new_yc, new_w, new_h))
        annotations = warped_annots

    return result_bgr, annotations


def generate_samples(
    num_physical: int = 25,
    num_digital: int = 25,
    num_negative: int = 10,
    base_dir: Path = Path("data/standardized"),
) -> None:
    """Generates synthetic dataset samples into data/standardized/{physical,digital,negative}."""
    phys_dir = base_dir / "physical"
    dig_dir = base_dir / "digital"
    neg_dir = base_dir / "negative"

    for d in (phys_dir, dig_dir, neg_dir):
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]🎨 Generating synthetic sample datasets into: [yellow]{base_dir}[/yellow][/bold cyan]")

    # 1. Generate Physical 3D Samples
    for i in range(num_physical):
        board = chess.Board()
        # Make a few random moves for varied positions
        for _ in range(random.randint(2, 20)):
            moves = list(board.legal_moves)
            if moves:
                board.push(random.choice(moves))

        img, annots = create_board_image(board, is_physical=True)
        img_name = f"sample_phys_{i:04d}.jpg"
        lbl_name = f"sample_phys_{i:04d}.txt"

        cv2.imwrite(str(phys_dir / "images" / img_name), img)
        with open(phys_dir / "labels" / lbl_name, "w", encoding="utf-8") as f:
            for cid, xc, yc, w, h in annots:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    console.print(f"  [green]✓ Generated {num_physical} Physical 3D samples in {phys_dir}[/green]")

    # 2. Generate Digital 2D Samples
    for i in range(num_digital):
        board = chess.Board()
        for _ in range(random.randint(0, 30)):
            moves = list(board.legal_moves)
            if moves:
                board.push(random.choice(moves))

        img, annots = create_board_image(board, is_physical=False)
        img_name = f"sample_dig_{i:04d}.png"
        lbl_name = f"sample_dig_{i:04d}.txt"

        cv2.imwrite(str(dig_dir / "images" / img_name), img)
        with open(dig_dir / "labels" / lbl_name, "w", encoding="utf-8") as f:
            for cid, xc, yc, w, h in annots:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    console.print(f"  [green]✓ Generated {num_digital} Digital 2D samples in {dig_dir}[/green]")

    # 3. Generate Negative Background Samples (Empty Boards)
    empty_board = chess.Board(fen=None)  # Empty board
    for i in range(num_negative):
        img, _ = create_board_image(empty_board, is_physical=(i % 2 == 0))
        img_name = f"sample_neg_{i:04d}.jpg"
        lbl_name = f"sample_neg_{i:04d}.txt"

        cv2.imwrite(str(neg_dir / "images" / img_name), img)
        # Create 0-byte label file per Ultralytics negative sample spec
        (neg_dir / "labels" / lbl_name).touch()

    console.print(f"  [green]✓ Generated {num_negative} 0-byte Negative samples in {neg_dir}[/green]\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic sample chessboards for local pipeline testing.")
    parser.add_argument("--num-physical", type=int, default=25, help="Number of physical 3D board samples.")
    parser.add_argument("--num-digital", type=int, default=25, help="Number of digital 2D board samples.")
    parser.add_argument("--num-negative", type=int, default=10, help="Number of 0-byte empty board negative samples.")
    parser.add_argument("--output-dir", type=str, default="data/standardized", help="Output directory.")

    args = parser.parse_args()
    generate_samples(
        num_physical=args.num_physical,
        num_digital=args.num_digital,
        num_negative=args.num_negative,
        base_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
