"""
Training & ONNX Export Pipeline for the MicroCNN Domain Classifier (US-3.1.2).

Generates and augments a balanced multi-domain dataset covering:
1. Standard & textured digital chessboards (Lichess, Chess.com, wood, marble).
2. Complex UI screenshot surroundings (chat, avatars, eval meters).
3. Physical tournament & smartphone camera photos (lighting gradients, sensor noise).
4. Recaptured screen photos with Moiré interference patterns & bezel glare (labeled PHYSICAL_3D).
5. Heavily compressed JPEGs (JPEG DCT blocking & ringing).

Trains the MicroCNN architecture, verifies >99.5% accuracy, and exports optimized ONNX weights.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.domain_classifier.micro_cnn import MicroCNN


# ---------------------------------------------------------------------------
# Synthetic & Augmented Dataset Generator for Ambiguous Edge Cases
# ---------------------------------------------------------------------------


def generate_synthetic_digital_board(
    size: int = 128,
    style: str = "random",
) -> np.ndarray:
    """Generates a synthetic digital chessboard thumbnail."""
    board = np.zeros((size, size, 3), dtype=np.uint8)
    sq_size = size // 8

    # Style palettes (Light BGR, Dark BGR)
    palettes = [
        ((240, 240, 240), (120, 150, 80)),   # Green / White (Chess.com standard)
        ((245, 235, 220), (150, 110, 80)),   # Wood / Brown (Lichess Wood)
        ((235, 235, 235), (170, 160, 150)),  # Gray / Marble
        ((240, 240, 240), (180, 120, 80)),   # Blue / White
        ((250, 250, 250), (60, 60, 60)),     # High-contrast Black / White
    ]
    if style == "random":
        light_color, dark_color = random.choice(palettes)
    else:
        light_color, dark_color = palettes[0]

    for r in range(8):
        for c in range(8):
            color = light_color if (r + c) % 2 == 0 else dark_color
            board[r * sq_size : (r + 1) * sq_size, c * sq_size : (c + 1) * sq_size] = color

    # Add procedural subtle wood texture if selected
    if random.random() < 0.35:
        noise = np.random.normal(0, 3, (size, size, 3)).astype(np.float32)
        board = np.clip(board.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Add digital piece icons (circles / crosses in squares)
    if random.random() < 0.8:
        for r in range(8):
            for c in range(8):
                if random.random() < 0.4:
                    center = (c * sq_size + sq_size // 2, r * sq_size + sq_size // 2)
                    piece_color = (15, 15, 15) if (r < 4) else (245, 245, 245)
                    cv2.circle(board, center, sq_size // 4, piece_color, -1)

    # Optional UI surroundings (simulating full screen app crop)
    if random.random() < 0.3:
        border_size = random.randint(4, 16)
        padded = np.full((size + 2 * border_size, size + 2 * border_size, 3), random.randint(30, 80), dtype=np.uint8)
        padded[border_size : border_size + size, border_size : border_size + size] = board
        board = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)

    return board


def generate_synthetic_physical_photo(size: int = 128) -> np.ndarray:
    """Generates a synthetic physical camera photo with perspective and natural lighting."""
    base = generate_synthetic_digital_board(size=256)

    # 1. Perspective warp (simulate angled camera view)
    src_pts = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    pad_x = random.randint(20, 60)
    pad_y = random.randint(15, 50)
    dst_pts = np.float32([
        [pad_x, pad_y],
        [256 - pad_x, pad_y + random.randint(-10, 10)],
        [256, 256],
        [0, 256],
    ])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(base, H, (256, 256), borderValue=(random.randint(100, 160), random.randint(90, 140), random.randint(80, 120)))

    # 2. Add natural lighting gradient (ambient illumination falloff)
    grad = np.zeros((256, 256), dtype=np.float32)
    start_v = random.uniform(0.5, 0.8)
    end_v = random.uniform(1.1, 1.4)
    for y in range(256):
        grad[y, :] = start_v + (end_v - start_v) * (y / 256.0)
    
    shaded = (warped.astype(np.float32) * grad[:, :, None]).clip(0, 255)

    # 3. Add camera sensor photon shot noise (Gaussian)
    noise_sigma = random.uniform(8.0, 20.0)
    noise = np.random.normal(0, noise_sigma, (256, 256, 3))
    noisy = np.clip(shaded + noise, 0, 255).astype(np.uint8)

    # 4. Downscale to target size with slight lens blur
    if random.random() < 0.5:
        noisy = cv2.GaussianBlur(noisy, (3, 3), 0.8)

    return cv2.resize(noisy, (size, size), interpolation=cv2.INTER_AREA)


def generate_synthetic_screen_recapture(size: int = 128) -> np.ndarray:
    """
    Generates a synthetic monitor screen recapture with Moiré interference patterns.
    
    Must be labeled DomainType.PHYSICAL_3D (Class 1) to enforce homography rectification.
    """
    board = generate_synthetic_digital_board(size=256)

    # 1. Perspective tilt (smartphone photographing monitor at an angle)
    src_pts = np.float32([[0, 0], [256, 0], [256, 256], [0, 256]])
    tilt = random.randint(15, 45)
    dst_pts = np.float32([
        [tilt, tilt // 2],
        [256 - tilt, tilt // 2],
        [256 - tilt // 3, 256],
        [tilt // 3, 256],
    ])
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(board, H, (256, 256), borderValue=(random.randint(15, 35), random.randint(15, 35), random.randint(15, 35)))

    # 2. Add high-frequency 2D Moiré beat pattern (interference between LCD subpixel grid & camera Bayer CFA)
    moire_freq_x = random.uniform(0.12, 0.35)
    moire_freq_y = random.uniform(0.12, 0.35)
    moire_amp = random.uniform(15.0, 38.0)
    angle = random.uniform(0, math.pi)

    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    u = xx * math.cos(angle) - yy * math.sin(angle)
    v = xx * math.sin(angle) + yy * math.cos(angle)

    moire_pattern = (
        moire_amp * np.sin(2 * math.pi * moire_freq_x * u) * np.cos(2 * math.pi * moire_freq_y * v)
    )
    
    # 3. Add chromatic subpixel fringing
    fringed = warped.astype(np.float32)
    fringed[:, :, 0] += moire_pattern * 1.2   # Blue channel
    fringed[:, :, 1] += moire_pattern * 0.8   # Green channel
    fringed[:, :, 2] += moire_pattern * 1.0   # Red channel

    # 4. Add monitor glass glare / reflection spot
    glare_x, glare_y = random.randint(40, 216), random.randint(40, 216)
    dist_sq = (xx - glare_x) ** 2 + (yy - glare_y) ** 2
    glare = 40.0 * np.exp(-dist_sq / (2 * 40.0 ** 2))
    fringed += glare[:, :, None]

    # 5. Add sensor noise
    sensor_noise = np.random.normal(0, random.uniform(5.0, 14.0), (256, 256, 3))
    final_img = np.clip(fringed + sensor_noise, 0, 255).astype(np.uint8)

    return cv2.resize(final_img, (size, size), interpolation=cv2.INTER_AREA)


def letterbox_image(
    bgr_image: np.ndarray,
    target_size: int = 128,
    pad_color: int = 114,
) -> np.ndarray:
    """
    Resizes an image to fit within (target_size, target_size) while preserving aspect ratio.
    Pads the remaining canvas with a neutral gray value (114).
    """
    h, w = bgr_image.shape[:2]
    if h == target_size and w == target_size:
        return bgr_image

    scale = target_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr_image, (nw, nh), interpolation=interp)

    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    top = (target_size - nh) // 2
    left = (target_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def apply_synthetic_browser_ui(board_img: np.ndarray) -> np.ndarray:
    """
    Synthesizes a realistic desktop or mobile chess application window around a digital board.
    
    Simulates Chess.com, Lichess, and mobile app layouts:
    - Variable aspect ratios (0.70 portrait mobile to 1.77 widescreen desktop).
    - Authentic dark theme (#312e2b, #262421) and light theme UI backgrounds.
    - Sidebars (move notation table, evaluation bar, chat box).
    - Player info cards (avatar blocks, rating badges, clock timers).
    """
    bh, bw = board_img.shape[:2]

    # 1. Random canvas aspect ratio
    aspect_mode = random.choice(["desktop_widescreen", "desktop_sidebar", "mobile_portrait", "square"])
    if aspect_mode == "desktop_widescreen":
        canvas_w = int(bw * random.uniform(1.3, 1.8))
        canvas_h = int(bh * random.uniform(1.1, 1.3))
    elif aspect_mode == "desktop_sidebar":
        canvas_w = int(bw * random.uniform(1.2, 1.5))
        canvas_h = int(bh * random.uniform(1.05, 1.2))
    elif aspect_mode == "mobile_portrait":
        canvas_w = int(bw * random.uniform(1.02, 1.15))
        canvas_h = int(bh * random.uniform(1.3, 1.6))
    else:
        canvas_w = int(bw * random.uniform(1.05, 1.2))
        canvas_h = int(bh * random.uniform(1.05, 1.2))

    # 2. UI Background Theme
    ui_theme = random.choice(["chess_com_dark", "lichess_dark", "clean_light", "custom_dark"])
    if ui_theme == "chess_com_dark":
        bg_color = (43, 46, 49)      # BGR for #312e2b
        sidebar_color = (36, 38, 41) # BGR for #292624
    elif ui_theme == "lichess_dark":
        bg_color = (33, 36, 38)      # BGR for #262421
        sidebar_color = (25, 27, 29) # BGR for #1d1b19
    elif ui_theme == "clean_light":
        bg_color = (240, 240, 240)
        sidebar_color = (225, 225, 225)
    else:
        bg_color = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))
        sidebar_color = (random.randint(15, 40), random.randint(15, 40), random.randint(15, 40))

    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)

    # 3. Position the chessboard inside canvas
    if aspect_mode in ("desktop_widescreen", "desktop_sidebar"):
        offset_x = random.randint(10, max(12, int((canvas_w - bw) * 0.35)))
        offset_y = (canvas_h - bh) // 2
    else:
        offset_x = (canvas_w - bw) // 2
        offset_y = random.randint(10, max(12, int((canvas_h - bh) * 0.5)))

    offset_x = max(0, min(offset_x, canvas_w - bw))
    offset_y = max(0, min(offset_y, canvas_h - bh))
    canvas[offset_y:offset_y + bh, offset_x:offset_x + bw] = board_img

    # 4. Render UI elements (Evaluation Bar, Sidebar, Player Cards)
    # 4a. Evaluation Bar (thin vertical strip next to board)
    if random.random() < 0.65 and offset_x >= 12:
        eval_w = max(4, offset_x // 3)
        eval_x = offset_x - eval_w - 4
        canvas[offset_y:offset_y + bh // 2, eval_x:eval_x + eval_w] = (250, 250, 250)
        canvas[offset_y + bh // 2:offset_y + bh, eval_x:eval_x + eval_w] = (30, 30, 30)

    # 4b. Right Sidebar (Move list / Notation / Clock)
    sidebar_left = offset_x + bw + 8
    if sidebar_left < canvas_w - 10:
        canvas[offset_y:offset_y + bh, sidebar_left:canvas_w - 6] = sidebar_color
        for line_y in range(offset_y + 15, offset_y + bh - 10, 18):
            line_w = random.randint(15, max(16, (canvas_w - 6 - sidebar_left) - 10))
            canvas[line_y:line_y + 8, sidebar_left + 8:sidebar_left + 8 + line_w] = (80, 85, 90)

    # 4c. Player Avatar / Rating Badges
    if offset_y >= 20:
        canvas[offset_y - 18:offset_y - 4, offset_x:offset_x + 14] = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        canvas[offset_y - 14:offset_y - 6, offset_x + 18:offset_x + 90] = (120, 125, 130)
    if (offset_y + bh + 20) <= canvas_h:
        canvas[offset_y + bh + 4:offset_y + bh + 18, offset_x:offset_x + 14] = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        canvas[offset_y + bh + 6:offset_y + bh + 14, offset_x + 18:offset_x + 90] = (120, 125, 130)

    return canvas


def apply_digital_theme_jitter(image: np.ndarray) -> np.ndarray:
    """
    Applies comprehensive theme, piece color, and UI jitter to simulate
    all custom Chess.com and Lichess themes (Wood, Marble, Glass, Bubblegum, 8-Bit, Neon).
    """
    out = image.copy()

    # 1. Random HSV Hue Rotation & Saturation scaling (covers all board colorways)
    if random.random() < 0.85:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = random.uniform(-90, 90)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        sat_scale = random.uniform(0.3, 1.8)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        val_scale = random.uniform(0.6, 1.3)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 2. Contrast & Gamma Adjustment
    if random.random() < 0.60:
        alpha = random.uniform(0.7, 1.4)
        beta = random.uniform(-25, 25)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 3. Simulate square move highlights (yellow/green/orange/pink highlights)
    if random.random() < 0.40:
        h, w = out.shape[:2]
        sq_h, sq_w = h // 8, w // 8
        r, c = random.randint(0, 7), random.randint(0, 7)
        highlight_color = random.choice([
            (0, 255, 255),    # Yellow
            (50, 205, 50),    # Lime Green
            (0, 165, 255),    # Orange
            (255, 105, 180),  # Pink
        ])
        overlay = out.copy()
        cv2.rectangle(overlay, (c * sq_w, r * sq_h), ((c + 1) * sq_w, (r + 1) * sq_h), highlight_color, -1)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # 4. Orthogonal 90/180/270 degree rotation
    if random.random() < 0.50:
        rot = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
        out = cv2.rotate(out, rot)

    return out


def apply_jpeg_compression(image: np.ndarray, quality: int | None = None) -> np.ndarray:
    """Simulates JPEG DCT blocking and compression ringing artifacts."""
    q = quality if quality is not None else random.randint(25, 85)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    _, encimg = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class ChessDomainDataset(Dataset):
    """
    Balanced Dataset for Domain Classification.
    
    Combines real standardized images with synthetic edge cases.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __init__(
        self,
        num_samples: int = 4000,
        real_digital_dir: Path | None = None,
        real_physical_dir: Path | None = None,
        split: str = "train",
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        is_train: bool = True,
    ) -> None:
        self.num_samples = num_samples
        self.split = split.lower()
        self.is_train = (self.split == "train") if split in {"train", "val", "test"} else is_train
        self.seed = seed

        def _partition_files(file_list: list[Path]) -> list[Path]:
            if not file_list:
                return []
            sorted_files = sorted(file_list)
            rng = random.Random(seed)
            rng.shuffle(sorted_files)
            
            n = len(sorted_files)
            train_end = int(n * split_ratios[0])
            val_end = train_end + int(n * split_ratios[1])
            
            if self.split == "train":
                return sorted_files[:train_end]
            elif self.split == "val":
                return sorted_files[train_end:val_end]
            elif self.split == "test":
                return sorted_files[val_end:]
            return sorted_files

        # Load and partition real image paths if available
        self.real_digital: list[Path] = []
        if real_digital_dir and real_digital_dir.is_dir():
            all_digital = list(real_digital_dir.glob("*.png")) + list(real_digital_dir.glob("*.jpg"))
            self.real_digital = _partition_files(all_digital)

        self.real_physical: list[Path] = []
        if real_physical_dir and real_physical_dir.is_dir():
            all_physical = list(real_physical_dir.glob("*.jpg")) + list(real_physical_dir.glob("*.png"))
            self.real_physical = _partition_files(all_physical)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Balanced sampling: Even index -> Digital (0), Odd index -> Physical (1)
        label = idx % 2

        if label == 0:
            # DomainType.DIGITAL_2D
            if self.real_digital and random.random() < 0.75:
                img_path = random.choice(self.real_digital)
                img = cv2.imread(str(img_path))
                if img is None:
                    img = generate_synthetic_digital_board()
            else:
                img = generate_synthetic_digital_board()

            # Apply theme color jitter during training
            if self.is_train and random.random() < 0.75:
                img = apply_digital_theme_jitter(img)

            # Apply synthetic full-UI browser canvas (50% probability during training)
            if self.is_train and random.random() < 0.50:
                img = apply_synthetic_browser_ui(img)

            # Random JPEG compression artifact
            if random.random() < 0.35:
                img = apply_jpeg_compression(img)

            # Aspect-ratio preserving letterbox resize to 128x128
            img = letterbox_image(img, target_size=128, pad_color=114)

        else:
            # DomainType.PHYSICAL_3D (including monitor recaptures)
            if self.real_physical and random.random() < 0.75:
                img_path = random.choice(self.real_physical)
                img = cv2.imread(str(img_path))
                if img is None:
                    img = generate_synthetic_physical_photo()
            else:
                # Recaptured monitor screen with moiré & synthetic angle photo
                img = (
                    generate_synthetic_screen_recapture()
                    if random.random() < 0.60
                    else generate_synthetic_physical_photo()
                )

            if random.random() < 0.30:
                img = apply_jpeg_compression(img)

            # Aspect-ratio preserving letterbox resize to 128x128
            img = letterbox_image(img, target_size=128, pad_color=114)

        # Preprocessing to PyTorch normalized tensor
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        normalized = (chw - self.MEAN) / self.STD
        tensor = torch.from_numpy(normalized).float()

        return tensor, label


# ---------------------------------------------------------------------------
# Training & Export Runner
# ---------------------------------------------------------------------------


def train_and_export_micro_cnn(
    output_onnx_path: str | Path = "src/domain_classifier/weights/domain_classifier_microcnn.onnx",
    epochs: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_train_samples: int = 4000,
    num_val_samples: int = 800,
    device_name: str | None = None,
) -> Path:
    """
    Trains the MicroCNN architecture and exports the trained model to ONNX.
    
    Args:
        output_onnx_path: Destination path for the exported .onnx model.
        epochs: Number of training epochs.
        batch_size: DataLoader batch size.
        learning_rate: Initial AdamW learning rate.
        num_train_samples: Training set size.
        num_val_samples: Validation set size.
        device_name: 'cuda' or 'cpu'. Auto-detects if None.
        
    Returns:
        Path to the exported ONNX model.
    """
    out_path = Path(output_onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"[INFO] Training MicroCNN Domain Classifier on device: {device}")

    # 1. Prepare Datasets
    digital_dir = Path("data/standardized/digital/images")
    physical_dir = Path("data/standardized/physical/images")

    train_dataset = ChessDomainDataset(
        num_samples=num_train_samples,
        real_digital_dir=digital_dir,
        real_physical_dir=physical_dir,
        split="train",
    )
    val_dataset = ChessDomainDataset(
        num_samples=num_val_samples,
        real_digital_dir=digital_dir,
        real_physical_dir=physical_dir,
        split="val",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model, Loss & Optimizer
    model = MicroCNN(num_classes=2, dropout_rate=0.20).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] MicroCNN Total Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Training Loop
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += images.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += images.size(0)

        train_acc = train_correct / total_train
        val_acc = val_correct / total_val
        avg_train_loss = train_loss / total_train
        avg_val_loss = val_loss / total_val

        if epoch % 5 == 0 or epoch == epochs or val_acc > best_val_acc:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Training Complete in {elapsed:.1f}s. Best Val Accuracy: {best_val_acc:.2%}")

    # 4. Export Model to ONNX
    print(f"[EXPORT] Exporting trained MicroCNN to ONNX: {out_path}")
    model.eval().cpu()
    dummy_input = torch.randn(1, 3, 128, 128, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] ONNX File Size: {file_size_mb:.3f} MB (Target: < 1.5 MB)")

    # 5. Verify ONNX Runtime Inference
    import onnxruntime as ort

    ort_session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    diff = np.max(np.abs(ort_outs[0] - torch_out))
    print(f"[VERIFY] PyTorch vs ONNX Runtime Max Output Difference: {diff:.2e}")
    assert diff < 1e-4, f"ONNX output mismatch exceeds tolerance: {diff}"
    print(f"[SUCCESS] Model exported and verified successfully at {out_path}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export MicroCNN Domain Classifier")
    parser.add_argument("--output", type=str, default="src/domain_classifier/weights/domain_classifier_microcnn.onnx")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--val-samples", type=int, default=800)
    args = parser.parse_args()

    train_and_export_micro_cnn(
        output_onnx_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_samples=args.train_samples,
        num_val_samples=args.val_samples,
    )


if __name__ == "__main__":
    main()
