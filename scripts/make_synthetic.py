"""
scripts/make_synthetic.py – Generate synthetic parking lot images for testing.

Creates fake parking lot images with rectangular "spaces" that are either
free (plain gray) or occupied (with colored rectangles simulating cars).
All synthetic images are marked as DEMO DATA.

Usage:
    python scripts/make_synthetic.py
"""

import sys
import json
import random
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

# ─── Setup paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, RANDOM_SEED, PATCH_SIZE


def generate_synthetic_lot(
    width: int = 1280,
    height: int = 720,
    rows: int = 2,
    cols: int = 8,
    occupancy_rate: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    """
    Generate a synthetic parking lot image with labeled spaces.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        rows: Number of rows of parking spaces.
        cols: Number of columns of parking spaces.
        occupancy_rate: Fraction of spaces that are occupied (0-1).
        seed: Random seed for reproducibility.

    Returns:
        (image, spaces) where spaces is a list of dicts with keys:
        id, polygon (list of [x,y] points), occupied (0 or 1)
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Create a parking lot background (gray asphalt with some noise)
    image = np.full((height, width, 3), (100, 100, 100), dtype=np.uint8)
    noise = np_rng.randint(-15, 15, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add lane markings
    cv2.line(image, (0, height // 2), (width, height // 2), (200, 200, 200), 3)

    # Calculate space dimensions
    margin_x, margin_y = 40, 60
    space_w = (width - 2 * margin_x) // cols
    space_h = (height - 2 * margin_y - 20) // rows  # -20 for lane

    spaces = []
    space_id = 1

    car_colors = [
        (180, 50, 50), (50, 50, 180), (50, 150, 50),
        (200, 200, 200), (30, 30, 30), (180, 180, 50),
        (50, 180, 180), (180, 50, 180),
    ]

    for row in range(rows):
        for col in range(cols):
            # Calculate space corners
            x1 = margin_x + col * space_w + 5
            y1 = margin_y + row * (space_h + 20) + 5
            x2 = x1 + space_w - 10
            y2 = y1 + space_h - 10

            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            # Draw space outline (white lines)
            pts = np.array(polygon, dtype=np.int32)
            cv2.polylines(image, [pts], True, (200, 200, 200), 2)

            # Decide if occupied
            occupied = 1 if rng.random() < occupancy_rate else 0

            if occupied:
                # Draw a "car" (colored rectangle with some detail)
                car_color = rng.choice(car_colors)
                car_margin = 8
                car_pts = np.array([
                    [x1 + car_margin, y1 + car_margin],
                    [x2 - car_margin, y1 + car_margin],
                    [x2 - car_margin, y2 - car_margin],
                    [x1 + car_margin, y2 - car_margin],
                ], dtype=np.int32)
                cv2.fillPoly(image, [car_pts], car_color)

                # Add "windows" (lighter rectangle on top portion)
                win_y1 = y1 + car_margin + 5
                win_y2 = y1 + car_margin + (y2 - y1) // 3
                cv2.rectangle(
                    image,
                    (x1 + car_margin + 5, win_y1),
                    (x2 - car_margin - 5, win_y2),
                    tuple(min(c + 60, 255) for c in car_color),
                    -1,
                )

                # Add "wheels" (dark circles)
                wheel_r = 4
                cv2.circle(image, (x1 + car_margin + 10, y2 - car_margin), wheel_r, (20, 20, 20), -1)
                cv2.circle(image, (x2 - car_margin - 10, y2 - car_margin), wheel_r, (20, 20, 20), -1)

            spaces.append({
                "id": str(space_id),
                "polygon": polygon,
                "occupied": occupied,
            })
            space_id += 1

    # Add "DEMO DATA" watermark
    cv2.putText(
        image, "DEMO DATA - SYNTHETIC",
        (width // 2 - 200, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2,
    )

    return image, spaces


def generate_synthetic_dataset(
    output_dir: Path,
    n_images: int = 10,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Generate a small synthetic dataset for testing.

    Args:
        output_dir: Directory to save images and a JSON index.
        n_images: Number of images to generate.
        seed: Base random seed.

    Returns:
        List of image record dicts in normalized format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for i in range(n_images):
        img_seed = seed + i
        occ_rate = 0.3 + (i / n_images) * 0.5  # vary occupancy

        image, spaces = generate_synthetic_lot(
            occupancy_rate=occ_rate, seed=img_seed
        )

        # Save image
        img_name = f"synthetic_{i:03d}.jpg"
        img_path = output_dir / img_name
        cv2.imwrite(str(img_path), image)

        # Build record
        record = {
            "image_path": str(img_path),
            "lot": "SYNTHETIC",
            "weather": "Synthetic",
            "datetime_str": datetime.now().isoformat(),
            "is_synthetic": True,
            "spaces": [
                {
                    "id": s["id"],
                    "polygon": s["polygon"],
                    "occupied": s["occupied"],
                }
                for s in spaces
            ],
        }
        records.append(record)

    # Save index
    index_path = output_dir / "synthetic_index.json"
    with open(index_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {n_images} synthetic images at {output_dir}")
    print(f"Index saved to {index_path}")
    return records


if __name__ == "__main__":
    output_dir = DATA_DIR / "synthetic"
    records = generate_synthetic_dataset(output_dir)

    # Show summary
    total_spaces = sum(len(r["spaces"]) for r in records)
    occupied = sum(
        sum(1 for s in r["spaces"] if s["occupied"]) for r in records
    )
    print(f"\nSummary: {total_spaces} total spaces, "
          f"{occupied} occupied ({occupied/total_spaces*100:.0f}%), "
          f"{total_spaces - occupied} free")

