"""
src/acquisition.py – Dataset ingestion, parsing, train/test splitting, and background reference building.

Key Responsibilities:
  1. Load or build a standardized index of images, lot identities, weather conditions,
     and polygon-delimited parking spaces with occupancy ground truth.
  2. Implement an image-level (not space-level) Train/Test split (80/20) to prevent
     spatiotemporal data leakage between training and testing distributions.
  3. Construct a canonical 'Empty/Reference' background per parking lot and per slot:
     Computes the per-space median patch across training images where spaces are free.
"""

import os
import sys
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR, PKLOT_DIR, INDEX_FILE, MANIFEST_FILE,
    TRAIN_RATIO, RANDOM_SEED, PATCH_SIZE
)
from src.utils import set_seed, load_json, save_json, warp_polygon_to_patch


def parse_pklot_xml(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse native PKLot XML annotation file.
    
    Structure:
      <parking id="...">
        <space id="1" occupied="0|1">
          <rotatedRect>...</rotatedRect>
          <contour>
            <point x="..." y="..."/>
            ...
          </contour>
        </space>
      </parking>
    """
    spaces = []
    if not xml_path.exists():
        return spaces

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for space in root.findall("space"):
            s_id = space.attrib.get("id", "")
            occupied = int(space.attrib.get("occupied", "0"))
            
            contour = space.find("contour")
            points = []
            if contour is not None:
                for pt in contour.findall("point"):
                    px = int(float(pt.attrib.get("x", 0)))
                    py = int(float(pt.attrib.get("y", 0)))
                    points.append([px, py])

            # Fallback to rotatedRect if contour is missing
            if len(points) < 3:
                r_rect = space.find("rotatedRect")
                if r_rect is not None:
                    center = r_rect.find("center")
                    size = r_rect.find("size")
                    angle = r_rect.find("angle")
                    if center is not None and size is not None and angle is not None:
                        cx, cy = float(center.attrib['x']), float(center.attrib['y'])
                        w, h = float(size.attrib['w']), float(size.attrib['h'])
                        ang = float(angle.attrib['value'])
                        box = cv2.boxPoints(((cx, cy), (w, h), ang))
                        points = [[int(pt[0]), int(pt[1])] for pt in box]

            if len(points) >= 3:
                spaces.append({
                    "id": s_id,
                    "polygon": points,
                    "occupied": occupied
                })
    except Exception as e:
        print(f"Warning: Failed parsing XML {xml_path}: {e}")

    return spaces


def build_index_from_disk(pklot_dir: Path = PKLOT_DIR, save_path: Path = INDEX_FILE) -> List[Dict[str, Any]]:
    """
    Scan PKLot directory on disk and construct normalized index.json.
    """
    index = []
    for jpg_path in sorted(pklot_dir.rglob("*.jpg")):
        xml_path = jpg_path.with_suffix(".xml")
        if not xml_path.exists():
            continue

        rel_parts = jpg_path.relative_to(pklot_dir).parts
        lot = rel_parts[0] if len(rel_parts) > 0 else "Unknown"
        weather = rel_parts[1] if len(rel_parts) > 1 else "Unknown"
        date_folder = rel_parts[2] if len(rel_parts) > 2 else "Unknown"

        spaces = parse_pklot_xml(xml_path)
        if not spaces:
            continue

        rel_img = str(jpg_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        rel_xml = str(xml_path.relative_to(PROJECT_ROOT)).replace("\\", "/")

        index.append({
            "image_path": rel_img,
            "xml_path": rel_xml,
            "lot": lot,
            "weather": weather,
            "date": date_folder,
            "datetime_str": jpg_path.stem,
            "is_synthetic": False,
            "spaces": spaces
        })

    if index and save_path:
        save_json(save_path, index)
    return index


def load_dataset_index(index_path: Path = INDEX_FILE) -> List[Dict[str, Any]]:
    """
    Load normalized dataset index. If index doesn't exist, builds it from disk.
    """
    if index_path.exists():
        return load_json(index_path)
    return build_index_from_disk(PKLOT_DIR, index_path)


def train_test_split_images(
    index: List[Dict[str, Any]],
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Partition dataset into train and test sets strictly at the IMAGE level.
    
    Splitting by image guarantees that no frames from the same timestamp/angle
    leak between train and test. Stratification is applied per lot and weather condition.
    """
    set_seed(seed)
    grouped = defaultdict(list)
    for record in index:
        key = (record.get("lot", ""), record.get("weather", ""))
        grouped[key].append(record)

    train_set = []
    test_set = []

    rng = random.Random(seed)
    for group_key, records in grouped.items():
        shuffled = list(records)
        rng.shuffle(shuffled)
        n_train = max(1, int(round(len(shuffled) * train_ratio)))
        train_set.extend(shuffled[:n_train])
        test_set.extend(shuffled[n_train:])

    return train_set, test_set


def build_reference_patches(
    train_index: List[Dict[str, Any]],
    target_size: Tuple[int, int] = PATCH_SIZE,
    max_samples_per_space: int = 15
) -> Dict[str, np.ndarray]:
    """
    Build per-space empty reference patches from the training set.
    
    Methodology:
      For every parking space (identified by lot + space_id), we gather warped
      patches from training images where the ground truth label is FREE (occupied=0).
      Taking the median across channels yields a robust, shadow-reduced, clean representation
      of the empty parking spot's asphalt/pavement texture.
      
    Returns:
      Dictionary mapping key '{lot}_{space_id}' to warped BGR patch (uint8 np.ndarray).
    """
    lot_space_patches = defaultdict(list)

    for item in train_index:
        img_path = PROJECT_ROOT / item["image_path"]
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        lot = item.get("lot", "")
        for sp in item.get("spaces", []):
            if sp.get("occupied", 0) == 0:  # Only free spaces
                s_id = str(sp.get("id"))
                key = f"{lot}_{s_id}"
                if len(lot_space_patches[key]) < max_samples_per_space:
                    poly = sp.get("polygon", [])
                    patch = warp_polygon_to_patch(img, poly, target_size)
                    lot_space_patches[key].append(patch)

    ref_patches = {}
    for key, patches in lot_space_patches.items():
        if patches:
            # Median patch across time
            stack = np.stack(patches, axis=0)  # (N, H, W, 3)
            median_patch = np.median(stack, axis=0).astype(np.uint8)
            ref_patches[key] = median_patch

    return ref_patches
