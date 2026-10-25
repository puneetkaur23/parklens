"""
scripts/download_data.py – High-speed multi-threaded download for PKLot subset.

Acquires real PKLot images and polygon annotations from Hugging Face (Voxel51/PKLot)
which contains full lot images and exact parking space polygon annotations for all 3 lots
(PUCPR, UFPR04, UFPR05) across all 3 weather conditions (Sunny, Cloudy, Rainy).

Outputs:
  - Images and XMLs saved to data/PKLot/<Lot>/<Weather>/<Date>/
  - data/MANIFEST.json with source info, date, and group counts
  - data/index.json with normalized records ready for the ParkLens pipeline

Usage:
    python scripts/download_data.py [count_per_group]
"""

import sys
import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PKLOT_DIR, DATA_DIR, MANIFEST_FILE, INDEX_FILE,
    IMAGES_PER_GROUP, LOTS, WEATHERS, RANDOM_SEED
)


def get_disk_free_gb(path: Path) -> float:
    """Return free disk space in GB."""
    total, used, free = shutil.disk_usage(path.anchor if hasattr(path, 'anchor') else ".")
    return free / (1024 ** 3)


def extract_polygon_points(poly: dict, img_w: int, img_h: int) -> List[List[int]]:
    """Safely extract integer polygon points handling all nesting variations."""
    raw_pts = poly.get("points", [])
    if not raw_pts:
        return []

    # Handle nesting: [[[x, y], ...]] or [[x, y], ...]
    pts_list = raw_pts
    if isinstance(pts_list[0], list):
        if not pts_list[0]:
            return []
        if isinstance(pts_list[0][0], list):
            pts_list = pts_list[0]

    coords = []
    for pt in pts_list:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            px = int(round(pt[0] * img_w))
            py = int(round(pt[1] * img_h))
            coords.append([px, py])
    return coords


def make_xml_from_spaces(lot_id: str, spaces_data: List[Dict[str, Any]]) -> str:
    """
    Construct original PKLot-style XML string containing <contour> points.
    """
    root = ET.Element("parking", id=lot_id.lower())
    for sp in spaces_data:
        s_id = str(sp.get("id", ""))
        is_occ = str(sp.get("occupied", 0))
        space_el = ET.SubElement(root, "space", id=s_id, occupied=is_occ)
        contour_el = ET.SubElement(space_el, "contour")
        for pt in sp.get("polygon", []):
            ET.SubElement(contour_el, "point", x=str(pt[0]), y=str(pt[1]))

    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def _download_and_process_sample(args):
    """Worker task for multi-threaded downloading and XML writing."""
    lot, weather, s, pklot_dir, project_root = args
    from huggingface_hub import hf_hub_download

    hf_filepath = s.get("filepath")
    filename = Path(hf_filepath).name
    date_folder = filename.split("_")[0] if "_" in filename else "2013-01-01"

    target_dir = pklot_dir / lot / weather / date_folder
    target_dir.mkdir(parents=True, exist_ok=True)

    target_jpg = target_dir / filename
    target_xml = target_dir / (Path(filename).stem + ".xml")

    # Download image if not cached
    if not target_jpg.exists():
        try:
            cached_file = hf_hub_download(
                repo_id="Voxel51/PKLot",
                filename=hf_filepath,
                repo_type="dataset"
            )
            shutil.copy2(cached_file, target_jpg)
        except Exception as e:
            return None

    # Parse metadata and polylines
    meta = s.get("metadata", {})
    img_w = meta.get("width", 1280)
    img_h = meta.get("height", 720)

    polylines = s.get("parking_spaces", {}).get("polylines", [])
    
    # Parse spaces into normalized index format
    spaces_data = []
    for p in polylines:
        occ_str = str(p.get("occupancy_status", "")).lower()
        occupied = 1 if "occupied" in occ_str and "not" not in occ_str else 0
        s_id = str(p.get("space_id", ""))
        polygon_pts = extract_polygon_points(p, img_w, img_h)

        if len(polygon_pts) >= 3:
            spaces_data.append({
                "id": s_id,
                "polygon": polygon_pts,
                "occupied": int(occupied)
            })

    # Save XML representation
    xml_content = make_xml_from_spaces(lot, spaces_data)
    with open(target_xml, "w", encoding="utf-8") as xf:
        xf.write(xml_content)

    return {
        "image_path": str(target_jpg.relative_to(project_root)).replace("\\", "/"),
        "xml_path": str(target_xml.relative_to(project_root)).replace("\\", "/"),
        "lot": lot,
        "weather": weather,
        "date": date_folder,
        "datetime_str": Path(filename).stem,
        "is_synthetic": False,
        "width": img_w,
        "height": img_h,
        "spaces": spaces_data
    }


def download_pklot_subset(target_per_group: int = 50, max_workers: int = 12) -> bool:
    """
    Acquire PKLot dataset subset using concurrent multi-threaded worker pool.
    """
    print("=" * 65)
    print("  ParkLens Data Acquisition: PKLot Dataset")
    print(f"  Target: {target_per_group} images per lot x weather (9 groups, ~{target_per_group * 9} total)")
    print(f"  Worker threads: {max_workers}")
    print("=" * 65)
    
    free_gb = get_disk_free_gb(DATA_DIR)
    print(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < 1.0:
        print("[ERROR] Free disk space is less than 1.0 GB!")
        return False

    try:
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
    except ImportError:
        print("[ERROR] Required packages huggingface_hub or tqdm missing.")
        return False

    print("\n[Step 1/3] Loading dataset metadata from Voxel51/PKLot...")
    samples_path = hf_hub_download("Voxel51/PKLot", "samples.json", repo_type="dataset")
    with open(samples_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    all_samples = meta_data.get("samples", [])
    print(f"Total available PKLot images in repository: {len(all_samples)}")

    lot_map = {"pucpr": "PUCPR", "ufpr04": "UFPR04", "ufpr05": "UFPR05"}
    weather_map = {"sunny": "Sunny", "cloudy": "Cloudy", "rainy": "Rainy"}

    grouped: Dict[str, List[Any]] = defaultdict(list)
    for s in all_samples:
        src = s.get("source", "").lower()
        wth = s.get("weather", {}).get("label", "").lower()
        lot = lot_map.get(src)
        weather = weather_map.get(wth)
        if lot and weather:
            grouped[f"{lot}/{weather}"].append(s)

    rng = random.Random(RANDOM_SEED)
    selected_tasks = []
    group_counts = {}

    for lot in LOTS:
        for weather in WEATHERS:
            key = f"{lot}/{weather}"
            candidates = grouped.get(key, [])
            if not candidates:
                continue

            by_date = defaultdict(list)
            for c in candidates:
                fp = c.get("filepath", "")
                fname = Path(fp).name
                date_part = fname.split("_")[0] if "_" in fname else "unknown"
                by_date[date_part].append(c)

            group_sel = []
            dates = list(by_date.keys())
            rng.shuffle(dates)
            
            idx = 0
            while len(group_sel) < target_per_group and any(by_date.values()):
                date_key = dates[idx % len(dates)]
                if by_date[date_key]:
                    chosen = rng.choice(by_date[date_key])
                    by_date[date_key].remove(chosen)
                    group_sel.append((lot, weather, chosen, PKLOT_DIR, PROJECT_ROOT))
                idx += 1
                if all(len(v) == 0 for v in by_date.values()):
                    break

            selected_tasks.extend(group_sel)
            group_counts[key] = len(group_sel)

    print(f"\n[Step 2/3] Processing {len(selected_tasks)} files (skipping already downloaded)...")
    PKLOT_DIR.mkdir(parents=True, exist_ok=True)

    normalized_index = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_and_process_sample, task): task for task in selected_tasks}
        pbar = tqdm(total=len(futures), desc="Acquiring PKLot Images", unit="img")
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                normalized_index.append(res)
            pbar.update(1)
        pbar.close()

    print(f"\n[Step 3/3] Writing metadata index and manifest...")
    manifest = {
        "source": "Hugging Face Hub (Voxel51/PKLot)",
        "date_downloaded": datetime.now().isoformat(),
        "total_images": len(normalized_index),
        "counts_per_lot_weather": group_counts,
        "citation": (
            "Almeida, P. R., Oliveira, L. S., Britto, A. S., Silva, E. J., & "
            "Koerich, A. L. (2015). PKLot – A robust dataset for parking lot "
            "classification. Expert Systems with Applications, 42(11), 4937-4949."
        )
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized_index, f, indent=2)

    print(f"\n[SUCCESS] Acquired {len(normalized_index)} PKLot images with full polygon ground-truth labels!")
    print(f"Manifest written to: {MANIFEST_FILE}")
    print(f"Index written to: {INDEX_FILE}")
    return True


if __name__ == "__main__":
    count = IMAGES_PER_GROUP if len(sys.argv) < 2 else int(sys.argv[1])
    success = download_pklot_subset(target_per_group=count)
    if not success:
        sys.exit(1)
