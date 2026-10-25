"""
main.py – Unified CLI Orchestrator for the ParkLens System.

Commands:
  download : Acquire PKLot real image & polygon subset (from HuggingFace)
  prepare  : Ingest index, split 80/20 train/test by image, build reference patches
  tune     : Hyperparameter grid-search on train split (maximize F1 per lot x weather)
  evaluate : Quantitative benchmark on test split (confusion matrix, ROC, tables)
  predict  : Annotate a single image (with optional XML contour file)
  demo     : Generate and run pipeline on synthetic sample data
  all      : Execute complete pipeline: download -> prepare -> tune -> evaluate

Usage:
  python main.py all
  python main.py predict --image data/PKLot/PUCPR/Sunny/2012-09-11/2012-09-11_15_16_58.jpg
"""

import sys
import os
import argparse
import pickle
from pathlib import Path
from typing import Optional

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR, PKLOT_DIR, INDEX_FILE, MANIFEST_FILE,
    OUTPUTS_DIR, PARAMS_FILE, RESULTS_FILE, PATCH_SIZE, RANDOM_SEED
)


def cmd_download(args):
    """Run data acquisition script."""
    from scripts.download_data import download_pklot_subset
    count = getattr(args, "count", 50)
    success = download_pklot_subset(target_per_group=count)
    if not success:
        sys.exit(1)


def cmd_prepare(args):
    """Prepare train/test splits and build empty background reference patches."""
    from src.acquisition import (
        load_dataset_index, train_test_split_images,
        build_reference_patches, build_index_from_disk
    )
    from src.utils import save_json

    if not INDEX_FILE.exists():
        if not PKLOT_DIR.exists() or len(list(PKLOT_DIR.rglob("*.jpg"))) == 0:
            print("[INFO] PKLot dataset missing. Running download first...")
            from scripts.download_data import download_pklot_subset
            download_pklot_subset(50)
        else:
            print("[INFO] Building index from disk...")
            build_index_from_disk(PKLOT_DIR, INDEX_FILE)

    print("\n[Stage 1/2] Loading dataset index and performing train/test split...")
    index = load_dataset_index(INDEX_FILE)
    train_set, test_set = train_test_split_images(index, train_ratio=0.80, seed=RANDOM_SEED)

    splits_file = DATA_DIR / "splits.json"
    save_json(splits_file, {
        "train": [t["image_path"] for t in train_set],
        "test": [t["image_path"] for t in test_set],
        "train_count": len(train_set),
        "test_count": len(test_set)
    })
    print(f"Partitioned {len(index)} images: {len(train_set)} train (80%), {len(test_set)} test (20%).")
    print(f"Splits saved to {splits_file}")

    print("\n[Stage 2/2] Synthesizing canonical empty reference patches per lot slot...")
    ref_patches = build_reference_patches(train_set, target_size=PATCH_SIZE)
    ref_path = DATA_DIR / "ref_patches.pkl"
    with open(ref_path, "wb") as f:
        pickle.dump(ref_patches, f)

    print(f"Computed {len(ref_patches)} median reference patches.")
    print(f"Reference models saved to {ref_path}")
    print("\n[SUCCESS] Preparation complete!")


def cmd_tune(args):
    """Tune parameters on train split."""
    from src.acquisition import load_dataset_index, train_test_split_images
    from src.classifier import tune_all_groups

    ref_path = DATA_DIR / "ref_patches.pkl"
    if not ref_path.exists():
        print("[INFO] Reference patches missing. Running prepare first...")
        cmd_prepare(args)

    with open(ref_path, "rb") as f:
        ref_patches = pickle.load(f)

    index = load_dataset_index(INDEX_FILE)
    train_set, _ = train_test_split_images(index, train_ratio=0.80, seed=RANDOM_SEED)

    tune_all_groups(train_set, ref_patches, save_path=PARAMS_FILE)
    print("\n[SUCCESS] Parameter tuning finished!")


def cmd_evaluate(args):
    """Run full test-set evaluation and save metrics and publication figures."""
    from src.acquisition import load_dataset_index, train_test_split_images
    from src.evaluation import evaluate_test_set, save_evaluation_results
    from src.visualization import save_all_evaluation_figures

    ref_path = DATA_DIR / "ref_patches.pkl"
    if not ref_path.exists():
        print("[INFO] Reference models missing. Running prepare first...")
        cmd_prepare(args)

    if not PARAMS_FILE.exists():
        print("[INFO] Tuned parameters missing. Running tune first...")
        cmd_tune(args)

    with open(ref_path, "rb") as f:
        ref_patches = pickle.load(f)

    index = load_dataset_index(INDEX_FILE)
    _, test_set = train_test_split_images(index, train_ratio=0.80, seed=RANDOM_SEED)

    report = evaluate_test_set(test_set, ref_patches, params_file=PARAMS_FILE)
    save_evaluation_results(report, OUTPUTS_DIR)
    save_all_evaluation_figures(report, OUTPUTS_DIR)

    # Print summary table
    print("\n" + "=" * 65)
    print("  PARKLENS TEST EVALUATION SUMMARY")
    print("=" * 65)
    o = report["overall"]
    print(f"  Overall Accuracy:    {o['accuracy']*100:.2f}%")
    print(f"  Sensitivity (Recall): {o['sensitivity']*100:.2f}%")
    print(f"  Specificity:         {o['specificity']*100:.2f}%")
    print(f"  Precision:           {o['precision']*100:.2f}%")
    print(f"  F1-Score:            {o['f1']*100:.2f}%")
    print(f"  ROC Area Under Curve:{o['auc']:.4f}")
    print(f"  Total Spaces Tested: {o['total']:,}")
    print("=" * 65)


def cmd_predict(args):
    """Predict occupancy on a single image and save the annotated output."""
    import cv2
    from src.acquisition import parse_pklot_xml, load_dataset_index
    from src.classifier import classify_spaces_in_image, load_tuned_params
    from src.visualization import draw_parking_overlay

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        sys.exit(1)

    # Resolve XML
    xml_path = Path(args.xml) if args.xml else img_path.with_suffix(".xml")
    spaces = []
    if xml_path.exists():
        spaces = parse_pklot_xml(xml_path)
    else:
        # Check if indexed
        idx = load_dataset_index(INDEX_FILE)
        rel_str = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        match = next((item for item in idx if item["image_path"] == rel_str), None)
        if match:
            spaces = match.get("spaces", [])

    if not spaces:
        print(f"[ERROR] No parking space contours found. Provide --xml path.xml")
        sys.exit(1)

    # Determine lot
    lot = "PUCPR"
    for l in ["PUCPR", "UFPR04", "UFPR05"]:
        if l.lower() in str(img_path).lower():
            lot = l
            break

    # Load references & params
    ref_patches = {}
    ref_path = DATA_DIR / "ref_patches.pkl"
    if ref_path.exists():
        with open(ref_path, "rb") as f:
            ref_patches = pickle.load(f)

    params = load_tuned_params(PARAMS_FILE, lot=lot)
    results = classify_spaces_in_image(img, spaces, lot, ref_patches, params)

    annotated = draw_parking_overlay(
        img, results,
        show_ids=args.show_ids,
        show_confidence=args.show_confidence,
        highlight_errors=args.highlight_errors
    )

    out_file = Path(args.output) if args.output else OUTPUTS_DIR / f"annotated_{img_path.stem}.jpg"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), annotated)

    free_n = sum(1 for r in results if r["prediction"] == 0)
    occ_n = len(results) - free_n
    print(f"\nPrediction Complete for {img_path.name}:")
    print(f"  - Total Spaces:    {len(results)}")
    print(f"  - Free Spaces:     {free_n}")
    print(f"  - Occupied Spaces: {occ_n}")
    print(f"  - Output Saved To: {out_file}")


def cmd_demo(args):
    """Run pipeline test on synthetic dataset."""
    from scripts.make_synthetic import generate_synthetic_dataset
    print("Generating synthetic demo lot data...")
    out_dir = DATA_DIR / "synthetic"
    generate_synthetic_dataset(out_dir, n_images=5)
    print("Demo dataset generated at data/synthetic/")


def cmd_all(args):
    """Execute complete end-to-end pipeline."""
    print("=" * 65)
    print("       PARKLENS COMPLETE PIPELINE EXECUTION")
    print("=" * 65)
    cmd_download(args)
    cmd_prepare(args)
    cmd_tune(args)
    cmd_evaluate(args)
    print("\n[ALL TASKS COMPLETED SUCCESSFULLY]")


def main():
    parser = argparse.ArgumentParser(
        description="ParkLens: Classical Computer Vision Parking Occupancy Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # download
    p_dl = subparsers.add_parser("download", help="Download PKLot dataset subset")
    p_dl.add_argument("--count", type=int, default=50, help="Images per group target")

    # prepare
    subparsers.add_parser("prepare", help="Prepare train/test splits & empty background references")

    # tune
    subparsers.add_parser("tune", help="Grid-search hyperparameter tuning on train split")

    # evaluate
    subparsers.add_parser("evaluate", help="Benchmark quantitative metrics on test split")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict parking occupancy on a single image")
    p_pred.add_argument("--image", required=True, help="Path to input image")
    p_pred.add_argument("--xml", default=None, help="Path to space annotation XML")
    p_pred.add_argument("--output", default=None, help="Path for output annotated image")
    p_pred.add_argument("--show-ids", action="store_true", help="Draw slot IDs")
    p_pred.add_argument("--show-confidence", action="store_true", help="Draw confidence %")
    p_pred.add_argument("--highlight-errors", action="store_true", help="Highlight FP/FN in yellow")

    # demo
    subparsers.add_parser("demo", help="Generate synthetic testing data")

    # all
    p_all = subparsers.add_parser("all", help="Execute download -> prepare -> tune -> evaluate")
    p_all.add_argument("--count", type=int, default=50, help="Images per group target")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "tune":
        cmd_tune(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

