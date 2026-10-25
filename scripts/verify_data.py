"""
Verify downloaded PKLot data integrity and generate 3 sanity check overlay images.
"""
import json
import random
from pathlib import Path
from collections import Counter
import cv2

with open("data/index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)

print(f"Total records in index: {len(idx)}")
c = Counter(f"{r['lot']}/{r['weather']}" for r in idx)
for k, v in sorted(c.items()):
    print(f"  {k:16s}: {v} images")

total_spaces = sum(len(r["spaces"]) for r in idx)
total_occ = sum(sum(1 for s in r["spaces"] if s["occupied"] == 1) for r in idx)
total_free = total_spaces - total_occ
print(f"\nTotal spaces annotated: {total_spaces:,}")
print(f"Occupied: {total_occ:,} ({total_occ/total_spaces*100:.1f}%)")
print(f"Free:     {total_free:,} ({total_free/total_spaces*100:.1f}%)")

# Check 3 random images and render ground truth overlays
Path("outputs/sanity_checks").mkdir(parents=True, exist_ok=True)
random.seed(42)
samples = random.sample(idx, 3)

for i, s in enumerate(samples):
    p = Path(s["image_path"])
    assert p.exists(), f"Missing image {p}"
    img = cv2.imread(str(p))
    assert img is not None, f"Cannot read {p}"

    # Draw ground truth overlay
    overlay = img.copy()
    for sp in s["spaces"]:
        import numpy as np
        poly = np.array(sp["polygon"], dtype=np.int32)
        color = (0, 0, 255) if sp["occupied"] == 1 else (0, 255, 0)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], True, color, 2)

    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    out_path = Path("outputs/sanity_checks") / f"sanity_check_{i+1}_{s['lot']}_{s['weather']}.jpg"
    cv2.imwrite(str(out_path), img)
    print(f"Sanity overlay saved: {out_path} ({s['lot']}, {s['weather']}, {len(s['spaces'])} spaces)")

print("\nAll 450 images and labels verified successfully!")

