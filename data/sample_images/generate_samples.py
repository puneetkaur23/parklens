"""
Generate sample synthetic parking lot images and region coordinates
for immediate out-of-the-box demonstration of ParkLens.
"""

import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def generate_sample_parking_lot():
    width = 960
    height = 640
    # Create asphalt background with realistic texture
    asphalt = np.random.randint(48, 56, (height, width, 3), dtype=np.uint8)
    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    asphalt = np.clip(asphalt.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Road driving lane in the middle
    cv2.rectangle(asphalt, (0, 260), (width, 380), (45, 45, 45), -1)
    # Center dashed yellow line
    for x in range(0, width, 40):
        cv2.line(asphalt, (x, 320), (x + 20, 320), (30, 200, 220), 2)

    # Define parking bay grid: 2 rows of 6 spaces = 12 spaces
    # Row 1 (Top): y from 60 to 240 (height 180, width 120)
    # Row 2 (Bottom): y from 400 to 580 (height 180, width 120)
    spaces_info = []

    # Occupancy pattern for demonstration:
    # Top row: [Occ, Vac, Occ, Vac, Occ, Vac]
    # Bottom row: [Vac, Occ, Vac, Occ, Vac, Occ]
    occupancy_pattern = [
        # Space ID, row, col_idx, is_occupied, car_color
        ("A1", 0, 0, True, (180, 50, 40)),     # Blue car
        ("A2", 0, 1, False, None),              # Empty
        ("A3", 0, 2, True, (220, 220, 225)),  # White car
        ("A4", 0, 3, False, None),              # Empty
        ("A5", 0, 4, True, (40, 40, 40)),      # Dark gray/black car
        ("A6", 0, 5, False, None),              # Empty
        ("B1", 1, 0, False, None),              # Empty
        ("B2", 1, 1, True, (30, 30, 180)),     # Red car
        ("B3", 1, 2, False, None),              # Empty
        ("B4", 1, 3, True, (200, 190, 190)),  # Silver car
        ("B5", 1, 4, False, None),              # Empty
        ("B6", 1, 5, True, (35, 140, 50)),     # Green car
    ]

    regions_json = []

    # XML root for PKLot benchmark demonstration
    xml_root = ET.Element("parking", id="ParkLens_Sample_Lot")

    for space_id, row_idx, col_idx, is_occ, car_color in occupancy_pattern:
        sx = 80 + col_idx * 135
        sy = 70 if row_idx == 0 else 410
        sw = 115
        sh = 160

        # Draw white parking divider lines
        cv2.rectangle(asphalt, (sx, sy), (sx + sw, sy + sh), (220, 220, 220), 3)

        # Space label on the asphalt
        cv2.putText(
            asphalt,
            space_id,
            (sx + 10, sy + 30 if row_idx == 0 else sy + sh - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

        # Region coordinates (clockwise rectangle)
        pts = [[sx + 4, sy + 4], [sx + sw - 4, sy + 4], [sx + sw - 4, sy + sh - 4], [sx + 4, sy + sh - 4]]
        regions_json.append({"id": space_id, "points": pts, "type": "rectangle"})

        # Add to XML
        space_elem = ET.SubElement(xml_root, "space", id=space_id, occupied="1" if is_occ else "0")
        contour_elem = ET.SubElement(space_elem, "contour")
        for pt in pts:
            ET.SubElement(contour_elem, "point", x=str(pt[0]), y=str(pt[1]))

        # If occupied, render a vehicle with body, windshield, roof, windows, and headlights
        if is_occ and car_color is not None:
            cx = sx + 15
            cy = sy + 15
            cw = sw - 30
            ch = sh - 30

            # Car shadow
            cv2.rectangle(asphalt, (cx - 3, cy - 3), (cx + cw + 4, cy + ch + 4), (20, 20, 20), -1)
            # Car body
            cv2.rectangle(asphalt, (cx, cy), (cx + cw, cy + ch), car_color, -1)
            cv2.rectangle(asphalt, (cx, cy), (cx + cw, cy + ch), (60, 60, 60), 2)

            # Windshield (dark glass)
            wy = cy + 25 if row_idx == 0 else cy + ch - 45
            cv2.rectangle(asphalt, (cx + 6, wy), (cx + cw - 6, wy + 20), (35, 35, 40), -1)

            # Roof (slightly lighter than car body)
            ry = cy + 50 if row_idx == 0 else cy + 20
            roof_color = tuple(min(255, int(c * 1.15)) for c in car_color)
            cv2.rectangle(asphalt, (cx + 8, ry), (cx + cw - 8, ry + ch - 70), roof_color, -1)

            # Rear window
            rwy = cy + ch - 25 if row_idx == 0 else cy + 5
            cv2.rectangle(asphalt, (cx + 8, rwy), (cx + cw - 8, rwy + 15), (35, 35, 40), -1)

            # Headlights / Taillights
            if row_idx == 0:
                # Facing down (towards road)
                cv2.circle(asphalt, (cx + 8, cy + ch - 4), 4, (20, 220, 255), -1)
                cv2.circle(asphalt, (cx + cw - 8, cy + ch - 4), 4, (20, 220, 255), -1)
            else:
                # Facing up (towards road)
                cv2.circle(asphalt, (cx + 8, cy + 4), 4, (20, 220, 255), -1)
                cv2.circle(asphalt, (cx + cw - 8, cy + 4), 4, (20, 220, 255), -1)

    # Save sample image
    img_path = r"C:\Users\Puneet Kaur\.gemini\antigravity\scratch\ParkLens\data\sample_images\sample_parking_lot.jpg"
    cv2.imwrite(img_path, asphalt)

    # Save regions JSON
    json_path = r"C:\Users\Puneet Kaur\.gemini\antigravity\scratch\ParkLens\data\sample_images\sample_regions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(regions_json, f, indent=2)

    # Save XML annotation (for PKLot benchmark demonstration)
    xml_path = r"C:\Users\Puneet Kaur\.gemini\antigravity\scratch\ParkLens\data\sample_images\sample_parking_lot.xml"
    tree = ET.ElementTree(xml_root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    print(f"Generated sample image at: {img_path}")
    print(f"Generated sample regions at: {json_path}")
    print(f"Generated sample XML at: {xml_path}")


if __name__ == "__main__":
    generate_sample_parking_lot()
