#!/usr/bin/env python3
"""Create side-by-side comparison images (input | output)."""

import os
import cv2
import numpy as np
from glob import glob

INPUT_DIR = "real_estate_test/input"
OUTPUT_DIR = "real_estate_test/output"
COMPARISON_DIR = "real_estate_test/comparison"

def create_comparison(input_path, output_path, save_path):
    """Create side-by-side comparison image."""
    img_in = cv2.imread(input_path)
    img_out = cv2.imread(output_path)

    if img_in is None or img_out is None:
        return False

    # Ensure same height
    h1, w1 = img_in.shape[:2]
    h2, w2 = img_out.shape[:2]

    if h1 != h2:
        # Resize output to match input height
        scale = h1 / h2
        img_out = cv2.resize(img_out, (int(w2 * scale), h1))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 4

    # Add "INPUT" label
    cv2.putText(img_in, "INPUT", (30, 80), font, font_scale, (0, 0, 0), thickness + 4)
    cv2.putText(img_in, "INPUT", (30, 80), font, font_scale, (255, 255, 255), thickness)

    # Add "OUTPUT" label
    cv2.putText(img_out, "OUTPUT", (30, 80), font, font_scale, (0, 0, 0), thickness + 4)
    cv2.putText(img_out, "OUTPUT", (30, 80), font, font_scale, (255, 255, 255), thickness)

    # Create divider line
    divider = np.ones((h1, 4, 3), dtype=np.uint8) * 128

    # Concatenate horizontally
    comparison = np.hstack([img_in, divider, img_out])

    cv2.imwrite(save_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True

def main():
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    input_files = sorted(glob(os.path.join(INPUT_DIR, "*.jpg")))

    print(f"Creating {len(input_files)} comparison images...")

    created = 0
    for i, inp_path in enumerate(input_files):
        filename = os.path.basename(inp_path)
        name = os.path.splitext(filename)[0]

        out_path = os.path.join(OUTPUT_DIR, filename)
        save_path = os.path.join(COMPARISON_DIR, f"{i:04d}_input_output.jpg")

        if os.path.exists(out_path):
            if create_comparison(inp_path, out_path, save_path):
                created += 1
                if created <= 5 or created % 20 == 0:
                    print(f"  [{created}] {save_path}")

    print(f"\nCreated {created} comparison images in {COMPARISON_DIR}/")

    # Show total size
    total_size = sum(os.path.getsize(f) for f in glob(os.path.join(COMPARISON_DIR, "*.jpg")))
    print(f"Total size: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main()
