import os
import cv2
import numpy as np

# -----------------------------
# Folder Paths
# -----------------------------
OVERLAY_BEFORE = "overlays_before"
OVERLAY_AFTER = "overlays_after"

MASK_BEFORE = "refined_masks_before"
MASK_AFTER = "refined_masks_after"

OUTPUT_DIR = "comparison_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_title(img, title):
    banner = np.full((40, img.shape[1], 3), 40, dtype=np.uint8)

    cv2.putText(
        banner,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.vstack((banner, img))


files = sorted(os.listdir(OVERLAY_BEFORE))

for file in files:

    before_overlay = cv2.imread(os.path.join(OVERLAY_BEFORE, file))
    after_overlay = cv2.imread(os.path.join(OVERLAY_AFTER, file))

    before_mask = cv2.imread(
        os.path.join(MASK_BEFORE, file),
        cv2.IMREAD_GRAYSCALE,
    )

    after_mask = cv2.imread(
        os.path.join(MASK_AFTER, file),
        cv2.IMREAD_GRAYSCALE,
    )

    if (
        before_overlay is None
        or after_overlay is None
        or before_mask is None
        or after_mask is None
    ):
        print(f"Skipping {file}")
        continue

    before_mask = cv2.cvtColor(before_mask, cv2.COLOR_GRAY2BGR)
    after_mask = cv2.cvtColor(after_mask, cv2.COLOR_GRAY2BGR)

    h, w = before_overlay.shape[:2]

    before_mask = cv2.resize(before_mask, (w, h))
    after_mask = cv2.resize(after_mask, (w, h))

    before_overlay = add_title(before_overlay, "Before Overlay")
    after_overlay = add_title(after_overlay, "After Overlay")

    before_mask = add_title(before_mask, "Before Refined Mask")
    after_mask = add_title(after_mask, "After Refined Mask")

    top = np.hstack((before_overlay, after_overlay))
    bottom = np.hstack((before_mask, after_mask))

    comparison = np.vstack((top, bottom))

    banner = np.full((55, comparison.shape[1], 3), 25, dtype=np.uint8)

    cv2.putText(
        banner,
        file,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    comparison = np.vstack((banner, comparison))

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, file),
        comparison,
    )

print("Comparison images generated successfully!")