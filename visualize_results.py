import os
import cv2

from config import (
    IMAGE_DIR,
    ROUGH_MASK_DIR,
    REFINED_MASK_DIR,
    VISUALIZATION_DIR,
    OVERLAY_DIR,
    IMAGE_EXTENSIONS,
)

from utils import (
    create_directories,
    read_image,
    save_image,
    overlay_mask,
    create_debug_visualization,
)


def find_original_image(mask_filename):
    """
    Find the corresponding original image for a mask.
    Supports different image extensions.
    """

    base_name = os.path.splitext(mask_filename)[0]

    for ext in IMAGE_EXTENSIONS:

        image_path = os.path.join(
            IMAGE_DIR,
            base_name + ext
        )

        if os.path.exists(image_path):
            return image_path

    return None


def main():

    create_directories([
        VISUALIZATION_DIR,
        OVERLAY_DIR,
    ])

    refined_masks = sorted([
        f for f in os.listdir(REFINED_MASK_DIR)
        if f.lower().endswith(".png")
    ])

    print(f"Found {len(refined_masks)} refined masks.\n")

    if not refined_masks:
        print("No refined masks found.")
        return

    for mask_name in refined_masks:

        original_path = find_original_image(mask_name)

        if original_path is None:
            print(f"Original image not found: {mask_name}")
            continue

        rough_path = os.path.join(
            ROUGH_MASK_DIR,
            mask_name
        )

        refined_path = os.path.join(
            REFINED_MASK_DIR,
            mask_name
        )

        if not os.path.exists(rough_path):
            print(f"Missing rough mask: {mask_name}")
            continue

        if not os.path.exists(refined_path):
            print(f"Missing refined mask: {mask_name}")
            continue

        # Read files
        original = read_image(original_path)

        rough_mask = cv2.imread(
            rough_path,
            cv2.IMREAD_GRAYSCALE
        )

        refined_mask = cv2.imread(
            refined_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Create overlay - the mask drawn directly on top of the donuts
        # (green fill + outline + numbered ID), not a separate binary
        # image. This is the primary result to share.
        overlay = overlay_mask(
            original.copy(),
            refined_mask
        )

        overlay_save_path = os.path.join(
            OVERLAY_DIR,
            mask_name
        )

        save_image(
            overlay_save_path,
            overlay
        )

        # Also keep the full 4-panel debug comparison for internal
        # troubleshooting (original / rough / refined / overlay side by
        # side) - useful for us, not the primary deliverable
        comparison = create_debug_visualization(
            original,
            rough_mask,
            refined_mask,
            overlay
        )

        comparison_save_path = os.path.join(
            VISUALIZATION_DIR,
            mask_name
        )

        save_image(
            comparison_save_path,
            comparison
        )

        print(f"Saved: {mask_name}")

    print("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()