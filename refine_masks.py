import os
import cv2

from config import (
    IMAGE_DIR,
    ROUGH_MASK_DIR,
    REFINED_MASK_DIR,
    VISUALIZATION_DIR,
    MORPH_KERNEL,
    CLOSE_KERNEL,
    MIN_COMPONENT_AREA,
    MAX_COMPONENT_AREA,
    MIN_AREA,
    MIN_CIRCULARITY,
    ENABLE_SHAPE_COMPLETION,
    SHAPE_COMPLETION_MAX_CIRCULARITY,
    COMPLETED_MASK_DIR,
    COMPLETED_OVERLAY_DIR,
)

from utils import (
    create_directories,
    read_image,
    save_image,
    apply_morphology,
    filter_connected_components,
    filter_by_circularity,
    overlay_mask,
    complete_ring_shapes,
)


def main():

    # Create output directory
    create_directories([
        REFINED_MASK_DIR,
        VISUALIZATION_DIR,
        COMPLETED_MASK_DIR,
        COMPLETED_OVERLAY_DIR,
    ])

    # Get all rough mask files
    mask_files = [
        f for f in os.listdir(ROUGH_MASK_DIR)
        if f.lower().endswith(".png")
    ]

    print(f"Found {len(mask_files)} masks.")

    if not mask_files:
        print("No rough masks found!")
        return

    # Process every rough mask
    for mask_name in mask_files:

        mask_path = os.path.join(
            ROUGH_MASK_DIR,
            mask_name
        )

        mask = read_image(mask_path)

        image_name = os.path.splitext(mask_name)[0] + ".jpg"

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        image = cv2.imread(image_path)

        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_BGR2GRAY
            )

        # Step 1 - Morphological refinement
        mask = apply_morphology(
            mask,
            MORPH_KERNEL,
            CLOSE_KERNEL
        )

        # Make sure mask is binary before connected components
        _,mask = cv2.threshold(
            mask,
            127,
            255,
            cv2.THRESH_BINARY
        )

        mask = filter_connected_components(
            mask,
            MIN_COMPONENT_AREA,
            MAX_COMPONENT_AREA
        )

        # Step 3 - Remove irregular contours
        mask = filter_by_circularity(
            mask,
            MIN_AREA,
            MIN_CIRCULARITY
        )

        # Save refined mask
        save_image(
            os.path.join(
                REFINED_MASK_DIR,
                mask_name
            ),
            mask
        )

        overlay = overlay_mask(image, mask)

        save_image(
            os.path.join(
                VISUALIZATION_DIR,
                mask_name
            ),
            overlay
        )

        # Shape completion (opt-in, separate output) - reconstructs
        # broken/partial rings (e.g. from lighting-related threshold
        # gaps) as clean annuli. Well-formed rings are left untouched.
        # Saved separately from REFINED_MASK_DIR so it can be compared
        # before deciding whether to use it as the primary result.
        if ENABLE_SHAPE_COMPLETION:

            completed = complete_ring_shapes(
                mask,
                MIN_AREA,
                SHAPE_COMPLETION_MAX_CIRCULARITY
            )

            save_image(
                os.path.join(
                    COMPLETED_MASK_DIR,
                    mask_name
                ),
                completed
            )

            completed_overlay = overlay_mask(image.copy(), completed)

            save_image(
                os.path.join(
                    COMPLETED_OVERLAY_DIR,
                    mask_name
                ),
                completed_overlay
            )

        print(f"Processed: {mask_name}")

    print("All masks refined successfully!")


if __name__ == "__main__":
    main()