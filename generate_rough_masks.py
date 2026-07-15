import os

from config import (
    IMAGE_DIR,
    ROUGH_MASK_DIR,
    REFINED_MASK_DIR,
    RESULTS_DIR,
    GAUSSIAN_KERNEL,
    LOWER_HSV,
    UPPER_HSV,
    MORPH_KERNEL,
    CLOSE_KERNEL,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    ROI_BOTTOM_FRACTION,
    EXCLUDE_REGIONS_FRACTION,
    MIN_COMPONENT_AREA,
    MAX_COMPONENT_AREA,
    MIN_AREA,
    MIN_CIRCULARITY,
    IMAGE_EXTENSIONS,
)

from utils import (
    create_directories,
    read_image,
    save_image,
    gaussian_blur,
    median_blur,
    convert_to_hsv,
    equalize_lighting,
    apply_roi_mask,
    generate_rough_mask,
    apply_morphology,
    filter_connected_components,
    filter_by_circularity,
)

def main():

    create_directories([
        ROUGH_MASK_DIR,
        REFINED_MASK_DIR,
        RESULTS_DIR
    ])

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]

    print(f"Found {len(image_files)} images.")

    if not image_files:
        print("No images found!")
        return

    for image_name in image_files:

        image_path = os.path.join(
            IMAGE_DIR,
            image_name
        )

        # Read original image
        image = read_image(image_path)

        # Zero out static non-product regions (fixture, reflection strip)
        # before any color processing touches them
        image = apply_roi_mask(
            image,
            ROI_BOTTOM_FRACTION,
            EXCLUDE_REGIONS_FRACTION
        )

        # Apply Gaussian Blur
        blurred = gaussian_blur(
            image,
            GAUSSIAN_KERNEL
        )

        blurred = median_blur(
            blurred,
            5
        )

        # Convert to HSV
        hsv = convert_to_hsv(
            blurred
        )

        # --- DEBUG: mask WITHOUT CLAHE, for direct before/after comparison ---
        no_clahe_mask = generate_rough_mask(
            hsv,
            LOWER_HSV,
            UPPER_HSV
        )
        save_image(
            os.path.join(RESULTS_DIR, f"debug_1_no_clahe_{image_name}"),
            no_clahe_mask
        )

        # Normalize uneven lighting (e.g. shadowed frame edges) before
        # thresholding, so dim regions aren't lost
        hsv = equalize_lighting(
            hsv,
            CLAHE_CLIP_LIMIT,
            CLAHE_TILE_GRID
        )

        # Generate rough mask
        rough_mask = generate_rough_mask(
            hsv,
            LOWER_HSV,
            UPPER_HSV
        )

        save_image(
            os.path.join(
                RESULTS_DIR,
                f"debug_2_with_clahe_{image_name}"
            ),
            rough_mask
        )

        # Morphological cleaning (larger kernel for closing bridges gaps
        # in partial/broken rings)
        rough_mask = apply_morphology(
            rough_mask,
            MORPH_KERNEL,
            CLOSE_KERNEL
        )

        save_image(
            os.path.join(RESULTS_DIR, f"debug_3_after_morph_{image_name}"),
            rough_mask
        )

        # Remove invalid connected components
        rough_mask = filter_connected_components(
            rough_mask,
            MIN_COMPONENT_AREA,
            MAX_COMPONENT_AREA,
        )

        save_image(
            os.path.join(RESULTS_DIR, f"debug_4_after_components_{image_name}"),
            rough_mask
        )

        # Remove non-ring-shaped fragments (dashes, partial arcs, noise)
        rough_mask = filter_by_circularity(
            rough_mask,
            MIN_AREA,
            MIN_CIRCULARITY
        )

        save_image(
            os.path.join(RESULTS_DIR, f"debug_5_after_circularity_{image_name}"),
            rough_mask
        )

        # Save rough mask
        output_name = os.path.splitext(image_name)[0] + ".png"

        save_image(
            os.path.join(
                ROUGH_MASK_DIR,
                output_name
            ),
            rough_mask
        )

        print(f"Processed: {image_name}")

    print("All rough masks generated successfully!")


if __name__ == "__main__":
    main()