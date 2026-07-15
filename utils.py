import os
import cv2
import numpy as np

def create_directories(paths):
    """
    Create output directories if they do not exist.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

def read_image(image_path):
    """
    Read an image using OpenCV.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return image


def save_image(save_path, image):
    """
    Save image to disk.
    """
    cv2.imwrite(save_path, image)

def gaussian_blur(image, kernel_size):
    """
    Apply Gaussian Blur to reduce noise.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def median_blur(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

def convert_to_hsv(image):
    """
    Convert BGR image to HSV.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def equalize_lighting(hsv_image, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Apply CLAHE to the V (brightness) channel of an HSV image to
    normalize uneven lighting (e.g. shadowed regions at frame edges)
    before thresholding. This recovers detail in dim regions without
    blowing out already-bright regions.
    """

    h, s, v = cv2.split(hsv_image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid
    )

    v_eq = clahe.apply(v)

    return cv2.merge([h, s, v_eq])

def apply_roi_mask(image, roi_bottom_fraction=1.0, exclude_regions_fraction=None):
    """
    Zero out static, non-product regions of the frame before thresholding:
    a bottom cutoff (e.g. conveyor rail + reflection strip below it) and
    any number of excluded rectangles (e.g. a fixed mechanical fixture).

    roi_bottom_fraction: keep rows above this fraction of image height,
        zero out everything from that row down.
    exclude_regions_fraction: list of (x_min, y_min, x_max, y_max) tuples,
        each as a fraction of (width, height), zeroed out regardless of
        the bottom cutoff.

    Works on grayscale/binary masks or BGR/HSV images (any number of
    channels) - operates on rows/columns, not pixel values.
    """

    result = image.copy()
    h, w = image.shape[:2]

    bottom_row = int(h * roi_bottom_fraction)
    result[bottom_row:, :] = 0

    if exclude_regions_fraction:
        for x_min_f, y_min_f, x_max_f, y_max_f in exclude_regions_fraction:
            x_min = int(w * x_min_f)
            y_min = int(h * y_min_f)
            x_max = int(w * x_max_f)
            y_max = int(h * y_max_f)
            result[y_min:y_max, x_min:x_max] = 0

    return result

def generate_rough_mask(hsv_image, lower_hsv, upper_hsv):
    """
    Generate rough mask using a single HSV range threshold.

    NOTE: Previously this also applied a standalone V > 120 threshold and
    an S < 110 threshold on top of the inRange() result. Those two extra
    thresholds silently narrowed the effective range to V>120 and S<110,
    which is tighter than (and inconsistent with) LOWER_HSV/UPPER_HSV in
    config.py. That caused valid pixels within the configured HSV range
    to be dropped. If you need to exclude the saturated blue conveyor,
    do it by tightening the S upper bound in UPPER_HSV in config.py
    instead of adding a second contradictory threshold here.
    """

    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)

    mask = cv2.inRange(
        hsv_image,
        lower,
        upper
    )

    return mask

def apply_morphology(mask, kernel_size, close_kernel_size=None):
    """
    Apply morphological opening and closing to clean the rough mask.

    close_kernel_size (optional) lets closing use a larger kernel than
    opening, which bridges gaps in partial/broken rings without
    over-thickening real edges during the opening step.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        kernel_size
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        close_kernel_size if close_kernel_size is not None else kernel_size
    )

    # First repair broken donut edges
    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    # Smooth edges while preserving shape
    closed = cv2.medianBlur(closed, 5)

    return closed

def filter_connected_components(
    mask,
    min_area,
    max_area=None
):
    """
    Remove only tiny connected components.

    We intentionally DO NOT filter by width or height because
    partially visible donuts near the image borders can have
    small bounding boxes but are still valid.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        # Remove tiny noise
        if area < min_area:
            continue

        # Optional upper limit
        if max_area is not None and area > max_area:
            continue

        cleaned[labels == i] = 255

    return cleaned

def filter_by_circularity(mask, min_area, min_circularity):
    """
    Keep only contours that are sufficiently circular.

    Circularity:
        1.0 = perfect circle
        ~0.7-0.9 = donut/ring
        <0.4 = elongated objects, rails, reflections

    NOTE: cv2.findContours with RETR_EXTERNAL only returns each ring's
    OUTER boundary - it has no notion of the inner hole. Filling that
    outer contour with cv2.FILLED therefore fills the hole solid too,
    turning every donut into a filled disc. We restore the hole at the
    end by re-intersecting with the original mask (which is 0 at the
    hole), rather than using the filled contour directly.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cleaned = np.zeros_like(mask)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        # Uncomment for debugging
        # print(f"Area={area:.0f}, Circularity={circularity:.3f}")

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Keep good circular contours
        if circularity >= min_circularity:
            cv2.drawContours(
                cleaned,
                [contour],
                -1,
                255,
                cv2.FILLED
            )
            continue

        # Also keep broken donuts that are still roughly square
        if 0.75 <= aspect_ratio <= 1.30:
            cv2.drawContours(
                cleaned,
                [contour],
                -1,
                255,
                cv2.FILLED
            )

    # Re-intersect with the original mask so donut holes (0 in the
    # original) are restored rather than left filled solid.
    return cv2.bitwise_and(cleaned, mask)

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.4):
    """
    Overlay the refined mask, draw contour outlines,
    and label each detected donut.
    """

    overlay = image.copy()

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Transparent mask overlay
    overlay[binary > 0] = (
        alpha * np.array(color) +
        (1 - alpha) * overlay[binary > 0]
    ).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours from top-to-bottom, then left-to-right
    contours = sorted(
        contours,
        key=lambda c: (
            cv2.boundingRect(c)[1],
            cv2.boundingRect(c)[0]
        )
    )

    # Draw contour outlines
    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    # Draw IDs
    for idx, contour in enumerate(contours, start=1):

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.putText(
            overlay,
            str(idx),
            (cx - 8, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        ) 

    return overlay

def create_debug_visualization(
    original,
    rough_mask,
    refined_mask,
    overlay
):
    """
    Create a 2x2 comparison image.

    +----------------------+----------------------+
    | Original             | Rough Mask           |
    +----------------------+----------------------+
    | Refined Mask         | Overlay              |
    +----------------------+----------------------+
    """

    # Convert masks to BGR
    rough_bgr = cv2.cvtColor(
        rough_mask,
        cv2.COLOR_GRAY2BGR
    )

    refined_bgr = cv2.cvtColor(
        refined_mask,
        cv2.COLOR_GRAY2BGR
    )

    # Copy images
    original = original.copy()
    rough_bgr = rough_bgr.copy()
    refined_bgr = refined_bgr.copy()
    overlay = overlay.copy()

    # Resize to same size
    h, w = original.shape[:2]

    rough_bgr = cv2.resize(
        rough_bgr,
        (w, h)
    )

    refined_bgr = cv2.resize(
        refined_bgr,
        (w, h)
    )

    overlay = cv2.resize(
        overlay,
        (w, h)
    )

    # Add titles
    original = add_title(original, "Original")
    rough_bgr = add_title(rough_bgr, "Rough Mask")
    refined_bgr = add_title(refined_bgr, "Refined Mask")
    overlay = add_title(overlay, "Overlay")

    # Stack images
    top = np.hstack((original, rough_bgr))
    bottom = np.hstack((refined_bgr, overlay))

    comparison = np.vstack((top, bottom))

    # Add overall title
    banner = np.full(
        (50, comparison.shape[1], 3),
        30,
        dtype=np.uint8
    )

    cv2.putText(
        banner,
        "Donut Segmentation Pipeline",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255,255,255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        banner,
        f"Detected Donuts : {len(cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}",
        (20,58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220,220,220),
        2,
        cv2.LINE_AA
    )

    comparison = np.vstack((banner, comparison))

    return comparison

def add_title(image, title):
    """
    Add a header above an image.
    """

    banner = np.full(
        (40, image.shape[1], 3),
        40,
        dtype=np.uint8
    )

    cv2.putText(
        banner,
        title,
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return np.vstack((banner, image))

def estimate_ring_ratio(mask, min_circularity=0.6):
    """
    Estimate the typical inner-hole-radius / outer-radius ratio from the
    well-formed rings already present in this mask. Used as a fallback
    for broken/partial rings that have no usable hole contour of their
    own to measure directly.
    """

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        return 0.5

    hierarchy = hierarchy[0]
    ratios = []

    for i, h in enumerate(hierarchy):
        _, _, child, parent = h

        if parent != -1:
            continue

        area = cv2.contourArea(contours[i])
        perimeter = cv2.arcLength(contours[i], True)

        if perimeter == 0 or area < 200:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity < min_circularity or child == -1:
            continue

        _, outer_r = cv2.minEnclosingCircle(contours[i])
        _, inner_r = cv2.minEnclosingCircle(contours[child])

        if outer_r > 0:
            ratios.append(inner_r / outer_r)

    return float(np.median(ratios)) if ratios else 0.5

def complete_ring_shapes(mask, min_area=200, max_circularity_for_completion=0.6):
    """
    Reconstruct broken/partial rings as clean annuli by fitting a
    minimum enclosing circle to each component's outer boundary (and,
    where present, its inner hole boundary).

    Only applied to components whose outer-boundary circularity falls
    below max_circularity_for_completion - well-formed rings are kept
    exactly as detected, so real shape defects on healthy-looking
    donuts aren't papered over.

    IMPORTANT ASSUMPTION: this treats an incomplete detection as a
    lighting/threshold gap, not a genuine defect. For defect-shape
    inspection, this reconstruction should NOT be used as the shape you
    measure/inspect against - it's a display/counting aid, not ground
    truth for the donut's actual physical edge.
    """

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = np.zeros_like(mask)

    if hierarchy is None:
        return mask.copy()

    hierarchy = hierarchy[0]
    fallback_ratio = estimate_ring_ratio(mask)

    for i, h in enumerate(hierarchy):
        _, _, child, parent = h

        if parent != -1:
            continue

        area = cv2.contourArea(contours[i])

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contours[i], True)

        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        (cx, cy), outer_r = cv2.minEnclosingCircle(contours[i])

        if circularity >= max_circularity_for_completion:
            # Already well-formed - keep the original detected pixels,
            # hole included, exactly as they were.
            cv2.drawContours(result, contours, i, 255, cv2.FILLED)
            if child != -1:
                cv2.drawContours(result, contours, child, 0, cv2.FILLED)
            continue

        # Broken/partial - reconstruct as a clean annulus rather than
        # leaving the jagged partial-arc shape
        if child != -1:
            _, inner_r = cv2.minEnclosingCircle(contours[child])
        else:
            inner_r = outer_r * fallback_ratio

        cv2.circle(result, (int(cx), int(cy)), int(outer_r), 255, -1)
        cv2.circle(result, (int(cx), int(cy)), int(inner_r), 0, -1)

    return result