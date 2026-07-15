import os

# Folder Paths
IMAGE_DIR = "images"
ROUGH_MASK_DIR = "rough_masks"
REFINED_MASK_DIR = "refined_masks"
RESULTS_DIR = "results"
VISUALIZATION_DIR = "visualizations"
OVERLAY_DIR = "overlays"

# Gaussian Blur
GAUSSIAN_KERNEL = (3, 3)

# Morphology
MORPH_KERNEL = (3, 3)
CLOSE_KERNEL = (5, 5)   # was (7,7) - too large, was merging separate noise into blobs

# CLAHE (lighting normalization, applied to V channel before HSV threshold)
# NOTE: clip_limit=2.0 with tile_grid=(8,8) was too aggressive on this
# footage - most tiles are mostly-black background, so local contrast
# stretching swung borderline pixels unpredictably: some real rings lost
# enough shape to fail the circularity filter, while background noise
# got pulled into range. Dialed back on both axes.
CLAHE_CLIP_LIMIT = 1.2
CLAHE_TILE_GRID = (16, 16)

# HSV Thresholds
# V floor split the difference between original (110, missed dim regions)
# and first attempt (90, let in too much noise under CLAHE)
LOWER_HSV = (10, 15, 90)
UPPER_HSV = (45, 160, 255)

# Connected Components
# NOTE: ring size increases toward the bottom of the frame (perspective -
# closer objects project larger). Measured range across a full frame is
# ~6,700-13,400 area and up to ~150px width/height. Bounds widened with
# margin so no valid row gets clipped.
MIN_COMPONENT_AREA = 80
MAX_COMPONENT_AREA = None

# Circularity filter (removes dash/fragment noise that isn't ring-shaped)
MIN_AREA = 1200
MIN_CIRCULARITY = 0.40

# Shape completion (opt-in) - reconstructs broken/partial rings as clean
# annuli, for cases where the break is a lighting/threshold gap rather
# than a real defect. Saved to separate directories so it can be
# compared against the standard refined output before deciding to use
# it. Components at or above this circularity are left untouched.
ENABLE_SHAPE_COMPLETION = True
SHAPE_COMPLETION_MAX_CIRCULARITY = 0.6
COMPLETED_MASK_DIR = "completed_masks"
COMPLETED_OVERLAY_DIR = "overlays_completed"

# Region of Interest (ROI)
# The rig has two static, fixed-position non-product regions that fall
# within the HSV band and were showing up as noise in every frame:
#   1. A metal clamp/bracket assembly in the bottom-left corner.
#   2. A reflective strip below the conveyor rail that mirrors the rings
#      above it (not real product - a distorted duplicate).
# Both are fixed relative to the camera, so they're excluded with a
# static mask rather than tuned around with color/shape thresholds.
# Expressed as fractions of image (height, width) so this still works if
# frame resolution changes.
ROI_BOTTOM_FRACTION = 0.82  # ignore everything below this row (rail + reflection)
EXCLUDE_REGIONS_FRACTION = [
    # (x_min, y_min, x_max, y_max) - bottom-left clamp/bracket
    (0.0, 0.60, 0.22, 1.0),
]

# Supported Image Formats
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)