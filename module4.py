import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import random

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

set_seed(1)

# =====================================================
# PART 1 — Classical OpenCV Segmentation
# =====================================================

def classical_segmentation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found.")

    # Step 1: Noise reduction
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 2: Otsu thresholding
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Step 4: Edge refinement
    edges = cv2.Canny(blurred, 50, 150)
    combined = cv2.bitwise_or(cleaned, edges)

    # Step 5: Contour extraction
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Select largest contour (assumed animal)
    animal_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(img)
    cv2.drawContours(mask, [animal_contour], -1, 255, thickness=-1)

    return mask


# =====================================================
# PART 2 — SAM2 Segmentation
# =====================================================

def sam2_segmentation(image_path):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cpu")
    print(f"Using device: {device}")

    model = build_sam2(
    config_file="sam2_hiera_l.yaml",
    checkpoint="sam2_hiera_large.pt",
    device=device
    )

    model = model.to(device)


    predictor = SAM2ImagePredictor(model)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict()

    # Select highest scoring mask
    best_mask = masks[np.argmax(scores)]

    sam_mask = (best_mask * 255).astype(np.uint8)

    return sam_mask


# =====================================================
# PART 3 — Evaluation Metrics
# =====================================================

def calculate_iou(mask_a, mask_b):
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    return intersection / union if union != 0 else 0


def calculate_dice(mask_a, mask_b):
    mask_a = mask_a > 0
    mask_b = mask_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    return (2 * intersection) / (mask_a.sum() + mask_b.sum())


# =====================================================
# PART 4 — Visualization
# =====================================================

def visualize_results(image_path, classical_mask, sam_mask):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Classical OpenCV Segmentation")
    plt.imshow(classical_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("SAM2 Segmentation")
    plt.imshow(sam_mask, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    image_path = "thermal_animal.jpg"

    print("Running Classical OpenCV segmentation...")
    start = time.time()
    classical_mask = classical_segmentation(image_path)
    classical_time = time.time() - start

    if classical_mask is None:
        print("No animal detected using classical method.")
        exit()

    print("Running SAM2 segmentation...")
    start = time.time()
    sam_mask = sam2_segmentation(image_path)
    sam_time = time.time() - start

    # Resize masks if needed
    if classical_mask.shape != sam_mask.shape:
        sam_mask = cv2.resize(
            sam_mask,
            (classical_mask.shape[1], classical_mask.shape[0])
        )

    # Compute metrics
    iou_score = calculate_iou(classical_mask, sam_mask)
    dice_score = calculate_dice(classical_mask, sam_mask)

    print("\n==== Segmentation Comparison Results ====")
    print(f"IoU Score  : {iou_score:.4f}")
    print(f"Dice Score : {dice_score:.4f}")
    print(f"Classical Runtime: {classical_time:.4f} seconds")
    print(f"SAM2 Runtime      : {sam_time:.4f} seconds")

    visualize_results(image_path, classical_mask, sam_mask)