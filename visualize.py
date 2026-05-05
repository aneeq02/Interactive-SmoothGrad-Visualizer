"""
visualize.py  —  Person B's file
Handles: all saliency map rendering, post-processing, matplotlib figure creation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# ─────────────────────────────────────────────
#  Core post-processing pipeline
#  (directly from Section 3.1 of the paper)
# ─────────────────────────────────────────────
def postprocess(grad_array, use_absolute=True, percentile_cap=True, multiply_input=None):
    """
    Post-process a raw gradient array into a displayable saliency map.

    Steps (all from the SmoothGrad paper, Section 3.1):
      1. Optionally take absolute value  (paper recommends True for ImageNet)
      2. Optionally cap outliers at 99th percentile  (prevents all-black maps)
      3. Optionally multiply by original image  (shows contribution, not just sensitivity)
      4. Normalize to [0, 1] for display

    Args:
        grad_array: numpy (224, 224, 3) — raw gradient output
        use_absolute: if True, take |gradient| before anything else
        percentile_cap: if True, clip values above 99th percentile
        multiply_input: numpy (224, 224, 3) original image pixels, or None

    Returns:
        numpy (224, 224, 3) float32 in range [0, 1]
    """
    result = grad_array.copy()

    if use_absolute:
        result = np.abs(result)

    if percentile_cap:
        cap = np.percentile(result, 99)
        if cap > 0:
            result = np.clip(result, 0, cap)

    if multiply_input is not None:
        result = result * np.abs(multiply_input)

    # normalize to [0, 1]
    vmin, vmax = result.min(), result.max()
    if vmax > vmin:
        result = (result - vmin) / (vmax - vmin)
    else:
        result = np.zeros_like(result)

    return result.astype(np.float32)


def to_grayscale(saliency_rgb):
    """
    Collapse (224, 224, 3) saliency to (224, 224) by taking max across channels.
    Max is better than mean here — highlights the most active channel per pixel.
    """
    return saliency_rgb.max(axis=2)


def pil_to_display_array(pil_image):
    """Convert a PIL image to a (224, 224, 3) float32 numpy array for display."""
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


# ─────────────────────────────────────────────
#  Single saliency map figure
# ─────────────────────────────────────────────
def render_single(pil_image, grad_array, method_name,
                  use_absolute=True, percentile_cap=True, multiply_input=False):
    """
    Render a single method's saliency map alongside the original image.
    Returns a matplotlib Figure.
    """
    original = pil_to_display_array(pil_image)
    input_for_multiply = original if multiply_input else None

    saliency = postprocess(grad_array, use_absolute, percentile_cap, input_for_multiply)
    sal_gray = to_grayscale(saliency)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.patch.set_facecolor("white")

    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    im = axes[1].imshow(sal_gray, cmap="hot")
    axes[1].set_title(f"Saliency Map — {method_name}", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Side-by-side comparison of all 4 methods
# ─────────────────────────────────────────────
def render_comparison(pil_image, grads_dict,
                      use_absolute=True, percentile_cap=True, multiply_input=False):
    """
    Render all available saliency methods side-by-side in one figure.

    Args:
        pil_image: PIL Image (user upload)
        grads_dict: dict like {"Vanilla": array, "SmoothGrad": array, ...}
        use_absolute, percentile_cap, multiply_input: post-processing options

    Returns:
        matplotlib Figure
    """
    original = pil_to_display_array(pil_image)
    input_for_multiply = original if multiply_input else None

    method_names = list(grads_dict.keys())
    n_methods = len(method_names)
    n_cols = n_methods + 1   # +1 for original image

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.patch.set_facecolor("white")

    # First column: original image
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for i, name in enumerate(method_names):
        saliency = postprocess(grads_dict[name], use_absolute, percentile_cap, input_for_multiply)
        sal_gray = to_grayscale(saliency)
        axes[i + 1].imshow(sal_gray, cmap="hot")
        axes[i + 1].set_title(name, fontsize=11)
        axes[i + 1].axis("off")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Noise level grid  (reproduces Figure 3 of paper)
# ─────────────────────────────────────────────
def render_noise_grid(pil_image, grad_fn, class_idx, sigmas, n_samples=50):
    """
    Show how noise level σ affects the saliency map quality.
    Reproduces Figure 3 from the SmoothGrad paper.

    Args:
        pil_image: PIL Image
        grad_fn: callable(image_tensor, class_idx, sigma, n) → grad array
                 (this will be the smoothgrad function from gradients.py)
        class_idx: integer
        sigmas: list of floats, e.g. [0.0, 0.10, 0.20, 0.30]
        n_samples: fixed sample count while varying σ

    Returns:
        matplotlib Figure
    """
    from gradients import preprocess_image

    image_tensor = preprocess_image(pil_image)
    original = pil_to_display_array(pil_image)
    n = len(sigmas)

    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Effect of noise level σ on SmoothGrad", fontsize=13)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for i, sigma in enumerate(sigmas):
        if sigma == 0.0:
            # sigma=0 is just vanilla gradient
            from gradients import vanilla_gradient
            grad = vanilla_gradient(model=None, image_tensor=image_tensor, class_idx=class_idx)
            label = "Vanilla (σ=0)"
        else:
            grad = grad_fn(image_tensor, class_idx, sigma, n_samples)
            label = f"σ = {int(sigma*100)}%"

        saliency = postprocess(grad)
        sal_gray = to_grayscale(saliency)
        axes[i + 1].imshow(sal_gray, cmap="hot")
        axes[i + 1].set_title(label, fontsize=11)
        axes[i + 1].axis("off")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Sample size grid  (reproduces Figure 4 of paper)
# ─────────────────────────────────────────────
def render_sample_grid(pil_image, grad_fn, class_idx, sample_counts, sigma=0.15):
    """
    Show how sample count n affects smoothness.
    Reproduces Figure 4 from the SmoothGrad paper.

    Args:
        pil_image: PIL Image
        grad_fn: smoothgrad function
        class_idx: integer
        sample_counts: list of ints, e.g. [5, 10, 25, 50, 100]
        sigma: fixed noise level while varying n

    Returns:
        matplotlib Figure
    """
    from gradients import preprocess_image

    image_tensor = preprocess_image(pil_image)
    original = pil_to_display_array(pil_image)
    n = len(sample_counts)

    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    fig.patch.set_facecolor("white")
    fig.suptitle("Effect of sample count n on SmoothGrad", fontsize=13)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for i, n_samples in enumerate(sample_counts):
        grad = grad_fn(image_tensor, class_idx, sigma, n_samples)
        saliency = postprocess(grad)
        sal_gray = to_grayscale(saliency)
        axes[i + 1].imshow(sal_gray, cmap="hot")
        axes[i + 1].set_title(f"n = {n_samples}", fontsize=11)
        axes[i + 1].axis("off")

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
#  Discriminativity map  (reproduces Figure 6 of paper)
# ─────────────────────────────────────────────
def render_discriminativity(pil_image, grad1, grad2, label1, label2):
    """
    Visualize which pixels belong to class 1 vs class 2.
    Shows: scale(M1) − scale(M2) on a diverging colormap.
    Red = important for class 1, Blue = important for class 2.

    Args:
        pil_image: PIL Image
        grad1, grad2: numpy (224, 224, 3) saliency arrays for each class
        label1, label2: string names for each class

    Returns:
        matplotlib Figure
    """
    original = pil_to_display_array(pil_image)

    def scale_grad(g):
        g = postprocess(g, use_absolute=True, percentile_cap=True)
        return to_grayscale(g)

    m1 = scale_grad(grad1)
    m2 = scale_grad(grad2)
    diff = m1 - m2   # range [-1, 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Discriminativity: {label1} vs {label2}", fontsize=13)

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(m1, cmap="hot")
    axes[1].set_title(f"Map for: {label1}", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(m2, cmap="hot")
    axes[2].set_title(f"Map for: {label2}", fontsize=11)
    axes[2].axis("off")

    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3].set_title(f"Difference (red={label1}, blue={label2})", fontsize=11)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig
