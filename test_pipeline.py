"""
test_pipeline.py — Run this FIRST to verify your setup works.
It tests model loading and all 4 gradient methods on a synthetic image.
No UI — just prints results to terminal.

Run with:  python test_pipeline.py
"""

import torch
import numpy as np
from PIL import Image
from gradients import (
    load_model,
    preprocess_image,
    get_top_predictions,
    get_imagenet_labels,
    vanilla_gradient,
    smoothgrad,
    integrated_gradients,
    GuidedBackprop,
)

print("=" * 50)
print("SmoothGrad Pipeline Test")
print("=" * 50)

# 1. Load model
print("\n[1/5] Loading ResNet18...")
model = load_model()
print("  ✓ Model loaded. Parameters:", sum(p.numel() for p in model.parameters()), "weights")

# 2. Create a test image (random noise — just to test shapes)
print("\n[2/5] Creating synthetic test image...")
test_img = Image.fromarray(
    np.random.randint(100, 200, (300, 400, 3), dtype=np.uint8)
)
tensor = preprocess_image(test_img)
print(f"  ✓ Image tensor shape: {tensor.shape}")   # should be (1, 3, 224, 224)

# 3. Get predictions
print("\n[3/5] Running inference...")
labels = get_imagenet_labels()
preds = get_top_predictions(model, tensor, labels, top_k=3)
top_class_idx = preds[0][0]
for idx, name, conf in preds:
    print(f"  {name}: {conf}%")

# 4. Test all gradient methods
print("\n[4/5] Testing gradient methods...")

print("  → Vanilla Gradient...", end=" ", flush=True)
vg = vanilla_gradient(model, tensor, top_class_idx)
assert vg.shape == (224, 224, 3), f"Expected (224,224,3), got {vg.shape}"
print(f"✓  shape={vg.shape}, range=[{vg.min():.4f}, {vg.max():.4f}]")

print("  → SmoothGrad (n=5 for speed)...", end=" ", flush=True)
sg = smoothgrad(model, tensor, top_class_idx, sigma=0.15, n_samples=5)
assert sg.shape == (224, 224, 3)
print(f"✓  shape={sg.shape}, range=[{sg.min():.4f}, {sg.max():.4f}]")

print("  → Integrated Gradients (steps=10 for speed)...", end=" ", flush=True)
ig = integrated_gradients(model, tensor, top_class_idx, steps=10)
assert ig.shape == (224, 224, 3)
print(f"✓  shape={ig.shape}, range=[{ig.min():.4f}, {ig.max():.4f}]")

print("  → Guided Backprop...", end=" ", flush=True)
gbp = GuidedBackprop(model)
gb = gbp.compute(tensor, top_class_idx)
gbp.remove_hooks()
assert gb.shape == (224, 224, 3)
print(f"✓  shape={gb.shape}, range=[{gb.min():.4f}, {gb.max():.4f}]")

# 5. Smoke test visualization
print("\n[5/5] Testing visualization rendering...")
from visualize import postprocess, to_grayscale, render_comparison
processed = postprocess(vg)
gray = to_grayscale(processed)
assert gray.shape == (224, 224)
print(f"  ✓ postprocess output shape: {processed.shape}")
print(f"  ✓ grayscale output shape: {gray.shape}")

print("\n" + "=" * 50)
print("✅ All tests passed! Run the app with:")
print("   streamlit run app.py")
print("=" * 50)
