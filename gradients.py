"""
gradients.py  —  Model loading, vanilla gradient, SmoothGrad, Integrated Gradients, Guided Backprop
Model: ResNet18 pretrained on ImageNet (no training needed)
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image


# ─────────────────────────────────────────────
#  ImageNet class labels (top 10 shown in UI)
# ─────────────────────────────────────────────
def get_imagenet_labels():
    """Returns dict {index: label} for all 1000 ImageNet classes."""
    import urllib.request, json
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as r:
            labels = json.loads(r.read().decode())
        return {i: label for i, label in enumerate(labels)}
    except Exception:
        # fallback: just return index strings if offline
        return {i: f"class_{i}" for i in range(1000)}


# ─────────────────────────────────────────────
#  Model setup
# ─────────────────────────────────────────────
def load_model():
    """
    Load pretrained ResNet18.
    - Pretrained on ImageNet (1000 classes)
    - No training needed — weights are downloaded automatically (~45 MB)
    - Set to eval mode so BatchNorm and Dropout behave correctly
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def preprocess_image(pil_image):
    """
    Convert a PIL image to a normalized tensor ready for ResNet18.
    ImageNet normalization values are standard — do not change these.
    Returns: tensor of shape (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean per channel
            std=[0.229, 0.224, 0.225]      # ImageNet std per channel
        )
    ])
    tensor = transform(pil_image.convert("RGB"))  # ensure 3-channel
    return tensor.unsqueeze(0)                     # add batch dim → (1,3,224,224)


def get_top_predictions(model, image_tensor, labels_dict, top_k=5):
    """
    Run a forward pass and return the top-k predicted class names + scores.
    Returns list of (class_index, class_name, confidence_percent)
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, top_k)
    results = []
    for prob, idx in zip(top_probs, top_idxs):
        results.append((
            idx.item(),
            labels_dict.get(idx.item(), f"class_{idx.item()}"),
            round(prob.item() * 100, 2)
        ))
    return results


# ─────────────────────────────────────────────
#  Method 1: Vanilla Gradient
# ─────────────────────────────────────────────
def vanilla_gradient(model, image_tensor, class_idx):
    """
    Compute the gradient of class score w.r.t. input pixels.
    This is the basic sensitivity map: ∂Sc/∂x

    Args:
        model: pretrained ResNet18
        image_tensor: (1, 3, 224, 224) — already preprocessed
        class_idx: integer, the class to explain

    Returns:
        numpy array of shape (224, 224, 3) — raw gradient values
    """
    img = image_tensor.clone().requires_grad_(True)

    logits = model(img)
    score = logits[0, class_idx]   # scalar — score for target class
    model.zero_grad()
    score.backward()               # compute ∂score/∂img

    grad = img.grad.data[0]        # shape (3, 224, 224)
    grad = grad.permute(1, 2, 0).numpy()   # → (224, 224, 3)
    return grad


# ─────────────────────────────────────────────
#  Method 2: SmoothGrad  ← the paper's method
# ─────────────────────────────────────────────
def smoothgrad(model, image_tensor, class_idx, sigma=0.15, n_samples=50):
    """
    SmoothGrad: average gradients over n noisy copies of the image.
    Formula: M̂c(x) = (1/n) Σ Mc(x + N(0, σ²))

    Args:
        model: pretrained ResNet18
        image_tensor: (1, 3, 224, 224)
        class_idx: integer
        sigma: noise level as fraction of pixel range (paper suggests 0.10–0.20)
        n_samples: number of noisy samples (paper finds diminishing return after 50)

    Returns:
        numpy array of shape (224, 224, 3) — averaged gradient (much less noisy)
    """
    # pixel range of normalized images is roughly [-2, 2] per channel
    # sigma is expressed as fraction of this range
    pixel_range = image_tensor.max().item() - image_tensor.min().item()
    noise_std = sigma * pixel_range

    accumulated_grad = np.zeros((224, 224, 3))

    for _ in range(n_samples):
        # add random Gaussian noise
        noise = torch.randn_like(image_tensor) * noise_std
        noisy_img = (image_tensor + noise).clone().requires_grad_(True)

        logits = model(noisy_img)
        score = logits[0, class_idx]
        model.zero_grad()
        score.backward()

        grad = noisy_img.grad.data[0]
        grad = grad.permute(1, 2, 0).numpy()
        accumulated_grad += grad

    return accumulated_grad / n_samples   # average


# ─────────────────────────────────────────────
#  Method 3: Integrated Gradients
# ─────────────────────────────────────────────
def integrated_gradients(model, image_tensor, class_idx, steps=50):
    """
    Integrated Gradients: accumulate gradients along the path from a
    black baseline image to the real image.
    IG(x) = (x − x') × ∫ (∂Sc/∂x) along the interpolation path

    Args:
        model: pretrained ResNet18
        image_tensor: (1, 3, 224, 224)
        class_idx: integer
        steps: number of interpolation steps (50 is standard)

    Returns:
        numpy array of shape (224, 224, 3)
    """
    baseline = torch.zeros_like(image_tensor)   # black image = zero baseline
    accumulated_grad = np.zeros((224, 224, 3))

    for step in range(steps):
        alpha = step / steps
        # interpolated image between baseline (α=0) and original (α=1)
        interpolated = (baseline + alpha * (image_tensor - baseline)).requires_grad_(True)

        logits = model(interpolated)
        score = logits[0, class_idx]
        model.zero_grad()
        score.backward()

        grad = interpolated.grad.data[0]
        grad = grad.permute(1, 2, 0).numpy()
        accumulated_grad += grad

    # multiply by (input − baseline) as per the IG formula
    diff = (image_tensor - baseline)[0].permute(1, 2, 0).detach().numpy()
    ig = accumulated_grad / steps * diff
    return ig


# ─────────────────────────────────────────────
#  Method 4: Guided Backpropagation
# ─────────────────────────────────────────────
class GuidedBackprop:
    """
    Guided Backprop: modifies the backward pass through ReLU layers
    to zero out any negative gradients, producing sharper maps.
    Uses PyTorch backward hooks.
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Attach a hook to every ReLU layer in the model."""
        def relu_hook(module, grad_input, grad_output):
            # zero out negative values in the incoming gradient
            return (torch.clamp(grad_input[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(relu_hook)
                self.hooks.append(hook)

    def compute(self, image_tensor, class_idx):
        """
        Run guided backprop and return the saliency map.
        Returns numpy array of shape (224, 224, 3)
        """
        img = image_tensor.clone().requires_grad_(True)
        logits = self.model(img)
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward()

        grad = img.grad.data[0].permute(1, 2, 0).numpy()
        return grad

    def remove_hooks(self):
        """Always call this after you're done to restore normal backprop."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
