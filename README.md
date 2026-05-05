# SmoothGrad Visualizer
### Computer Vision End-of-Semester Project
Implementation of: *SmoothGrad: removing noise by adding noise* (Smilkov et al., 2017)

---

## Project Structure

```
smoothgrad_project/
├── app.py              ← Streamlit web app (Person B)
├── gradients.py        ← Model + all gradient methods (Person A)
├── visualize.py        ← Rendering + post-processing (Person B)
├── test_pipeline.py    ← Verification script (run this first)
└── requirements.txt    ← Dependencies
```

---

## Setup Instructions (Windows, CPU)

### Step 1 — Create a virtual environment
Open Command Prompt or PowerShell in the project folder:

```
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies
```
pip install -r requirements.txt
```
This installs PyTorch (CPU version), Streamlit, matplotlib, and Pillow.
Total download: ~700 MB (mostly PyTorch).

### Step 3 — Verify everything works
```
python test_pipeline.py
```
You should see: `✅ All tests passed!`

### Step 4 — Run the app
```
streamlit run app.py
```
Opens automatically in your browser at http://localhost:8501

---

## Model Information

| Property | Value |
|----------|-------|
| Architecture | ResNet18 |
| Pretrained on | ImageNet (1000 classes) |
| Input size | 224 × 224 pixels |
| Parameters | ~11 million |
| Download size | ~45 MB (auto-downloaded on first run) |

**No training needed.** The pretrained weights are downloaded automatically
by torchvision on first run and cached locally.

---

## Implemented Methods

| Method | File | Description |
|--------|------|-------------|
| Vanilla Gradient | gradients.py | Raw ∂Sc/∂x |
| SmoothGrad | gradients.py | Averaged gradient over n noisy samples |
| Integrated Gradients | gradients.py | Accumulated gradient from black baseline |
| Guided Backprop | gradients.py | Modified backprop zeroing negative gradients |

---

## Performance on CPU (Windows)

| Method | Approx. time |
|--------|-------------|
| Vanilla Gradient | < 1 second |
| SmoothGrad (n=50) | 15–30 seconds |
| Integrated Gradients (steps=50) | 10–20 seconds |
| Guided Backprop | < 1 second |

Tip: For live demos, use n=25 for SmoothGrad to halve the wait time with minimal quality loss.

---

## What the Paper Showed (your key talking points)

1. Raw gradients are noisy because ReLU networks have discontinuous derivatives
2. Averaging gradients over n noisy images smooths out local fluctuations
3. Optimal σ is 10–20% of the pixel value range
4. Diminishing returns after n ≈ 50 samples
5. SmoothGrad can be combined with any gradient-based method
