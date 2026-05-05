"""
app.py  —  Main Streamlit app
Run with:  streamlit run app.py

This is the interactive web UI for the SmoothGrad visualizer.
Person B owns this file. Person A owns gradients.py.
"""

import io
import streamlit as st
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit

from gradients import (
    load_model,
    preprocess_image,
    get_imagenet_labels,
    get_top_predictions,
    vanilla_gradient,
    smoothgrad,
    integrated_gradients,
    GuidedBackprop,
)
from visualize import (
    render_comparison,
    render_noise_grid,
    render_sample_grid,
    render_discriminativity,
)


def fig_to_png(fig):
    """Render a matplotlib figure to a PNG bytes buffer for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmoothGrad Visualizer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 SmoothGrad Visualizer")
st.caption("Interactive implementation of *SmoothGrad: removing noise by adding noise* (Smilkov et al., 2017)")


# ─────────────────────────────────────────────
#  Load model & labels once (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    """Cache the model and labels so they don't reload on every interaction."""
    with st.spinner("Loading ResNet18 pretrained model..."):
        model = load_model()
        labels = get_imagenet_labels()
    return model, labels

model, labels_dict = load_resources()


# ─────────────────────────────────────────────
#  Sidebar — controls
# ─────────────────────────────────────────────
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Best results with clear photos of animals, objects, or vehicles on simple backgrounds."
)

st.sidebar.markdown("---")
st.sidebar.subheader("SmoothGrad parameters")

sigma = st.sidebar.slider(
    "Noise level σ",
    min_value=0.0,
    max_value=0.30,
    value=0.15,
    step=0.01,
    help="Standard deviation of Gaussian noise added to the image. Paper recommends 0.10–0.20."
)

n_samples = st.sidebar.slider(
    "Sample count n",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
    help="Number of noisy copies to average over. More = smoother but slower. Diminishing return after 50."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Visualization options")

use_absolute = st.sidebar.checkbox("Take absolute value of gradients", value=True,
    help="Recommended for ImageNet. Shows magnitude of influence, not direction.")
percentile_cap = st.sidebar.checkbox("Cap outliers at 99th percentile", value=True,
    help="Prevents bright outlier pixels from washing out the rest of the map.")
multiply_input = st.sidebar.checkbox("Multiply gradient × input image", value=False,
    help="Highlights contribution (pixel value × sensitivity). Can be misleading for dark pixels.")


# ─────────────────────────────────────────────
#  Main content
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload an image from the sidebar to get started.")
    st.markdown("""
    **What this app shows:**
    - **Vanilla Gradient** — raw ∂Sc/∂x, often very noisy
    - **SmoothGrad** — averaged gradient over n noisy copies (the paper's method)
    - **Integrated Gradients** — accumulated gradients from a black baseline
    - **Guided Backprop** — modified backprop that zeros negative gradients

    Use the sliders to adjust σ (noise level) and n (sample count) and see how they affect map quality.
    """)
    st.stop()


# ─────────────────────────────────────────────
#  Process uploaded image
# ─────────────────────────────────────────────
pil_image = Image.open(uploaded_file).convert("RGB")
image_tensor = preprocess_image(pil_image)

# show original + top predictions
col1, col2 = st.columns([1, 2])
with col1:
    st.image(pil_image, caption="Uploaded image", use_column_width=True)

with col2:
    st.subheader("Top predictions (ResNet18)")
    predictions = get_top_predictions(model, image_tensor, labels_dict, top_k=5)
    for rank, (idx, name, conf) in enumerate(predictions):
        bar_color = "#1a73e8" if rank == 0 else "#a8c7fa"
        st.markdown(
            f"**{name}** &nbsp; `{conf}%` &nbsp; {'🏆' if rank == 0 else ''}",
            unsafe_allow_html=True
        )
        st.progress(conf / 100)

    top_class_idx = predictions[0][0]
    top_class_name = predictions[0][1]
    st.success(f"Explaining prediction: **{top_class_name}** (class {top_class_idx})")


# ─────────────────────────────────────────────
#  Tabs for different views
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Method Comparison",
    "🔬 Noise Level Study",
    "📈 Sample Size Study",
    "🎯 Discriminativity"
])


# ── Tab 1: Side-by-side comparison of all methods ──
with tab1:
    st.subheader("All 4 methods compared side by side")
    st.caption("Same image, same class, same post-processing options applied to all.")

    if st.button("▶ Run all methods", key="run_comparison"):
        grads = {}

        with st.spinner("Computing Vanilla Gradient..."):
            grads["Vanilla"] = vanilla_gradient(model, image_tensor, top_class_idx)

        with st.spinner(f"Computing SmoothGrad (σ={sigma}, n={n_samples})... this takes ~10–20s on CPU"):
            grads["SmoothGrad"] = smoothgrad(model, image_tensor, top_class_idx, sigma, n_samples)

        with st.spinner("Computing Integrated Gradients (50 steps)..."):
            grads["Integrated\nGradients"] = integrated_gradients(model, image_tensor, top_class_idx)

        with st.spinner("Computing Guided Backprop..."):
            gbp = GuidedBackprop(model)
            grads["Guided\nBackprop"] = gbp.compute(image_tensor, top_class_idx)
            gbp.remove_hooks()

        fig = render_comparison(
            pil_image, grads,
            use_absolute=use_absolute,
            percentile_cap=percentile_cap,
            multiply_input=multiply_input
        )
        st.pyplot(fig)
        st.download_button("⬇ Save PNG", data=fig_to_png(fig),
                           file_name="comparison_all_methods.png", mime="image/png")
        st.caption("Lighter pixels = higher importance to the model's decision.")


# ── Tab 2: Noise level study (Figure 3 from paper) ──
with tab2:
    st.subheader("How noise level σ affects SmoothGrad quality")
    st.caption("Reproduces Figure 3 from the paper. Fixed n=50, varying σ.")

    if st.button("▶ Run noise study", key="run_noise"):
        sigmas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

        def sg_fn(tensor, cls_idx, sig, n):
            if sig == 0.0:
                return vanilla_gradient(model, tensor, cls_idx)
            return smoothgrad(model, tensor, cls_idx, sig, n)

        progress = st.progress(0)
        grads_by_sigma = {}

        for i, s in enumerate(sigmas):
            label = "Vanilla (σ=0)" if s == 0.0 else f"σ={int(s*100)}%"
            with st.spinner(f"Computing {label}..."):
                grads_by_sigma[label] = sg_fn(image_tensor, top_class_idx, s, 50)
            progress.progress((i + 1) / len(sigmas))

        from visualize import render_comparison
        fig = render_comparison(pil_image, grads_by_sigma,
                                use_absolute=use_absolute, percentile_cap=percentile_cap)
        st.pyplot(fig)
        st.download_button("⬇ Save PNG", data=fig_to_png(fig),
                           file_name="figure3_noise_levels.png", mime="image/png")
        st.caption("Notice how maps become cleaner with moderate noise, then degrade at very high σ.")


# ── Tab 3: Sample size study (Figure 4 from paper) ──
with tab3:
    st.subheader("How sample count n affects SmoothGrad smoothness")
    st.caption(f"Reproduces Figure 4 from the paper. Fixed σ={sigma}, varying n.")

    if st.button("▶ Run sample study", key="run_samples"):
        sample_counts = [5, 10, 25, 50, 100]
        grads_by_n = {}

        progress = st.progress(0)
        for i, n in enumerate(sample_counts):
            with st.spinner(f"Computing SmoothGrad n={n}..."):
                grads_by_n[f"n={n}"] = smoothgrad(model, image_tensor, top_class_idx, sigma, n)
            progress.progress((i + 1) / len(sample_counts))

        from visualize import render_comparison
        fig = render_comparison(pil_image, grads_by_n,
                                use_absolute=use_absolute, percentile_cap=percentile_cap)
        st.pyplot(fig)
        st.download_button("⬇ Save PNG", data=fig_to_png(fig),
                           file_name="figure4_sample_counts.png", mime="image/png")
        st.caption("Maps get smoother as n increases, with diminishing returns after n≈50.")


# ── Tab 4: Discriminativity (Figure 6 from paper) ──
with tab4:
    st.subheader("Discriminativity: can the map tell two classes apart?")
    st.caption("Reproduces Figure 6 from the paper. Red = supports class 1, Blue = supports class 2.")

    st.markdown("Choose two classes from the top predictions to compare:")

    available = [(name, idx) for idx, name, _ in predictions]
    choice1 = st.selectbox("Class 1 (red)", options=[n for n, _ in available], index=0)
    choice2 = st.selectbox("Class 2 (blue)", options=[n for n, _ in available], index=1 if len(available) > 1 else 0)

    if choice1 == choice2:
        st.warning("Please select two different classes.")
    elif st.button("▶ Run discriminativity", key="run_discrim"):
        idx1 = dict(available)[choice1]
        idx2 = dict(available)[choice2]

        with st.spinner(f"Computing SmoothGrad for {choice1}..."):
            grad1 = smoothgrad(model, image_tensor, idx1, sigma, n_samples)
        with st.spinner(f"Computing SmoothGrad for {choice2}..."):
            grad2 = smoothgrad(model, image_tensor, idx2, sigma, n_samples)

        fig = render_discriminativity(pil_image, grad1, grad2, choice1, choice2)
        st.pyplot(fig)
        st.download_button("⬇ Save PNG", data=fig_to_png(fig),
                           file_name="figure6_discriminativity.png", mime="image/png")
        st.caption("A good method should highlight different regions for different classes.")
