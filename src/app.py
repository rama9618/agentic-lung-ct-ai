from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

import streamlit as st
from dataset import get_transforms
from gradcam import GradCAM, IMAGE_SIZE, apply_colormap, get_device, overlay_heatmap
from model import CLASS_NAMES
from model_factory import create_model


BEST_MODEL_FILE = Path("reports/best_model.txt")
MODEL_COMPARISON_CSV = Path("reports/model_comparison.csv")
MODEL_CHECKPOINTS = {
    "resnet50": Path("models/best_resnet50.pth"),
    "densenet121": Path("models/best_densenet121.pth"),
    "efficientnet_b0": Path("models/best_efficientnet_b0.pth"),
}
MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
}
SURFACE_SIZE = (128, 128)
GRADCAM_THRESHOLD = 0.5
GRADCAM_BLUR_RADIUS = 2.0
GRADCAM_CONTRAST_POWER = 1.6
DEFAULT_PSEUDO_SLICES = 24

DISPLAY_NAMES = {
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large cell carcinoma",
    "normal": "Normal",
    "squamous.cell.carcinoma": "Squamous cell carcinoma",
}

CANCER_CLASSES = {
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
}


@st.cache_resource
def get_trained_model(model_name: str, model_path: str):
    device = get_device()
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(model_name=model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    return model, class_names, device


def read_best_model_name() -> str:
    if BEST_MODEL_FILE.exists():
        model_name = BEST_MODEL_FILE.read_text(encoding="utf-8").strip()
        if model_name in MODEL_CHECKPOINTS:
            return model_name
    return "resnet50"


def get_model_checkpoint_path(model_name: str) -> Path:
    return MODEL_CHECKPOINTS.get(model_name, MODEL_CHECKPOINTS["resnet50"])


def get_model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def get_gradcam_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported Grad-CAM model: {model_name}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = get_transforms(train=False)
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> tuple[int, float]:
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probabilities = F.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(dim=1)
    return int(prediction.item()), float(confidence.item())


def generate_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    model_name: str,
):
    gradcam = GradCAM(model=model, target_layer=get_gradcam_target_layer(model, model_name))
    try:
        heatmap, predicted_class = gradcam.generate(image=image_tensor.to(device))
    finally:
        gradcam.close()
    return heatmap, predicted_class


def load_uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def prepare_surface_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    heatmap_image = Image.fromarray(np.uint8(255 * normalize_array(heatmap)), mode="L")
    resized_heatmap = heatmap_image.resize(size)
    smoothed_heatmap = resized_heatmap.filter(ImageFilter.GaussianBlur(radius=GRADCAM_BLUR_RADIUS))
    normalized_heatmap = np.asarray(smoothed_heatmap, dtype=np.float32) / 255.0
    normalized_heatmap = normalize_array(normalized_heatmap)

    thresholded = np.where(normalized_heatmap > GRADCAM_THRESHOLD, normalized_heatmap, 0.0)
    high_activation = (thresholded - GRADCAM_THRESHOLD) / (1.0 - GRADCAM_THRESHOLD)
    high_activation = np.clip(high_activation, 0.0, 1.0)
    return np.power(high_activation, GRADCAM_CONTRAST_POWER)


def create_pseudo_3d_surface(image: Image.Image, heatmap: np.ndarray) -> go.Figure:
    grayscale = image.convert("L").resize(SURFACE_SIZE)
    intensity = np.asarray(grayscale, dtype=np.float32)
    normalized_heatmap = prepare_surface_heatmap(heatmap, SURFACE_SIZE)

    figure = go.Figure(
        data=[
            go.Surface(
                z=intensity,
                surfacecolor=normalized_heatmap,
                cmin=0.0,
                cmax=1.0,
                colorscale=[
                    [0.0, "blue"],
                    [0.65, "blue"],
                    [0.88, "yellow"],
                    [1.0, "red"],
                ],
                showscale=True,
                colorbar={"title": "Grad-CAM activation"},
            )
        ]
    )
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Intensity",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 0.9}},
            "aspectratio": {"x": 1, "y": 1, "z": 0.35},
        },
    )
    return figure


def create_pseudo_volume(
    image: Image.Image,
    heatmap: np.ndarray,
    num_slices: int = DEFAULT_PSEUDO_SLICES,
) -> np.ndarray:
    grayscale = image.convert("L").resize(SURFACE_SIZE)
    image_slice = np.asarray(grayscale, dtype=np.float32) / 255.0

    heatmap = heatmap.astype(np.float32)
    heatmap_max = float(heatmap.max())
    if heatmap_max > 1e-8:
        heatmap = heatmap / heatmap_max
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)
    heatmap = gaussian_filter(heatmap, sigma=1)
    heatmap = normalize_array(heatmap)
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap), mode="L").resize(SURFACE_SIZE)
    heatmap_slice = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    # Grad-CAM weighted volume enhances tumor regions.
    center_slice = num_slices // 2
    sigma = num_slices / 4
    volume_slices = []
    for slice_index in range(num_slices):
        shift_x = int(2 * np.sin(slice_index / 3))
        shift_y = int(2 * np.cos(slice_index / 3))
        shifted_heatmap = np.roll(heatmap_slice, shift=(shift_x, shift_y), axis=(0, 1))
        depth_weight = np.exp(-((slice_index - center_slice) ** 2) / (2 * sigma**2))
        weighted_slice = image_slice * (0.3 + shifted_heatmap * 2.0) * depth_weight
        volume_slices.append(weighted_slice)

    volume = np.stack(volume_slices)
    volume = gaussian_filter(volume, sigma=1)
    volume[volume < 0.25] = 0
    volume_max = float(volume.max())
    if volume_max > 1e-8:
        volume = volume / volume_max
    return volume.astype(np.float32)


def create_pseudo_volume_figure(volume: np.ndarray) -> go.Figure:
    z_values, y_values, x_values = np.mgrid[
        0 : volume.shape[0],
        0 : volume.shape[1],
        0 : volume.shape[2],
    ]

    figure = go.Figure(
        data=[
            go.Volume(
                x=x_values.flatten(),
                y=y_values.flatten(),
                z=z_values.flatten(),
                value=volume.flatten(),
                colorscale="Turbo",
                isomin=0.2,
                isomax=1.0,
                opacity=0.4,
                opacityscale=[
                    [0.0, 0.0],
                    [0.2, 0.08],
                    [0.45, 0.18],
                    [0.75, 0.32],
                    [1.0, 0.48],
                ],
                surface_count=22,
                colorbar={"title": "Grad-CAM weighted intensity"},
                lighting={"ambient": 0.6, "diffuse": 0.8},
            )
        ]
    )

    rotation_frames = []
    for angle in np.linspace(0, 2 * np.pi, 48, endpoint=False):
        rotation_frames.append(
            go.Frame(
                layout={
                    "scene": {
                        "camera": {
                            "eye": {
                                "x": 1.5 * np.cos(angle),
                                "y": 1.5 * np.sin(angle),
                                "z": 1.2,
                            }
                        }
                    }
                }
            )
        )
    figure.frames = rotation_frames
    figure.update_layout(
        title={
            "text": "3D Tumor Attention Map (Grad-CAM)",
            "x": 0.5,
            "font": {"color": "white", "size": 20},
        },
        height=650,
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
        paper_bgcolor="black",
        plot_bgcolor="black",
        scene={
            "xaxis": {"showgrid": False, "visible": False},
            "yaxis": {"showgrid": False, "visible": False},
            "zaxis": {"showgrid": False, "visible": False},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
            "aspectratio": {"x": 1, "y": 1, "z": 0.45},
            "bgcolor": "black",
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "Rotate",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 120, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                ],
            }
        ],
    )
    return figure


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace(".", " ").title())


def prediction_badge(class_name: str) -> str:
    label = display_name(class_name)
    if class_name == "normal":
        return f'<span class="prediction-badge badge-normal">{label}</span>'
    if class_name in CANCER_CLASSES:
        return f'<span class="prediction-badge badge-alert">{label}</span>'
    return f'<span class="prediction-badge badge-caution">{label}</span>'


def interpretation_text(class_name: str, confidence: float) -> str:
    label = display_name(class_name).lower()
    confidence_text = f"{confidence * 100:.2f}%"
    if class_name == "normal":
        return (
            f"The model found image patterns most consistent with {label} lung tissue "
            f"with {confidence_text} confidence. Grad-CAM highlights the regions that "
            "most influenced this prediction."
        )
    return (
        f"The model found image patterns most consistent with {label} with "
        f"{confidence_text} confidence. Grad-CAM highlights the image regions that "
        "contributed most strongly to this classification."
    )


def class_meaning(class_name: str) -> str:
    meanings = {
        "adenocarcinoma": (
            "Adenocarcinoma is a type of non-small cell lung cancer that often starts "
            "in mucus-producing cells and is commonly found toward the outer regions "
            "of the lung."
        ),
        "large.cell.carcinoma": (
            "Large cell carcinoma is a type of non-small cell lung cancer that can grow "
            "and spread quickly and may appear in different parts of the lung."
        ),
        "normal": (
            "Normal means the model did not identify image patterns that match the tumor "
            "classes it was trained to recognize."
        ),
        "squamous.cell.carcinoma": (
            "Squamous cell carcinoma is a type of non-small cell lung cancer that often "
            "starts in cells lining the airways of the lung."
        ),
    }
    return meanings.get(
        class_name,
        "This class represents one of the lung CT image categories learned by the model.",
    )


def typical_ct_patterns(class_name: str) -> str:
    patterns = {
        "adenocarcinoma": (
            "Adenocarcinoma may appear on CT as a peripheral lung nodule or mass, "
            "sometimes with ground-glass opacity, mixed solid components, irregular "
            "margins, or air bronchogram-like features."
        ),
        "large.cell.carcinoma": (
            "Large cell carcinoma may appear as a larger pulmonary mass with irregular "
            "or bulky morphology. Imaging appearance can overlap with other non-small "
            "cell lung cancers."
        ),
        "normal": (
            "A normal-class prediction means the model did not detect tumor-like visual "
            "patterns from the classes it was trained on in this uploaded slice."
        ),
        "squamous.cell.carcinoma": (
            "Squamous cell carcinoma is often associated with central airway-adjacent "
            "lesions and may show mass-like opacity, airway involvement, or cavitation "
            "in some cases."
        ),
    }
    return patterns.get(
        class_name,
        "CT appearance should be interpreted by a radiologist in the context of the full study.",
    )


def confidence_interpretation(confidence: float) -> str:
    if confidence > 0.90:
        return "High confidence: the model assigned more than 90% probability to this class."
    if confidence >= 0.70:
        return "Moderate confidence: the model assigned 70-90% probability to this class."
    return "Lower confidence: the model assigned less than 70% probability to this class."


def generate_ai_report(predicted_class: str, confidence: float) -> dict[str, str]:
    label = display_name(predicted_class)
    risk_level = "Low" if predicted_class == "normal" else "High"
    summary = (
        f"The ResNet50 model classified the uploaded CT image as {label} "
        f"with {confidence * 100:.2f}% confidence."
    )
    model_reasoning = (
        "Model reasoning is based on the Grad-CAM activation map. Regions with warmer "
        "colors, especially red in the pseudo-3D view, represent image areas that most "
        "strongly influenced the predicted class."
    )

    return {
        "summary": summary,
        "class_meaning": class_meaning(predicted_class),
        "typical_ct_patterns": typical_ct_patterns(predicted_class),
        "confidence_interpretation": confidence_interpretation(confidence),
        "model_reasoning": model_reasoning,
        "risk_level": risk_level,
        "suggested_next_step": "Further radiological review recommended.",
        "disclaimer": (
            "This system is a research prototype and does not provide a medical diagnosis. "
            "Please consult a qualified radiologist or physician."
        ),
    }


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f7fb;
            color: #1f2937;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-left: 6px solid #2563eb;
            border-radius: 8px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            color: #0f172a;
            font-size: 2.15rem;
            letter-spacing: 0;
            line-height: 1.15;
        }
        .hero p {
            margin: 0;
            color: #475569;
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .section-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0 0 0.75rem 0;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border-color: #e5e7eb;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
        }
        .prediction-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            line-height: 1;
            border: 1px solid transparent;
        }
        .badge-normal {
            color: #065f46;
            background: #d1fae5;
            border-color: #a7f3d0;
        }
        .badge-alert {
            color: #991b1b;
            background: #fee2e2;
            border-color: #fecaca;
        }
        .badge-caution {
            color: #9a3412;
            background: #ffedd5;
            border-color: #fed7aa;
        }
        .confidence-value {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 800;
            margin: 0.45rem 0 0.15rem 0;
        }
        .muted {
            color: #64748b;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .report {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            margin-top: 1.2rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
        }
        .report-row {
            border-bottom: 1px solid #edf2f7;
            padding: 0.7rem 0;
        }
        .report-row:last-child {
            border-bottom: 0;
        }
        .report-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.18rem;
        }
        .report-value {
            color: #0f172a;
            font-size: 0.98rem;
            line-height: 1.5;
        }
        .risk-low {
            color: #065f46;
            background: #d1fae5;
            border: 1px solid #a7f3d0;
            border-radius: 999px;
            display: inline-flex;
            font-weight: 800;
            padding: 0.35rem 0.7rem;
        }
        .risk-high {
            color: #991b1b;
            background: #fee2e2;
            border: 1px solid #fecaca;
            border-radius: 999px;
            display: inline-flex;
            font-weight: 800;
            padding: 0.35rem 0.7rem;
        }
        .disclaimer {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-radius: 8px;
            color: #7c2d12;
            line-height: 1.5;
            margin-top: 0.8rem;
            padding: 0.85rem 1rem;
        }
        textarea, input {
            background-color: #1e1e1e !important;
            color: white !important;
        }
        textarea::placeholder {
            color: #aaaaaa !important;
        }
        .stChatMessage {
            background-color: #2b2b2b !important;
            color: white !important;
            border-radius: 10px;
            padding: 10px;
        }
        [data-testid="stChatMessageContent"] {
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_model_comparison_rows() -> list[dict[str, str]]:
    if not MODEL_COMPARISON_CSV.exists():
        return []
    with MODEL_COMPARISON_CSV.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def render_model_performance_card(active_model_name: str) -> None:
    rows = read_model_comparison_rows()
    active_row = next(
        (row for row in rows if row.get("model_name") == active_model_name),
        None,
    )
    if active_row is None or active_row.get("status") != "evaluated":
        st.info("Model comparison metrics are not available yet.")
        return

    st.markdown("**Model Performance**")
    metric_cols = st.columns(2)
    metric_cols[0].metric("Accuracy", f"{float(active_row['accuracy']) * 100:.2f}%")
    metric_cols[1].metric("Weighted F1", f"{float(active_row['f1_score']) * 100:.2f}%")
    st.caption(
        f"Precision: {float(active_row['precision']) * 100:.2f}% | "
        f"Recall: {float(active_row['recall']) * 100:.2f}% | "
        f"Loss: {float(active_row['test_loss']):.4f}"
    )


def render_sidebar(active_model_name: str) -> None:
    with st.sidebar:
        st.title("Project Info")
        st.markdown("**Task:** Lung CT tumor classification")
        st.markdown(f"**Active Model:** {get_model_display_name(active_model_name)}")
        st.markdown(f"**Checkpoint:** `{get_model_checkpoint_path(active_model_name)}`")
        render_model_performance_card(active_model_name)
        st.markdown("**Classes**")
        st.markdown(
            """
            - Adenocarcinoma
            - Large cell carcinoma
            - Normal
            - Squamous cell carcinoma
            """
        )
        st.warning("This is a research prototype, not a medical diagnosis.")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Agentic AI Lung CT Analysis</h1>
            <p>Deep learning-based lung tumor classification with explainable Grad-CAM visualization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_report(predicted_class: str, confidence: float) -> None:
    ai_report = generate_ai_report(predicted_class, confidence)
    risk_class = "risk-low" if ai_report["risk_level"] == "Low" else "risk-high"
    st.markdown(
        f"""
        <div class="report">
            <div class="section-title">AI Clinical Support Note</div>
            <div class="report-row">
                <div class="report-label">Model Result</div>
                <div class="report-value">{ai_report["summary"]}</div>
            </div>
            <div class="report-row">
                <div class="report-label">Clinical-style Explanation</div>
                <div class="report-value">{ai_report["class_meaning"]}</div>
            </div>
            <div class="report-row">
                <div class="report-label">Typical CT Patterns</div>
                <div class="report-value">{ai_report["typical_ct_patterns"]}</div>
            </div>
            <div class="report-row">
                <div class="report-label">Confidence Interpretation</div>
                <div class="report-value">{ai_report["confidence_interpretation"]}</div>
            </div>
            <div class="report-row">
                <div class="report-label">Model Reasoning</div>
                <div class="report-value">{ai_report["model_reasoning"]}</div>
            </div>
            <div class="report-row">
                <div class="report-label">Risk Level</div>
                <div class="report-value"><span class="{risk_class}">{ai_report["risk_level"]}</span></div>
            </div>
            <div class="report-row">
                <div class="report-label">Suggested Next Step</div>
                <div class="report-value">{ai_report["suggested_next_step"]}</div>
            </div>
            <div class="disclaimer">{ai_report["disclaimer"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pseudo_3d_visualization(image: Image.Image, heatmap: np.ndarray) -> None:
    figure = create_pseudo_3d_surface(image, heatmap)
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Pseudo-3D Tumor Highlight (Grad-CAM Guided)</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(figure, use_container_width=True)
        st.info(
            "Red regions indicate areas where the model focuses for prediction."
        )


def render_pseudo_volume_visualization(image: Image.Image, heatmap: np.ndarray) -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Multi-Slice Pseudo-3D Tumor Attention Volume</div>',
            unsafe_allow_html=True,
        )
        num_slices = st.slider(
            "Number of pseudo-slices",
            min_value=8,
            max_value=40,
            value=DEFAULT_PSEUDO_SLICES,
            step=1,
        )
        volume = create_pseudo_volume(
            image=image,
            heatmap=heatmap,
            num_slices=num_slices,
        )
        figure = create_pseudo_volume_figure(volume)
        st.plotly_chart(figure, use_container_width=True)
        st.caption("High-activation regions (red) indicate model attention")
        st.warning(
            "This is a pseudo-3D visualization generated from a single 2D CT image "
            "and Grad-CAM attention. It is not a true CT volume reconstruction."
        )


def detect_user_intent(question: str) -> str:
    normalized = question.lower().strip()
    if normalized in {"hi", "hello", "hey"}:
        return "greeting"
    if any(phrase in normalized for phrase in ("explain everything", "full report", "summarize result")):
        return "full_report"
    if any(phrase in normalized for phrase in ("what is in this image", "what is in the picture")):
        return "image_question"
    if any(phrase in normalized for phrase in ("what is the result", "prediction", "which class")):
        return "prediction"
    if any(phrase in normalized for phrase in ("confidence", "how sure")):
        return "confidence"
    if any(phrase in normalized for phrase in ("gradcam", "grad-cam", "heatmap", "red area", "highlight")):
        return "gradcam"
    if any(phrase in normalized for phrase in ("3d", "3-d", "pseudo", "volume", "visualization")):
        return "pseudo_3d"
    if any(
        phrase in normalized
        for phrase in ("risk", "danger", "serious", "do i have cancer", "am i safe", "can i ignore this")
    ):
        return "risk"
    if any(phrase in normalized for phrase in ("next step", "what should i do", "doctor")):
        return "next_steps"
    if any(phrase in normalized for phrase in ("model", "resnet", "densenet", "efficientnet", "accuracy", "f1")):
        return "model_info"
    if any(
        phrase in normalized
        for phrase in ("explain adenocarcinoma", "what is lung cancer", "medical term")
    ):
        return "medical_term"
    if any(
        phrase in normalized
        for phrase in (
            "what is adenocarcinoma",
            "what is squamous cell carcinoma",
            "what is large cell carcinoma",
            "what does non-small cell mean",
            "what is non-small cell",
        )
    ):
        return "medical_term"
    return "general"


def detect_follow_up(question: str, chat_history: list[dict[str, str]]) -> bool:
    normalized = question.lower().strip().rstrip("?")
    exact_follow_ups = {
        "explain more",
        "what does that mean",
        "tell me more",
        "why",
        "how",
        "is it serious",
        "what should i do",
    }
    follow_up_phrases = {
        "explain more",
        "what does that mean",
        "tell me more",
    }
    has_previous_assistant_message = any(
        message.get("role") == "assistant" for message in chat_history
    )
    return has_previous_assistant_message and (
        normalized in exact_follow_ups
        or any(phrase in normalized for phrase in follow_up_phrases)
    )


def suggested_followups() -> str:
    return (
        "\n\nSuggested questions: What does the red region mean? | "
        "How confident is the model? | What should I do next? | "
        "Explain adenocarcinoma simply."
    )


def medical_term_response(question: str, predicted_class: str) -> str:
    normalized = question.lower()
    if "squamous" in normalized:
        return (
            "Squamous cell carcinoma is a type of non-small cell lung cancer that often starts "
            "in the cells lining the airways. In simple terms, it is one category of lung cancer "
            "that doctors evaluate using imaging, clinical history, and sometimes biopsy."
        )
    if "large cell" in normalized:
        return (
            "Large cell carcinoma is a type of non-small cell lung cancer. It is called 'large cell' "
            "because of how the cancer cells can look under a microscope, not because the scan alone "
            "can confirm it."
        )
    if "non-small cell" in normalized or "lung cancer" in normalized:
        return (
            "Non-small cell lung cancer is a broad group of lung cancers that includes adenocarcinoma, "
            "squamous cell carcinoma, and large cell carcinoma. The term helps doctors group cancers "
            "that may behave and be treated differently from small cell lung cancer."
        )
    if "adenocarcinoma" in normalized:
        return class_meaning("adenocarcinoma")
    return class_meaning(predicted_class)


def last_assistant_message(chat_history: list[dict[str, str]]) -> str:
    for message in reversed(chat_history):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def full_report_response(predicted_class: str, confidence: float) -> str:
    class_label = display_name(predicted_class)
    confidence_text = f"{confidence * 100:.2f}%"
    risk_level = "Low" if predicted_class == "normal" else "High"
    return (
        f"**Model Result:** The model predicts **{class_label}** for the uploaded CT image.\n\n"
        f"**Confidence:** {confidence_text}. {confidence_interpretation(confidence)}\n\n"
        f"**What It Means:** {class_meaning(predicted_class)}\n\n"
        "**Grad-CAM Explanation:** The heatmap shows which image regions most influenced the model. "
        "Red or warmer areas mean stronger model attention, not confirmed tumor boundaries.\n\n"
        "**3D Visualization Explanation:** The pseudo-3D view uses the same single CT image and Grad-CAM "
        "attention to create an illustrative attention volume. It is not a true CT volume reconstruction.\n\n"
        f"**Risk Interpretation:** The app labels this as **{risk_level}** based on the model class, "
        "but this is not a diagnosis.\n\n"
        "**Recommended Next Steps:** Further radiological review recommended. A clinician may compare prior "
        "scans, review the full CT study, and decide whether follow-up imaging or testing is needed.\n\n"
        "**Safety Note:** This system cannot diagnose or rule out disease. Please consult a qualified radiologist "
        "or physician."
    )


def generate_agent_response(
    question: str,
    predicted_class: str,
    confidence: float,
    active_model: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    chat_history = chat_history or []
    intent = "follow_up" if detect_follow_up(question, chat_history) else detect_user_intent(question)
    class_label = display_name(predicted_class)
    confidence_text = f"{confidence * 100:.2f}%"
    model_label = get_model_display_name(active_model) if active_model else "the active model"

    if intent == "greeting":
        response = (
            "Hi! I’m your AI medical assistant. You can ask me about the CT image, "
            "prediction, confidence, Grad-CAM heatmap, 3D visualization, or recommended next steps."
        )
        return response + suggested_followups()
    if intent == "full_report":
        return full_report_response(predicted_class, confidence) + suggested_followups()
    if intent in {"image_question", "prediction"}:
        response = (
            f"The model predicts {class_label} for the uploaded CT image with {confidence_text} confidence. "
            f"In simple terms: {class_meaning(predicted_class)} This is a research model output, not a diagnosis."
        )
        return response + suggested_followups()
    if intent == "confidence":
        response = (
            f"The confidence score is {confidence_text}. {confidence_interpretation(confidence)} "
            "Confidence reflects the model's probability for its selected class, not clinical certainty."
        )
        return response + suggested_followups()
    if intent == "gradcam":
        response = (
            "Grad-CAM shows where the model focused when making its prediction. Warmer or red regions indicate "
            "stronger model attention, including in the 3D visualization, but these highlights are not confirmed "
            "tumor boundaries."
        )
        return response + suggested_followups()
    if intent == "pseudo_3d":
        response = (
            "The 3D view is an illustrative pseudo-volume made from one 2D CT image plus Grad-CAM attention. "
            "Red regions show where the model focused most strongly. It is useful for visual explanation, but "
            "it is not a true CT volume reconstruction."
        )
        return response + suggested_followups()
    if intent == "risk":
        response = (
            f"I cannot diagnose cancer. The model predicts {class_label} with {confidence_text} confidence, "
            "but this must be reviewed by a qualified radiologist or physician."
        )
        return response + suggested_followups()
    if intent == "next_steps":
        response = (
            "The recommended next step is clinical review by a qualified radiologist or physician. "
            "They may compare prior scans, review the full CT study, consider symptoms, and decide whether "
            "follow-up imaging or additional testing is needed."
        )
        return response + suggested_followups()
    if intent == "model_info":
        response = (
            f"The active model is {model_label}. The sidebar shows available comparison metrics such as "
            "accuracy and weighted F1 when `reports/model_comparison.csv` is available."
        )
        return response + suggested_followups()
    if intent == "medical_term":
        response = (
            f"{medical_term_response(question, predicted_class)} Typical imaging patterns can vary, and papers "
            "should be interpreted as general research context rather than patient-specific advice."
        )
        return response + suggested_followups()
    if intent == "follow_up":
        previous = last_assistant_message(chat_history).lower()
        if "grad-cam" in previous or "heatmap" in previous or "red" in previous:
            response = (
                "In plain language, the colored area is the part of the image the model paid most attention to. "
                "It helps explain the model, but it does not prove that the highlighted area is cancer."
            )
        elif "confidence" in previous:
            response = (
                f"It means the model assigned {confidence_text} probability to {class_label}. "
                "That is model certainty, not clinical certainty."
            )
        elif "next step" in previous or "doctor" in previous or "radiologist" in previous:
            response = (
                "The practical next step is to have a radiologist or physician review the image in the context "
                "of the full scan, symptoms, and medical history."
            )
        else:
            response = (
                f"The key point is that the model output is {class_label}, but it is only decision support. "
                "A clinician should review it before any medical conclusion is made."
            )
        return response + suggested_followups()

    response = (
        "I can help with this CT analysis, but I need a more specific question. For example, ask: "
        "'What is the prediction?', 'How confident is the model?', 'What does the heatmap mean?', "
        "'Is this serious?', or 'What should I do next?'"
    )
    return response + suggested_followups()


def generate_chat_response(
    question: str,
    predicted_class: str,
    confidence: float,
    active_model: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    return generate_agent_response(
        question,
        predicted_class,
        confidence,
        active_model,
        chat_history,
    )


def render_chatbot(predicted_class: str, confidence: float, active_model: str | None = None) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">AI Medical Assistant</div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "content": (
                        "Ask me about the model result, confidence, Grad-CAM explanation, "
                        "risk level, or recommended next steps."
                    ),
                }
            ]

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask about this CT analysis")
        if user_question:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_question}
            )
            assistant_response = generate_chat_response(
                question=user_question,
                predicted_class=predicted_class,
                confidence=confidence,
                active_model=active_model,
                chat_history=st.session_state.chat_history,
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Agentic AI Lung CT Analysis", layout="wide")
    active_model_name = read_best_model_name()
    active_model_path = get_model_checkpoint_path(active_model_name)
    render_styles()
    render_sidebar(active_model_name)
    render_hero()

    with st.container(border=True):
        st.markdown('<div class="section-title">Upload CT Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a CT image file",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            label_visibility="collapsed",
        )

    if uploaded_file is None:
        st.info("Upload a CT image to run prediction and Grad-CAM analysis.")
        return

    try:
        model, class_names, device = get_trained_model(
            active_model_name,
            str(active_model_path),
        )
    except FileNotFoundError as error:
        st.error(str(error))
        return

    original_image = load_uploaded_image(uploaded_file)
    display_image = original_image.resize(IMAGE_SIZE)
    image_tensor = preprocess_image(original_image)

    predicted_index, confidence = predict(
        model=model,
        image_tensor=image_tensor,
        device=device,
    )
    heatmap, gradcam_class = generate_heatmap(
        model=model,
        image_tensor=image_tensor,
        device=device,
        model_name=active_model_name,
    )

    heatmap_image = apply_colormap(heatmap)
    overlay_image = overlay_heatmap(display_image, heatmap)
    predicted_class = class_names[predicted_index]
    interpretation = interpretation_text(predicted_class, confidence)

    image_col, prediction_col, explanation_col = st.columns([1.1, 1, 1.25])

    with image_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Uploaded CT Image</div>', unsafe_allow_html=True)
            st.image(display_image, caption="Original image", use_container_width=True)

    with prediction_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(prediction_badge(predicted_class), unsafe_allow_html=True)
            st.markdown(f'<div class="confidence-value">{confidence * 100:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted">Model confidence</div>', unsafe_allow_html=True)
            st.progress(confidence)

    with explanation_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">AI Explanation</div>', unsafe_allow_html=True)
            st.image(heatmap_image, caption="Grad-CAM heatmap", use_container_width=True)
            st.image(overlay_image, caption="Heatmap overlay", use_container_width=True)
            if gradcam_class != predicted_index:
                st.caption(f"Grad-CAM generated for: {display_name(class_names[gradcam_class])}")
            st.markdown(f'<div class="muted">{interpretation}</div>', unsafe_allow_html=True)

    render_report(
        predicted_class=predicted_class,
        confidence=confidence,
    )
    render_pseudo_3d_visualization(original_image, heatmap)
    render_pseudo_volume_visualization(original_image, heatmap)
    render_chatbot(predicted_class, confidence, active_model_name)


if __name__ == "__main__":
    main()
