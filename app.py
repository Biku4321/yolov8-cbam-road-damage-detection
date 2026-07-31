"""
Road Damage Detection — Streamlit App
Baseline YOLOv8n vs CBAM (Dual) vs CBAM (Single) — interactive comparison tool.

Run locally with:
    streamlit run app.py
"""

import io
import os
import time

import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG — point these at your trained weight files
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "Baseline (YOLOv8n)": "weights/baseline_best.pt",
    "CBAM Dual (2 blocks)": "weights/cbam_dual_best.pt",
    "CBAM Single (before SPPF)": "weights/cbam_single_best.pt",
}

# Verified benchmark figures from the project report (validation split, RDD2022)
BENCHMARKS = {
    "Baseline (YOLOv8n)":        {"P": 0.570, "R": 0.510, "mAP50": 0.521, "mAP5095": 0.260, "FPS": 33.56, "Params": "3.01M"},
    "CBAM Dual (2 blocks)":      {"P": 0.558, "R": 0.480, "mAP50": 0.489, "mAP5095": 0.240, "FPS": 37.92, "Params": "3.09M"},
    "CBAM Single (before SPPF)": {"P": 0.559, "R": 0.483, "mAP50": 0.495, "mAP5095": 0.241, "FPS": 38.90, "Params": "3.08M"},
}

CLASS_COLORS = {
    "Longitudinal": "#F2B705",
    "Transverse": "#5DA9E9",
    "Alligator": "#9B7EDE",
    "Pothole": "#FF5A5F",
}

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — asphalt / road-signage design system
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --asphalt-950: #101317;
        --asphalt-900: #15181D;
        --asphalt-800: #1E2228;
        --asphalt-700: #2A2F37;
        --line-white:  #EDEFF2;
        --line-dim:    #9AA1AC;
        --amber:       #F2B705;
        --amber-dim:   #C99C08;
        --hazard:      #FF5A5F;
        --teal:        #4CC9A7;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: var(--line-white);
    }

    .stApp {
        background: var(--asphalt-950);
    }

    h1, h2, h3 {
        font-family: 'Oswald', sans-serif !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    /* ---- Hero ---- */
    .hero {
        padding: 1.6rem 1.8rem 1.4rem 1.8rem;
        background: linear-gradient(135deg, var(--asphalt-900) 0%, var(--asphalt-800) 100%);
        border: 1px solid var(--asphalt-700);
        border-radius: 10px;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 4px;
        background-image: repeating-linear-gradient(
            90deg, var(--amber) 0px, var(--amber) 28px, transparent 28px, transparent 52px
        );
        opacity: 0.85;
    }
    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: var(--amber);
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .hero-title {
        font-family: 'Oswald', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: var(--line-white);
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--line-dim);
        max-width: 640px;
        margin: 0;
    }
    .badge-row { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        border: 1px solid var(--asphalt-700);
        background: var(--asphalt-950);
        color: var(--line-dim);
    }

    /* ---- Cards ---- */
    .panel {
        background: var(--asphalt-900);
        border: 1px solid var(--asphalt-700);
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }
    .panel-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--line-dim);
        margin-bottom: 0.6rem;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: var(--asphalt-900);
        border: 1px solid var(--asphalt-700);
        border-left: 3px solid var(--amber);
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .metric-card .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--line-dim);
    }
    .metric-card .value {
        font-family: 'Oswald', sans-serif;
        font-size: 1.7rem;
        font-weight: 600;
        color: var(--line-white);
        margin-top: 0.1rem;
    }

    /* ---- Class chips ---- */
    .chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        color: #14161A;
        font-weight: 600;
        margin-right: 0.3rem;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: var(--asphalt-900);
        border-right: 1px solid var(--asphalt-700);
    }
    [data-testid="stSidebar"] h3 {
        font-size: 1rem;
    }

    /* ---- Buttons ---- */
    .stButton>button, .stDownloadButton>button {
        background: var(--amber);
        color: #14161A;
        font-weight: 600;
        border: none;
        border-radius: 6px;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--amber-dim);
        color: #14161A;
    }

    /* ---- File uploader ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--asphalt-900);
        border: 1.5px dashed var(--asphalt-700);
        border-radius: 10px;
    }

    /* ---- Dataframe ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--asphalt-700);
        border-radius: 8px;
    }

    /* ---- Divider (lane line) ---- */
    .lane-divider {
        height: 2px;
        margin: 1.4rem 0;
        background-image: repeating-linear-gradient(
            90deg, var(--asphalt-700) 0px, var(--asphalt-700) 18px, transparent 18px, transparent 32px
        );
    }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading — cached across reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)


def class_chip(name: str) -> str:
    color = CLASS_COLORS.get(name, "#9AA1AC")
    return f'<span class="chip" style="background:{color}">{name}</span>'


def run_inference(model, image, conf, iou, imgsz):
    start = time.time()
    results = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    elapsed = time.time() - start
    return results[0], elapsed


def render_result_panel(label, result, elapsed, key_prefix):
    annotated = result.plot()
    n_det = len(result.boxes)
    fps = 1.0 / elapsed if elapsed > 0 else 0.0

    st.markdown(f'<div class="panel-label">{label}</div>', unsafe_allow_html=True)
    st.image(annotated, channels="BGR", use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for col, lbl, val in zip(
        (c1, c2, c3),
        ("Detections", "Latency", "Equiv. FPS"),
        (str(n_det), f"{elapsed * 1000:.0f} ms", f"{fps:.1f}"),
    ):
        col.markdown(
            f'<div class="metric-card"><div class="label">{lbl}</div>'
            f'<div class="value">{val}</div></div>',
            unsafe_allow_html=True,
        )

    if n_det > 0:
        names = result.names
        detected_classes = sorted({names[int(b.cls[0])] for b in result.boxes})
        chips = " ".join(class_chip(c) for c in detected_classes)
        st.markdown(f"<div style='margin-top:0.6rem'>{chips}</div>", unsafe_allow_html=True)

    buf = io.BytesIO()
    Image.fromarray(annotated[:, :, ::-1]).save(buf, format="PNG")
    st.download_button(
        "Download annotated image",
        data=buf.getvalue(),
        file_name=f"{key_prefix}_detections.png",
        mime="image/png",
        use_container_width=True,
        key=f"dl_{key_prefix}",
    )

    return n_det, result


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Scan Settings")

    mode = st.radio("Mode", ["Single model", "Compare all models"], index=0)

    if mode == "Single model":
        model_choice = st.selectbox("Model", list(MODEL_PATHS.keys()))
    else:
        model_choice = None

    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)
    img_size = st.select_slider("Inference size", [320, 480, 640, 960], value=640)

    st.markdown('<div class="lane-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Damage Classes")
    st.markdown(
        " ".join(class_chip(c) for c in CLASS_COLORS) + "<br>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lane-divider"></div>', unsafe_allow_html=True)
    with st.expander("Model benchmarks (RDD2022 val)"):
        for name, m in BENCHMARKS.items():
            st.markdown(f"**{name}**")
            st.markdown(
                f"<span class='badge'>P {m['P']:.3f}</span> "
                f"<span class='badge'>R {m['R']:.3f}</span> "
                f"<span class='badge'>mAP50 {m['mAP50']:.3f}</span> "
                f"<span class='badge'>FPS {m['FPS']:.1f}</span> "
                f"<span class='badge'>{m['Params']}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">RDD2022 &middot; YOLOv8n &middot; CBAM Attention</div>
        <div class="hero-title">Road Damage Detection</div>
        <p class="hero-sub">
            Drop a road photo below to scan it for cracks and potholes.
            Switch models in the sidebar to compare a plain YOLOv8n baseline
            against two CBAM-attention variants trained for this project.
        </p>
        <div class="badge-row">
            <span class="badge">4 damage classes</span>
            <span class="badge">3 trained models</span>
            <span class="badge">Real-time inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Drop a road image here, or click to browse",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("Upload a road photo above to run a scan.")
    st.stop()

image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

st.markdown('<div class="lane-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Single model mode
# ---------------------------------------------------------------------------
if mode == "Single model":
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    model = load_model(MODEL_PATHS[model_choice])
    if model is None:
        st.error(
            f"Weight file not found: `{MODEL_PATHS[model_choice]}`. "
            f"Update MODEL_PATHS at the top of app.py, or add the file under `weights/`."
        )
        st.stop()

    result, elapsed = run_inference(model, image, conf_threshold, iou_threshold, img_size)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        n_det, result = render_result_panel(model_choice, result, elapsed, "single")
        st.markdown("</div>", unsafe_allow_html=True)

    if n_det > 0:
        st.markdown('<div class="lane-divider"></div>', unsafe_allow_html=True)
        st.markdown("### Detection Report")
        names = result.names
        rows = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            rows.append(
                {
                    "Class": names[cls_id],
                    "Confidence": round(float(box.conf[0]), 2),
                    "BBox (x1, y1, x2, y2)": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No damage detected above the current confidence threshold. Try lowering it in the sidebar.")

# ---------------------------------------------------------------------------
# Compare-all mode
# ---------------------------------------------------------------------------
else:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">Input Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="lane-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Model Comparison")

    cols = st.columns(len(MODEL_PATHS))
    for col, (name, path) in zip(cols, MODEL_PATHS.items()):
        with col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            model = load_model(path)
            if model is None:
                st.error(f"Weight not found:\n`{path}`")
            else:
                result, elapsed = run_inference(model, image, conf_threshold, iou_threshold, img_size)
                render_result_panel(name, result, elapsed, name.split()[0].lower())
            st.markdown("</div>", unsafe_allow_html=True)