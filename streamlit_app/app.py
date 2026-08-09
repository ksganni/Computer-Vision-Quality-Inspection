"""Streamlit interface for the Computer Vision Quality Inspection System."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import streamlit as st

# Allow importing the app package when running: streamlit run streamlit_app/app.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import OUTPUTS_DIR, SAMPLE_IMAGES_DIR  # noqa: E402
from app.detector import QualityInspector  # noqa: E402

SAMPLE_CATALOG = [
    {
        "file": "sample_street.jpg",
        "title": "Street scene",
        "blurb": "Multi-class detection: vehicles and pedestrians",
        "detect_all": True,
    },
    {
        "file": "sample_people.jpg",
        "title": "People",
        "blurb": "Person detection benchmark image",
        "detect_all": True,
    },
]

st.set_page_config(
    page_title="Quality Inspection System",
    page_icon="◎",
    layout="wide",
)

# ---------- Theme: deep-navy machine-vision palette with cyan accents ----------
st.markdown(
    """
<style>
:root {
    --bg-deep: #081826;
    --bg-panel: #10273a;
    --bg-panel-2: #143049;
    --accent: #2dd4bf;
    --accent-soft: rgba(45, 212, 191, 0.14);
    --accent-line: rgba(45, 212, 191, 0.35);
    --ink: #dceef5;
    --ink-dim: #8fb3c4;
}

/* Full-page background: layered gradient with a sensor-dot lattice */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 90% 60% at 15% -10%, rgba(45, 212, 191, 0.10), transparent 55%),
        radial-gradient(ellipse 70% 50% at 95% 10%, rgba(56, 152, 199, 0.12), transparent 60%),
        linear-gradient(160deg, #081826 0%, #0b2233 45%, #0a1c2e 100%);
    background-attachment: fixed;
}
/* Dot lattice: evokes a camera sensor / feature-point array */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(rgba(45, 212, 191, 0.10) 1.5px, transparent 1.5px),
        radial-gradient(rgba(143, 179, 196, 0.06) 1px, transparent 1px);
    background-size: 56px 56px, 28px 28px;
    background-position: 0 0, 14px 14px;
}
/* Soft diagonal light sweep across the page */
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(
        115deg,
        transparent 0%,
        rgba(45, 212, 191, 0.045) 38%,
        rgba(56, 152, 199, 0.055) 50%,
        rgba(45, 212, 191, 0.045) 62%,
        transparent 100%
    );
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2133 0%, #0a1b2b 100%);
    border-right: 1px solid var(--accent-line);
}
[data-testid="stSidebar"] hr {
    border-color: rgba(143, 179, 196, 0.25);
}

/* Hero banner */
.hero {
    border: 1px solid var(--accent-line);
    border-radius: 14px;
    padding: 1.6rem 1.8rem 1.4rem;
    margin-bottom: 1.2rem;
    background:
        radial-gradient(ellipse 60% 120% at 100% 0%, rgba(45, 212, 191, 0.10), transparent 60%),
        linear-gradient(135deg, rgba(20, 48, 73, 0.92), rgba(13, 33, 51, 0.92));
    box-shadow: 0 8px 28px rgba(3, 12, 20, 0.45);
}
.hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.65rem;
    letter-spacing: 0.02em;
    color: var(--ink);
}
.hero p {
    margin: 0;
    color: var(--ink-dim);
    font-size: 0.95rem;
}
.hero .badge-row { margin-top: 0.85rem; }
.badge {
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent-line);
    background: var(--accent-soft);
    border-radius: 999px;
    padding: 0.22rem 0.7rem;
    margin-right: 0.45rem;
}

/* Metric cards */
.metric-card {
    border: 1px solid var(--accent-line);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    background: linear-gradient(150deg, var(--bg-panel-2), var(--bg-panel));
    box-shadow: 0 4px 18px rgba(3, 12, 20, 0.35);
}
.metric-card .label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.1;
}

/* Section headings */
.section-title {
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--accent-line);
    padding-bottom: 0.35rem;
    margin: 1.4rem 0 0.8rem 0;
}

/* Panels for images */
.panel-caption {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--ink-dim);
    margin-bottom: 0.4rem;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border: 1px solid var(--accent-line);
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 8px;
    transition: all 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent);
    background: rgba(45, 212, 191, 0.22);
    color: #eafffb;
}

/* Expanders and file uploader */
[data-testid="stExpander"] {
    border: 1px solid rgba(143, 179, 196, 0.22);
    border-radius: 10px;
    background: rgba(16, 39, 58, 0.55);
}
[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed var(--accent-line);
    background: rgba(16, 39, 58, 0.55);
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(143, 179, 196, 0.22);
    border-radius: 10px;
}

/* Select controls */
[data-testid="stSelectbox"] > div > div {
    border: 1px solid var(--accent-line);
    border-radius: 8px;
    background: rgba(16, 39, 58, 0.7);
}

/* Layering: main content forms its own stacking context above the page
   gradient, and the animation canvas sits inside it at negative z-index,
   i.e. above the gradient but below every widget. */
[data-testid="stMain"] {
    position: relative;
    z-index: 1;
    background: transparent !important;
}
[data-testid="stSidebar"] {
    z-index: 2;
}

/* Animated node-network canvas: pinned behind all content */
canvas#net {
    position: fixed !important;
    inset: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: -1 !important;
    border: none !important;
    pointer-events: none !important;
    background: transparent !important;
}
/* Collapse the empty element slot left by the pinned canvas */
[data-testid="stElementContainer"]:has(canvas#net) {
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# Animated background: drifting feature points connected by graph edges,
# representing keypoint matching in a computer vision pipeline.
st.html(
    """
<canvas id="net"></canvas>
<script>
(function () {
  const canvas = document.getElementById("net");
  // The element is re-rendered on some reruns; only one loop per canvas node.
  if (!canvas || canvas.dataset.started) return;
  canvas.dataset.started = "1";

  const ctx = canvas.getContext("2d");
  let w, h, nodes;

  const NODE_COUNT = 60;
  const LINK_DIST = 150;
  const SPEED = 0.35;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  function makeNodes() {
    nodes = Array.from({ length: NODE_COUNT }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * SPEED,
      vy: (Math.random() - 0.5) * SPEED,
      r: 1.2 + Math.random() * 1.8,
      square: Math.random() < 0.18
    }));
  }

  function step() {
    // Abandon the loop if Streamlit replaced the canvas on a rerun.
    if (!canvas.isConnected) return;

    ctx.clearRect(0, 0, w, h);

    for (const n of nodes) {
      n.x += n.vx;
      n.y += n.vy;
      if (n.x < -20) n.x = w + 20; else if (n.x > w + 20) n.x = -20;
      if (n.y < -20) n.y = h + 20; else if (n.y > h + 20) n.y = -20;
    }

    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d < LINK_DIST) {
          const alpha = 0.16 * (1 - d / LINK_DIST);
          ctx.strokeStyle = "rgba(45, 212, 191," + alpha + ")";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    for (const n of nodes) {
      if (n.square) {
        // Small detection-box markers among the feature points
        ctx.strokeStyle = "rgba(45, 212, 191, 0.45)";
        ctx.lineWidth = 1;
        ctx.strokeRect(n.x - 4, n.y - 4, 8, 8);
      } else {
        ctx.fillStyle = "rgba(45, 212, 191, 0.55)";
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    requestAnimationFrame(step);
  }

  window.addEventListener("resize", () => { resize(); makeNodes(); });
  resize();
  makeNodes();
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    requestAnimationFrame(step);
  } else {
    step && ctx.clearRect(0, 0, w, h);
  }
})();
</script>
""",
    unsafe_allow_javascript=True,
)


@st.cache_resource
def load_inspector() -> QualityInspector:
    return QualityInspector()


def list_available_samples() -> list[dict]:
    available = []
    for item in SAMPLE_CATALOG:
        path = SAMPLE_IMAGES_DIR / item["file"]
        if path.exists() and path.stat().st_size > 100:
            available.append({**item, "path": path})
    return available


def clear_previous_results() -> None:
    """Remove the previous image and result from the session state."""
    for key in (
        "last_result",
        "last_image_bytes",
        "active_image_bytes",
        "active_image_name",
        "active_detect_all",
        "active_source_key",
    ):
        st.session_state.pop(key, None)


def set_active_image(
    image_bytes: bytes,
    image_name: str,
    *,
    detect_all: bool,
    source_key: str,
) -> None:
    """Replace any previously selected image and schedule inference."""
    previous = st.session_state.get("active_source_key")
    if previous != source_key:
        clear_previous_results()

    st.session_state["active_image_bytes"] = image_bytes
    st.session_state["active_image_name"] = image_name
    st.session_state["active_detect_all"] = detect_all
    st.session_state["active_source_key"] = source_key
    st.session_state["run_now"] = True


def run_inspection(image_bytes: bytes, image_name: str, detect_all: bool) -> dict:
    temp_path = OUTPUTS_DIR / f"streamlit_{image_name}"
    temp_path.write_bytes(image_bytes)
    inspector = load_inspector()
    result = inspector.inspect(temp_path, detect_all=detect_all)
    annotated_path = OUTPUTS_DIR / f"streamlit_annotated_{image_name}"
    inspector.save_annotated_image(result["annotated_image"], annotated_path)
    result["source_name"] = image_name
    return result


def metric_card(label: str, value: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f"</div>"
    )


def render_metric_help() -> None:
    with st.expander("Interpretation of reported values", expanded=False):
        st.markdown(
            """
**Quality Grade** - categorical summary of mean detection confidence:
- **Good** - mean confidence ≥ 70%
- **Fair** - mean confidence between 45% and 70%
- **Poor** - mean confidence below 45%
- **Needs Review** - no target objects detected

**Quality Score** - mean detection confidence expressed on a 0-100 scale.

**Objects Detected** - number of retained detections in the image.

**Confidence** (per detection) - model certainty for an individual bounding box,
on a 0-1 scale. For example, `0.91` indicates approximately 91% certainty.

**Bounding Box** - pixel coordinates of the detected region
(`x1, y1` upper-left corner; `x2, y2` lower-right corner).

*Note: the grade summarizes detection confidence produced by a pretrained COCO model.
Domain-specific defect grading requires training on annotated defect data.*
"""
        )


def render_results(image_bytes: bytes, result: dict) -> None:
    st.markdown(
        f'<div class="section-title">Inference Results - {result.get("source_name", "image")}</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel-caption">Input Image</div>', unsafe_allow_html=True)
        st.image(image_bytes, width="stretch")
    with col2:
        st.markdown('<div class="panel-caption">Annotated Detections</div>', unsafe_allow_html=True)
        rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
        st.image(rgb, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.markdown(metric_card("Quality Grade", str(result["quality_grade"])), unsafe_allow_html=True)
    m2.markdown(metric_card("Quality Score", f"{result['quality_score']}"), unsafe_allow_html=True)
    m3.markdown(
        metric_card("Objects Detected", str(result["num_detections"])), unsafe_allow_html=True
    )

    st.write("")
    st.caption(result["quality_notes"])
    render_metric_help()

    if result["detections"]:
        st.markdown('<div class="section-title">Detection Table</div>', unsafe_allow_html=True)
        st.dataframe(result["detections"], width="stretch")
    else:
        st.info(
            "No matching objects were detected. Select the **Street scene** or **People** "
            "sample from the sidebar, or enable **Detect all object classes**."
        )


# ---------- Sidebar ----------
with st.sidebar:
    if "page" not in st.session_state:
        st.session_state["page"] = "Inspection"

    nav1, nav2 = st.columns(2)
    with nav1:
        if st.button("Inspection", width="stretch"):
            st.session_state["page"] = "Inspection"
    with nav2:
        if st.button("About", width="stretch"):
            st.session_state["page"] = "About"
    page = st.session_state["page"]

    st.markdown('<div class="section-title">Detection Scope</div>', unsafe_allow_html=True)
    scope = st.selectbox(
        "Detection scope",
        ["All object classes", "Food-related classes only"],
        index=0,
        label_visibility="collapsed",
    )
    detect_all = scope == "All object classes"
    st.caption(
        "Determines which detections are retained after inference. "
        "*All object classes* reports every category the COCO-pretrained model can detect "
        "(recommended for the reference images). *Food-related classes only* discards "
        "detections outside the configured food classes, reflecting the fruit-inspection "
        "use case."
    )

    st.markdown('<div class="section-title">Reference Images</div>', unsafe_allow_html=True)
    st.caption("Run inference on a benchmark image, or download it for reuse.")

    samples = list_available_samples()
    if not samples:
        st.warning("No reference images found. Run: `python scripts/download_samples.py`")
    else:
        for item in samples:
            path: Path = item["path"]
            with st.expander(item["title"], expanded=False):
                st.image(str(path), width="stretch")
                st.caption(item["blurb"])
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Analyze", key=f"use_{path.name}", width="stretch"):
                        # Reset the file uploader so a previous upload cannot override the sample
                        st.session_state["uploader_reset"] = (
                            st.session_state.get("uploader_reset", 0) + 1
                        )
                        set_active_image(
                            path.read_bytes(),
                            path.name,
                            detect_all=True,
                            source_key=f"sample:{path.name}",
                        )
                        st.rerun()
                with c2:
                    st.download_button(
                        "Download",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime="image/jpeg",
                        key=f"dl_{path.name}",
                        width="stretch",
                    )


# ---------- Main ----------
st.markdown(
    """
<div class="hero">
  <h1>◎ Computer Vision Quality Inspection System</h1>
  <p>Automated object detection and confidence-based quality assessment.
  Submit an image to obtain class predictions, bounding boxes, and per-detection
  confidence scores produced by a YOLOv8 model.</p>
  <div class="badge-row">
    <span class="badge">YOLOv8</span>
    <span class="badge">PyTorch</span>
    <span class="badge">OpenCV</span>
    <span class="badge">Object Detection</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if page == "About":
    st.markdown('<div class="section-title">System Description</div>', unsafe_allow_html=True)
    st.markdown(
        """
        This application performs object detection on user-submitted images and reports
        a confidence-based quality summary.

        **Processing pipeline**
        1. An image is submitted through this interface or the REST API.
        2. The YOLOv8 model produces class predictions, bounding boxes, and confidence scores.
        3. Detections are filtered by the configured detection scope.
        4. A quality grade and score are derived from mean detection confidence.
        5. The annotated image and detection table are returned.
        """
    )
    st.stop()

st.markdown('<div class="section-title">Image Submission</div>', unsafe_allow_html=True)
if "uploader_reset" not in st.session_state:
    st.session_state["uploader_reset"] = 0

uploaded = st.file_uploader(
    "Submit an image for inspection",
    type=["jpg", "jpeg", "png", "webp"],
    help="Submitting a new image replaces the previous image and its results.",
    key=f"image_uploader_{st.session_state['uploader_reset']}",
)

if uploaded is not None:
    source_key = f"upload:{uploaded.name}:{uploaded.size}:{st.session_state['uploader_reset']}"
    # Apply only when this upload is new; do not re-apply on every rerun over a sample
    if st.session_state.get("active_source_key") != source_key:
        set_active_image(
            uploaded.getvalue(),
            uploaded.name,
            detect_all=detect_all,
            source_key=source_key,
        )

should_run = st.session_state.get("run_now") and "active_image_bytes" in st.session_state
has_cached = "last_result" in st.session_state and "active_image_bytes" in st.session_state

if should_run:
    image_bytes = st.session_state["active_image_bytes"]
    image_name = st.session_state["active_image_name"]
    use_detect_all = st.session_state.get("active_detect_all", detect_all)

    with st.spinner("Running inference..."):
        try:
            result = run_inspection(image_bytes, image_name, detect_all=use_detect_all)
            st.session_state["last_result"] = result
            st.session_state["last_image_bytes"] = image_bytes
            render_results(image_bytes, result)
        except Exception as exc:  # noqa: BLE001
            clear_previous_results()
            st.error(f"Inference failed: {exc}")
    st.session_state["run_now"] = False
elif has_cached:
    render_results(st.session_state["last_image_bytes"], st.session_state["last_result"])
else:
    st.info(
        "Submit an image above, or select **Analyze** on a reference image in the sidebar."
    )
    st.markdown(
        """
        **Procedure**
        1. Select a reference image (*Street scene* or *People*) from the sidebar, or submit your own.
        2. Review the annotated detections alongside the input image.
        3. Interpret the quality grade and score using the explanation provided below the metrics.
        """
    )
