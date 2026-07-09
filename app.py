"""
POD-FCDNN Streamlit Web Application
-------------------------------------
Interactive dashboard for POD-based surrogate modeling of fluid dynamics.

Tabs:
- Overview           : landing page, thumbnail preview of all 4 cases
- Flow Visualization : LIC (Line Integral Convolution) + jet colormap hero render
- Parameter Comparison: bordered grid comparing several parameter values at once
- Field Explorer     : interactive Plotly contour panels for p, u, v, |V|
- Model Diagnostics  : POD energy spectrum + network architecture
- Validation         : upload a ground-truth snapshot, get error metrics + diff maps
"""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator
import lic as lic_lib

from engine import (
    load_checkpoint,
    predict_and_reconstruct,
    compute_errors,
    infer_architecture,
    load_snapshot_uvp_from_buffer,
    subsample_indices,
)

# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="POD-FCDNN Surrogate Model",
    page_icon="〰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHECKPOINT_PATHS = {
    "Cavity": "checkpoints/cavity_checkpoint.pt",
    "Cylinder": "checkpoints/cylinder_checkpoint.pt",
    "Backward Facing Step": "checkpoints/bfs_checkpoint.pt",
    "NACA0012": "checkpoints/naca_checkpoint.pt",
}

CASE_ICONS = {
    "Cavity": "◲",
    "Cylinder": "◯",
    "Backward Facing Step": "⌐",
    "NACA0012": "✈",
}

CASE_BLURBS = {
    "Cavity": "Lid-driven cavity flow, parameterized by Reynolds number.",
    "Cylinder": "Flow past a circular cylinder, parameterized by Reynolds number.",
    "Backward Facing Step": "Separated flow over a backward-facing step, parameterized by Reynolds number.",
    "NACA0012": "Flow over a NACA0012 airfoil, parameterized by angle of attack.",
}

DEFAULT_PARAM = {
    "Cavity": 1000, "Cylinder": 1000, "Backward Facing Step": 1000, "NACA0012": 4.0,
}

FIELD_META = {
    "p": {"label": "Pressure", "colorscale": "RdBu"},
    "u": {"label": "U Velocity", "colorscale": "Jet"},
    "v": {"label": "V Velocity", "colorscale": "Jet"},
}

ACCENT = "#22d3ee"
BG = "#0b0f19"

# ============================================================================
# Minimal, clean styling
# ============================================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{background: transparent;}}

.app-header {{
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 14px;
    margin-bottom: 18px;
}}
.app-title {{
    font-size: 1.55rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.01em;
}}
.app-title span {{ color: {ACCENT}; }}
.app-sub {{ color: #7c8aa0; font-size: 0.9rem; margin-top: 2px; }}

div[data-testid="stMetric"] {{
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 10px 14px;
}}
div[data-testid="stMetricLabel"] {{ font-size: 0.78rem; color: #7c8aa0; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
.stTabs [data-baseweb="tab"] {{
    padding: 8px 16px;
    background: transparent;
    color: #7c8aa0;
}}
.stTabs [aria-selected="true"] {{ color: {ACCENT} !important; }}

div[data-testid="stButton"] button {{
    border-radius: 8px;
    font-weight: 600;
}}

.case-card {{
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.015);
    height: 100%;
}}
.case-card-title {{ font-weight: 600; font-size: 1rem; color: #f1f5f9; }}
.case-card-sub {{ color: #7c8aa0; font-size: 0.82rem; margin-top: 2px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Cached resources
# ============================================================================

@st.cache_resource(show_spinner="Loading model checkpoint...")
def get_model(case_name: str):
    return load_checkpoint(CHECKPOINT_PATHS[case_name])


@st.cache_resource(show_spinner="Preparing mesh geometry...")
def get_geometry(case_name: str):
    """
    Triangulation + KD-tree for a case, built once and reused for every
    prediction. For very large meshes, uses a fixed reproducible subsample
    so rendering stays fast without visibly changing the output.
    """
    trainer = get_model(case_name)
    xy_full = trainer.pod.xy
    n_full = xy_full.shape[0]

    idx = subsample_indices(n_full, cap=45000)
    xy = xy_full[idx]

    tri = Delaunay(xy)
    kdt = cKDTree(xy)
    d_nn, _ = kdt.query(xy, k=2)
    spacing = float(np.median(d_nn[:, 1]))

    return {
        "idx": idx, "xy": xy, "tri": tri, "kdt": kdt, "spacing": spacing,
        "n_full": n_full, "x_min": xy_full[:, 0].min(), "x_max": xy_full[:, 0].max(),
        "y_min": xy_full[:, 1].min(), "y_max": xy_full[:, 1].max(),
    }


@st.cache_data(show_spinner=False)
def get_grid(case_name: str, resolution: int):
    geo = get_geometry(case_name)
    gx = np.linspace(geo["x_min"], geo["x_max"], resolution)
    gy = np.linspace(geo["y_min"], geo["y_max"], resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)
    return gx, gy, grid_x, grid_y


def masked_interpolate(geo, values, grid_x, grid_y, hole_factor=2.5):
    interp = LinearNDInterpolator(geo["tri"], values)
    grid_z = interp(grid_x, grid_y)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    dist, _ = geo["kdt"].query(grid_pts)
    mask = (dist > hole_factor * geo["spacing"]).reshape(grid_x.shape)
    grid_z[mask] = np.nan
    return grid_z, mask


def build_lic_composite(grid_u, grid_v, grid_speed, mask, lic_length, vmin=None, vmax=None):
    """
    Shared LIC + jet-speed compositing logic, used by both the single-panel
    Flow Visualization and the multi-panel Parameter Comparison, so both
    stay visually consistent and there's one place to tune the look.
    """
    u_f = np.nan_to_num(grid_u, nan=0.0)
    v_f = np.nan_to_num(grid_v, nan=0.0)
    lic_result = lic_lib.lic(u_f, v_f, length=lic_length)

    if vmin is None:
        vmin = float(np.nanmin(grid_speed))
    if vmax is None:
        vmax = float(np.nanmax(grid_speed))

    speed_norm = np.nan_to_num((grid_speed - vmin) / (vmax - vmin + 1e-9), nan=0.0)
    cmap = plt.get_cmap("jet")
    color_img = cmap(speed_norm)[:, :, :3]

    lic_norm = (lic_result - lic_result.min()) / (lic_result.max() - lic_result.min() + 1e-9)
    texture = 0.55 + 0.45 * lic_norm[..., None]
    blended = np.clip(color_img * texture, 0, 1)
    blended[mask] = [0.4, 0.4, 0.45]
    return blended, vmin, vmax


def get_case_uv_speed(case_name, param):
    """Prediction + subsampled u, v, speed for a case/param, ready for interpolation."""
    trainer = get_model(case_name)
    geo = get_geometry(case_name)
    result = predict_and_reconstruct(trainer, param)
    idx = geo["idx"]
    u = result["u"][idx]
    v = result["v"][idx]
    speed = np.sqrt(result["u"] ** 2 + result["v"] ** 2)[idx]
    return u, v, speed


@st.cache_data(show_spinner=False)
def compute_lic_image(case_name: str, param: float, resolution: int, lic_length: int = 25):
    """Full LIC + jet-speed composite for one case/param, cached."""
    geo = get_geometry(case_name)
    u, v, speed = get_case_uv_speed(case_name, param)
    gx, gy, grid_x, grid_y = get_grid(case_name, resolution)

    grid_u, mask = masked_interpolate(geo, u, grid_x, grid_y)
    grid_v, _ = masked_interpolate(geo, v, grid_x, grid_y)
    grid_speed, _ = masked_interpolate(geo, speed, grid_x, grid_y)

    blended, smin, smax = build_lic_composite(grid_u, grid_v, grid_speed, mask, lic_length)
    return blended, gx, gy, smin, smax


def render_flow_figure(case_name, param, param_label, blended, gx, gy, smin, smax):
    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    ax.set_facecolor(BG)
    ax.imshow(blended, origin="lower", extent=[gx.min(), gx.max(), gy.min(), gy.max()], aspect="equal")
    ax.set_title(f"{case_name} — Velocity Field ({param_label} = {param:g})",
                 color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(vmin=smin, vmax=smax))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Velocity magnitude", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")
    plt.tight_layout()
    return fig


def render_comparison_grid(case_name, params, param_label, resolution, lic_length):
    """Bordered grid comparing the flow field across several parameter values,
    with a shared color scale for fair comparison and the value labeled
    directly inside each panel."""
    geo = get_geometry(case_name)
    gx, gy, grid_x, grid_y = get_grid(case_name, resolution)

    panel_data = []
    for param in params:
        u, v, speed = get_case_uv_speed(case_name, param)
        grid_u, mask = masked_interpolate(geo, u, grid_x, grid_y)
        grid_v, _ = masked_interpolate(geo, v, grid_x, grid_y)
        grid_speed, _ = masked_interpolate(geo, speed, grid_x, grid_y)
        panel_data.append((grid_u, grid_v, grid_speed, mask))

    valid_speeds = np.concatenate([p[2][~np.isnan(p[2])] for p in panel_data])
    vmin, vmax = float(valid_speeds.min()), float(valid_speeds.max())

    n = len(params)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows), facecolor=BG)
    axes = np.atleast_1d(axes).ravel()

    for i, (param, (grid_u, grid_v, grid_speed, mask)) in enumerate(zip(params, panel_data)):
        ax = axes[i]
        ax.set_facecolor(BG)
        blended, _, _ = build_lic_composite(grid_u, grid_v, grid_speed, mask, lic_length, vmin, vmax)
        ax.imshow(blended, origin="lower", extent=[gx.min(), gx.max(), gy.min(), gy.max()], aspect="equal")
        for spine in ax.spines.values():
            spine.set_color(ACCENT)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        label = f"{param_label} = {param:g}"
        ax.text(
            0.03, 0.95, label, transform=ax.transAxes, color="white", fontsize=13,
            fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=BG, edgecolor=ACCENT, alpha=0.9),
        )

    for j in range(n, len(axes)):
        axes[j].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Velocity magnitude", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")
    fig.suptitle(f"{case_name} — Parameter Comparison", color="white", fontsize=15, fontweight="bold")
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def make_contour(gx, gy, gz, title, colorscale):
    fig = go.Figure(
        data=go.Contour(
            x=gx, y=gy, z=gz,
            colorscale=colorscale,
            contours=dict(coloring="heatmap"),
            line=dict(width=0),
            colorbar=dict(title=""),
        )
    )
    fig.update_layout(
        title=title, xaxis_title="x", yaxis_title="y", height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


import re

RE_FILE_RE = re.compile(r"Re[\s_\-]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
ALPHA_FILE_RE = re.compile(r"alpha[\s_\-]*(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def detect_param_from_filename(filename: str, is_naca: bool):
    pattern = ALPHA_FILE_RE if is_naca else RE_FILE_RE
    m = pattern.search(filename)
    return float(m.group(1)) if m else None


# ============================================================================
# Header
# ============================================================================

st.markdown(f"""
<div class="app-header">
    <div class="app-title">POD<span>·</span>FCDNN Surrogate Model</div>
    <div class="app-sub">Real-time CFD flow field prediction — Proper Orthogonal Decomposition + Neural Network surrogate</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("**Configuration**")

    case = st.selectbox(
        "Case",
        list(CHECKPOINT_PATHS.keys()),
        format_func=lambda c: f"{CASE_ICONS[c]}  {c}"
    )

    if case == "NACA0012":
        param = st.slider("Angle of Attack (α)", -5.0, 15.0, 0.0, 0.5)
        param_label = "α"
    else:
        param = st.slider("Reynolds Number", 100, 10000, 1000, 100)
        param_label = "Re"

    with st.expander("Rendering options"):
        resolution = st.slider("Grid resolution", 80, 300, 180, 10)
        lic_length = st.slider("LIC streak length", 10, 45, 25, 5)
        show_vectors = st.checkbox("Overlay vector arrows (Field Explorer)", value=False)

    predict_btn = st.button("Predict Flow Field", use_container_width=True, type="primary")

    st.divider()
    if st.button("Reset session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ============================================================================
# Load model + geometry, always available (cheap) for KPI row
# ============================================================================

trainer = get_model(case)
geo_full_N = trainer.pod.N

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mesh nodes", f"{geo_full_N:,}")
k2.metric("POD modes (r)", trainer.pod.r)
arch = infer_architecture(trainer.model.state_dict())
k3.metric("Network width", arch["width"])
k4.metric("Hidden layers", arch["depth"])

state_key = f"result_{case}"
if predict_btn:
    st.session_state[state_key] = {"param": param}
has_result = state_key in st.session_state

# ============================================================================
# Tabs
# ============================================================================

tab_overview, tab_flow, tab_compare, tab_explorer, tab_diag, tab_validate = st.tabs(
    ["Overview", "Flow Visualization", "Parameter Comparison", "Field Explorer", "Model Diagnostics", "Validate vs Ground Truth"]
)

# ----------------------------------------------------------------------
# TAB — Overview (landing page)
# ----------------------------------------------------------------------
with tab_overview:
    st.caption("All four benchmark cases at a glance. Pick one in the sidebar to explore it in detail.")
    cols = st.columns(4)
    for col, c in zip(cols, CHECKPOINT_PATHS.keys()):
        with col:
            with st.spinner(f"Rendering {c}..."):
                blended, gx, gy, smin, smax = compute_lic_image(c, DEFAULT_PARAM[c], 90, 20)
            fig, ax = plt.subplots(figsize=(3, 2.6), facecolor=BG)
            ax.set_facecolor(BG)
            ax.imshow(blended, origin="lower", extent=[gx.min(), gx.max(), gy.min(), gy.max()], aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("rgba(255,255,255,0.15)")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown(f"""
            <div class="case-card">
                <div class="case-card-title">{CASE_ICONS[c]} {c}</div>
                <div class="case-card-sub">{CASE_BLURBS[c]}</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# TAB — Flow Visualization (LIC hero render)
# ----------------------------------------------------------------------
with tab_flow:
    if not has_result:
        st.info("Set your parameters in the sidebar and click **Predict Flow Field**.")
    else:
        used_param = st.session_state[state_key]["param"]
        t0 = time.time()
        with st.spinner("Rendering flow texture..."):
            blended, gx, gy, smin, smax = compute_lic_image(case, used_param, resolution, lic_length)
        render_time = time.time() - t0

        fig = render_flow_figure(case, used_param, param_label, blended, gx, gy, smin, smax)
        st.pyplot(fig, use_container_width=True)

        col_a, col_b = st.columns([3, 1])
        col_a.caption(
            "Line Integral Convolution (LIC) renders continuous flow-direction texture, "
            f"colored by velocity magnitude — the classic CFD post-processing look. Rendered in {render_time:.2f}s."
        )
        col_b.download_button(
            "Download PNG",
            data=fig_to_png_bytes(fig),
            file_name=f"{case.replace(' ', '_').lower()}_{param_label}_{used_param:g}_flow.png",
            mime="image/png",
            use_container_width=True,
        )
        plt.close(fig)

# ----------------------------------------------------------------------
# TAB — Parameter Comparison (bordered grid across several values)
# ----------------------------------------------------------------------
with tab_compare:
    st.caption(
        f"Compare {case} across several {param_label} values side-by-side, "
        "each panel bordered and labeled, with a shared color scale for fair comparison."
    )

    default_values = {
        "Cavity": [500, 1000, 2500, 5000],
        "Cylinder": [500, 1000, 2500, 5000],
        "Backward Facing Step": [500, 1000, 2500, 5000],
        "NACA0012": [-5.0, 0.0, 5.0, 10.0],
    }[case]

    cols = st.columns(4)
    compare_values = []
    for i, col in enumerate(cols):
        default_val = default_values[i] if i < len(default_values) else default_values[-1]
        step = 0.5 if case == "NACA0012" else 100.0
        v = col.number_input(f"Value {i+1}", value=float(default_val), step=step, key=f"cmp_{case}_{i}")
        compare_values.append(v)

    compare_btn = st.button("Generate Comparison Grid", use_container_width=True, type="primary")
    compare_key = f"compare_{case}"
    if compare_btn:
        st.session_state[compare_key] = list(compare_values)

    if compare_key in st.session_state:
        with st.spinner("Rendering comparison grid..."):
            fig_cmp = render_comparison_grid(case, st.session_state[compare_key], param_label, resolution, lic_length)
        st.pyplot(fig_cmp, use_container_width=True)
        st.download_button(
            "Download PNG",
            data=fig_to_png_bytes(fig_cmp),
            file_name=f"{case.replace(' ', '_').lower()}_comparison.png",
            mime="image/png",
        )
        plt.close(fig_cmp)
    else:
        st.info("Set your comparison values above and click **Generate Comparison Grid**.")

# ----------------------------------------------------------------------
# TAB — Field Explorer (interactive Plotly panels)
# ----------------------------------------------------------------------
with tab_explorer:
    if not has_result:
        st.info("Set your parameters in the sidebar and click **Predict Flow Field**.")
    else:
        used_param = st.session_state[state_key]["param"]
        geo = get_geometry(case)
        result = predict_and_reconstruct(trainer, used_param)
        idx = geo["idx"]
        u, v, p = result["u"][idx], result["v"][idx], result["p"][idx]
        xy = geo["xy"]

        gx, gy, grid_x, grid_y = get_grid(case, resolution)
        grids = {}
        for key, values in {"p": p, "u": u, "v": v}.items():
            grids[key], _ = masked_interpolate(geo, values, grid_x, grid_y)

        col1, col2 = st.columns(2)
        col1.plotly_chart(make_contour(gx, gy, grids["p"], "Pressure", FIELD_META["p"]["colorscale"]), use_container_width=True)
        col2.plotly_chart(make_contour(gx, gy, grids["u"], "U Velocity", FIELD_META["u"]["colorscale"]), use_container_width=True)

        col3, col4 = st.columns(2)
        col3.plotly_chart(make_contour(gx, gy, grids["v"], "V Velocity", FIELD_META["v"]["colorscale"]), use_container_width=True)

        speed = np.sqrt(u**2 + v**2)
        grid_speed, _ = masked_interpolate(geo, speed, grid_x, grid_y)
        fig_speed = make_contour(gx, gy, grid_speed, "Velocity Magnitude", "Turbo")

        if show_vectors:
            step = max(1, len(idx) // 400)
            fig_vec = ff.create_quiver(
                xy[::step, 0], xy[::step, 1], u[::step], v[::step],
                scale=0.05, arrow_scale=0.3, line=dict(width=1, color="white"),
            )
            for trace in fig_vec.data:
                fig_speed.add_trace(trace)
        col4.plotly_chart(fig_speed, use_container_width=True)

        df_out = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "u": u, "v": v, "p": p})
        st.download_button(
            "Download predicted field as CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"{case.replace(' ', '_').lower()}_{param_label}_{used_param:g}_prediction.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------
# TAB — Model Diagnostics
# ----------------------------------------------------------------------
with tab_diag:
    pod = trainer.pod
    if np.isnan(pod.svals).any():
        st.info(
            "This checkpoint doesn't include singular-value data (it was saved by a script "
            "that stores `r_modes` but not `pod_svals`), so the energy spectrum below isn't "
            "available for it. Everything else is unaffected."
        )
    else:
        energy = pod.svals ** 2
        cum_energy = np.cumsum(energy / energy.sum())

        st.metric("Energy captured by retained modes", f"{cum_energy[-1]*100:.3f}%")

        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(y=cum_energy * 100, mode="lines+markers", line=dict(color=ACCENT)))
        fig_energy.update_layout(
            title="Cumulative POD Energy Spectrum", xaxis_title="Mode index",
            yaxis_title="Cumulative energy (%)", height=380,
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_energy, use_container_width=True)

        fig_svals = go.Figure()
        fig_svals.add_trace(go.Scatter(y=pod.svals, mode="lines+markers", line=dict(color="#818cf8")))
        fig_svals.update_layout(
            title="POD Singular Value Decay", xaxis_title="Mode index", yaxis_title="Singular value",
            yaxis_type="log", height=380,
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_svals, use_container_width=True)

# ----------------------------------------------------------------------
# TAB — Validation vs Ground Truth
# ----------------------------------------------------------------------
with tab_validate:
    st.caption(
        "Upload ground-truth CFD snapshot(s) to compare against surrogate predictions. "
        "Expected columns: nodenumber, x-coordinate, y-coordinate, absolute-pressure, "
        "x-velocity, y-velocity. If a filename contains \"Re 1000\" (or \"alpha 5\" for "
        "NACA0012), the matching parameter value is detected automatically."
    )

    is_naca = case == "NACA0012"
    uploaded_files = st.file_uploader(
        "Ground-truth snapshot file(s)", type=["dat", "csv", "txt"], accept_multiple_files=True
    )

    def _load_and_predict(file, manual_param):
        buffer = io.StringIO(file.getvalue().decode("utf-8"))
        xvec_truth, xy_truth = load_snapshot_uvp_from_buffer(buffer)
        N_truth = xy_truth.shape[0]
        u_t = xvec_truth[:N_truth]
        v_t = xvec_truth[N_truth:2 * N_truth]
        p_t = xvec_truth[2 * N_truth:3 * N_truth]

        xy_full = trainer.pod.xy
        if N_truth != xy_full.shape[0]:
            tri_truth = Delaunay(xy_truth)
            u_t = LinearNDInterpolator(tri_truth, u_t)(xy_full[:, 0], xy_full[:, 1])
            v_t = LinearNDInterpolator(tri_truth, v_t)(xy_full[:, 0], xy_full[:, 1])
            p_t = LinearNDInterpolator(tri_truth, p_t)(xy_full[:, 0], xy_full[:, 1])

        pred_result = predict_and_reconstruct(trainer, manual_param)
        return u_t, v_t, p_t, pred_result["u"], pred_result["v"], pred_result["p"], xy_full

    if uploaded_files:
        # Resolve a parameter value per file: auto-detect from filename, else fall back
        # to the sidebar value (with a manual override always available).
        file_params = []
        for f in uploaded_files:
            detected = detect_param_from_filename(f.name, is_naca)
            file_params.append(detected if detected is not None else param)

        if len(uploaded_files) > 1:
            st.markdown("**Detected/assigned parameter values** (edit any that look wrong):")
            cols = st.columns(len(uploaded_files))
            for i, (f, col) in enumerate(zip(uploaded_files, cols)):
                file_params[i] = col.number_input(
                    f.name if len(f.name) < 18 else f.name[:15] + "...",
                    value=float(file_params[i]), key=f"batch_param_{case}_{i}",
                    step=0.5 if is_naca else 100.0,
                )

            # Batch summary table across all files
            rows = {}
            for f, p_val in zip(uploaded_files, file_params):
                try:
                    u_t, v_t, p_t, u_p, v_p, p_p, _ = _load_and_predict(f, p_val)
                    for field_name, t_arr, p_arr in [("U", u_t, u_p), ("V", v_t, v_p), ("P", p_t, p_p)]:
                        err = compute_errors(t_arr, p_arr)
                        rows[f"{f.name} — {field_name} ({param_label}={p_val:g})"] = err
                except Exception as e:
                    st.error(f"Could not process {f.name}: {e}")

            if rows:
                summary_df = pd.DataFrame(rows).T
                summary_df.columns = ["MAE", "L2 error", "L2 rel. error (%)", "Max error"]
                st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)
                st.download_button(
                    "Download summary as CSV",
                    data=summary_df.to_csv().encode("utf-8"),
                    file_name=f"{case.replace(' ', '_').lower()}_batch_validation.csv",
                    mime="text/csv",
                )

            detail_file = st.selectbox("View detailed comparison for:", [f.name for f in uploaded_files])
            sel_idx = [f.name for f in uploaded_files].index(detail_file)
            sel_file, sel_param = uploaded_files[sel_idx], file_params[sel_idx]
        else:
            sel_file, sel_param = uploaded_files[0], file_params[0]
            if abs(sel_param - param) > 1e-9:
                st.caption(f"Detected {param_label} = {sel_param:g} from filename \"{sel_file.name}\".")

        try:
            u_t, v_t, p_t, u_p, v_p, p_p, xy_full = _load_and_predict(sel_file, sel_param)

            metrics = {
                "U Velocity": compute_errors(u_t, u_p),
                "V Velocity": compute_errors(v_t, v_p),
                "Pressure": compute_errors(p_t, p_p),
            }
            metrics_df = pd.DataFrame(metrics).T
            metrics_df.columns = ["MAE", "L2 error", "L2 rel. error (%)", "Max error"]
            st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

            field_choice = st.radio("Field to visualize", ["Pressure", "U Velocity", "V Velocity"], horizontal=True)
            field_map = {
                "Pressure": (p_t, p_p, FIELD_META["p"]["colorscale"]),
                "U Velocity": (u_t, u_p, FIELD_META["u"]["colorscale"]),
                "V Velocity": (v_t, v_p, FIELD_META["v"]["colorscale"]),
            }
            truth_vals, pred_vals, cscale = field_map[field_choice]
            mae = float(np.mean(np.abs(pred_vals - truth_vals)))

            geo = get_geometry(case)
            gx, gy, grid_x, grid_y = get_grid(case, resolution)
            grid_truth, _ = masked_interpolate(geo, truth_vals[geo["idx"]], grid_x, grid_y)
            grid_pred, _ = masked_interpolate(geo, pred_vals[geo["idx"]], grid_x, grid_y)
            grid_abs_err, _ = masked_interpolate(geo, np.abs(pred_vals - truth_vals)[geo["idx"]], grid_x, grid_y)

            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(make_contour(gx, gy, grid_truth, f"{field_choice} — CFD (Ground Truth)", cscale), use_container_width=True)
            c2.plotly_chart(make_contour(gx, gy, grid_pred, f"{field_choice} — Prediction", cscale), use_container_width=True)
            c3.plotly_chart(make_contour(gx, gy, grid_abs_err, f"Absolute Error (MAE = {mae:.4e})", "Inferno"), use_container_width=True)

        except Exception as e:
            st.error(f"Could not process uploaded file: {str(e)}")
    else:
        st.info("Upload one or more ground-truth snapshots to compute error metrics and see a side-by-side comparison.")
