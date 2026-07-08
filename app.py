"""
POD-FCDNN Streamlit Web Application — Modern UI
-------------------------------------------------
Interactive dashboard for POD-based surrogate modeling of fluid dynamics.

Sections:
- Hero header + live model KPIs
- Flow Visualization: LIC (Line Integral Convolution) texture + jet colormap,
  the CFD-post-processing look, rendered fast via cached triangulation and
  automatic mesh subsampling for large cases.
- Field Explorer: interactive Plotly contour panels for p, u, v, |V|.
- Model Diagnostics: POD energy spectrum + network architecture.
- Validation: upload a ground-truth snapshot, get error metrics + diff maps.
"""

import io
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
    page_icon="🌊",
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
    "Cavity": "🧊",
    "Cylinder": "⭕",
    "Backward Facing Step": "📐",
    "NACA0012": "✈️",
}

FIELD_META = {
    "p": {"label": "Pressure", "colorscale": "Viridis"},
    "u": {"label": "U Velocity", "colorscale": "RdBu_r"},
    "v": {"label": "V Velocity", "colorscale": "RdBu_r"},
}

# ============================================================================
# Custom styling
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.hero {
    background: linear-gradient(135deg, rgba(34,211,238,0.15), rgba(59,130,246,0.05));
    border: 1px solid rgba(34,211,238,0.25);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.hero-sub { color: #94a3b8; font-size: 0.95rem; }

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 16px;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
    background: rgba(255,255,255,0.02);
}

div[data-testid="stButton"] button {
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
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


@st.cache_data(show_spinner=False)
def compute_lic_image(case_name: str, param: float, resolution: int, lic_length: int = 25):
    """Full LIC + jet-speed composite, cached per (case, param, resolution)."""
    trainer = get_model(case_name)
    geo = get_geometry(case_name)
    result = predict_and_reconstruct(trainer, param)
    idx = geo["idx"]

    u = result["u"][idx]
    v = result["v"][idx]
    speed = np.sqrt(result["u"] ** 2 + result["v"] ** 2)[idx]

    gx, gy, grid_x, grid_y = get_grid(case_name, resolution)
    grid_u, mask = masked_interpolate(geo, u, grid_x, grid_y)
    grid_v, _ = masked_interpolate(geo, v, grid_x, grid_y)
    grid_speed, _ = masked_interpolate(geo, speed, grid_x, grid_y)

    u_f = np.nan_to_num(grid_u, nan=0.0)
    v_f = np.nan_to_num(grid_v, nan=0.0)
    lic_result = lic_lib.lic(u_f, v_f, length=lic_length)

    smin, smax = np.nanmin(grid_speed), np.nanmax(grid_speed)
    speed_norm = np.nan_to_num((grid_speed - smin) / (smax - smin + 1e-9), nan=0.0)
    cmap = plt.get_cmap("jet")
    color_img = cmap(speed_norm)[:, :, :3]

    lic_norm = (lic_result - lic_result.min()) / (lic_result.max() - lic_result.min() + 1e-9)
    texture = 0.55 + 0.45 * lic_norm[..., None]
    blended = np.clip(color_img * texture, 0, 1)
    blended[mask] = [0.4, 0.4, 0.45]

    return blended, gx, gy, float(smin), float(smax)


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


# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">🌊 POD-FCDNN Surrogate Model</div>
    <div class="hero-sub">Real-time CFD flow field prediction · Proper Orthogonal Decomposition + Neural Network surrogate</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

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

    with st.expander("🎨 Rendering options"):
        resolution = st.slider("Grid resolution", 80, 300, 180, 10)
        lic_length = st.slider("LIC streak length", 10, 45, 25, 5)
        show_vectors = st.checkbox("Overlay vector arrows (Field Explorer)", value=False)

    predict_btn = st.button("🚀 Predict Flow Field", use_container_width=True, type="primary")

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

tab_flow, tab_explorer, tab_diag, tab_validate = st.tabs(
    ["🌈 Flow Visualization", "🔎 Field Explorer", "📊 Model Diagnostics", "✅ Validate vs Ground Truth"]
)

# ----------------------------------------------------------------------
# TAB 1 — Flow Visualization (LIC hero render)
# ----------------------------------------------------------------------
with tab_flow:
    if not has_result:
        st.info("Set your parameters in the sidebar and click **Predict Flow Field**.")
    else:
        used_param = st.session_state[state_key]["param"]
        with st.spinner("Rendering flow texture..."):
            blended, gx, gy, smin, smax = compute_lic_image(case, used_param, resolution, lic_length)

        fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0b0f19")
        ax.set_facecolor("#0b0f19")
        ax.imshow(blended, origin="lower", extent=[gx.min(), gx.max(), gy.min(), gy.max()], aspect="equal")
        ax.set_title(f"{case} — Velocity Field ({param_label} = {used_param:g})",
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

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption("This view uses Line Integral Convolution (LIC) to render continuous flow-direction texture, colored by velocity magnitude — the classic CFD post-processing look.")

# ----------------------------------------------------------------------
# TAB 2 — Field Explorer (interactive Plotly panels)
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
            "⬇ Download predicted field as CSV",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name=f"{case.replace(' ', '_').lower()}_{param_label}_{used_param:g}_prediction.csv",
            mime="text/csv",
        )

# ----------------------------------------------------------------------
# TAB 3 — Model Diagnostics
# ----------------------------------------------------------------------
with tab_diag:
    pod = trainer.pod
    energy = pod.svals ** 2
    cum_energy = np.cumsum(energy / energy.sum())

    st.metric("Energy captured by retained modes", f"{cum_energy[-1]*100:.3f}%")

    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(y=cum_energy * 100, mode="lines+markers", line=dict(color="#22d3ee")))
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
# TAB 4 — Validation vs Ground Truth
# ----------------------------------------------------------------------
with tab_validate:
    st.caption(
        "Upload the raw .dat/.csv snapshot for the same case at the same "
        f"{param_label} value shown in the sidebar. Expected columns: "
        "nodenumber, x-coordinate, y-coordinate, absolute-pressure, x-velocity, y-velocity."
    )
    uploaded = st.file_uploader("Ground-truth snapshot file", type=["dat", "csv", "txt"])

    if uploaded is not None:
        try:
            buffer = io.StringIO(uploaded.getvalue().decode("utf-8"))
            xvec_truth, xy_truth = load_snapshot_uvp_from_buffer(buffer)
            N_truth = xy_truth.shape[0]
            u_t = xvec_truth[:N_truth]
            v_t = xvec_truth[N_truth:2 * N_truth]
            p_t = xvec_truth[2 * N_truth:3 * N_truth]

            xy_full = trainer.pod.xy
            if N_truth != xy_full.shape[0]:
                st.warning(f"Uploaded snapshot has {N_truth} nodes vs. model mesh {xy_full.shape[0]}; interpolating onto model mesh.")
                tri_truth = Delaunay(xy_truth)
                u_t = LinearNDInterpolator(tri_truth, u_t)(xy_full[:, 0], xy_full[:, 1])
                v_t = LinearNDInterpolator(tri_truth, v_t)(xy_full[:, 0], xy_full[:, 1])
                p_t = LinearNDInterpolator(tri_truth, p_t)(xy_full[:, 0], xy_full[:, 1])

            pred_result = predict_and_reconstruct(trainer, param)
            u_p, v_p, p_p = pred_result["u"], pred_result["v"], pred_result["p"]

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

            geo = get_geometry(case)
            gx, gy, grid_x, grid_y = get_grid(case, resolution)
            grid_truth, _ = masked_interpolate(geo, truth_vals[geo["idx"]], grid_x, grid_y)
            grid_pred, _ = masked_interpolate(geo, pred_vals[geo["idx"]], grid_x, grid_y)
            grid_diff = grid_pred - grid_truth

            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(make_contour(gx, gy, grid_truth, f"Ground Truth — {field_choice}", cscale), use_container_width=True)
            c2.plotly_chart(make_contour(gx, gy, grid_pred, f"Surrogate — {field_choice}", cscale), use_container_width=True)
            c3.plotly_chart(make_contour(gx, gy, grid_diff, "Difference (Pred − Truth)", "RdBu_r"), use_container_width=True)

        except Exception as e:
            st.error(f"Could not process uploaded file: {str(e)}")
    else:
        st.info("Upload a ground-truth snapshot to compute error metrics and see a side-by-side comparison.")
