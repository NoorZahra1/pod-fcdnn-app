"""
AeroMind Engineering Copilot Workbench
Ties multi-modal user input strings safely into underlying POD-FCDNN physics checkpoints.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import torch
from pathlib import Path

# Safely import the core ML engine directly from your companion script
try:
    from engine import load_checkpoint, predict_and_reconstruct
except ModuleNotFoundError:
    st.error("❌ **Critical File Error:** Could not find `engine.py` in the root directory. Please make sure `engine.py` is uploaded directly alongside `app.py` on GitHub.")
    st.stop()

# ==========================================
# PAGE ARCHITECTURE & BRANDING STYLING
# ==========================================
st.set_page_config(
    page_title="AeroMind | Agentic Fluid Analytics Workspace",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .report-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #10b981;
        font-family: 'Inter', sans-serif;
        color: #1f2937;
        line-height: 1.6;
        font-size: 14px;
    }
    .agent-bubble {
        background-color: #f0fdf4;
        padding: 15px;
        border-radius: 10px;
        border-bottom-left-radius: 0px;
        margin-bottom: 15px;
        border: 1px solid #bbf7d0;
        font-size: 13px;
    }
    .code-span {
        font-family: 'Courier New', monospace;
        background-color: #e5e7eb;
        padding: 2px 6px;
        border-radius: 4px;
        color: #b91c1c;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Project AeroMind: Multimodal Engineering Copilot")
st.markdown("`MS Thesis Platform: LLM Agent Tool Orchestration for Physics Reduced-Order Models`")
st.markdown("---")

# ==========================================
# DEVICE & AUTOMATED PATH DISCOVERY
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("./checkpoints")

# Case configuration mapping dictionary
CASE_MAPPING = {
    "Cylinder Wake Vortex": {"file": "cylinder_checkpoint.pt", "default_re": 200.0, "bounds": (100.0, 400.0)},
    "Lid-Driven Cavity Flow": {"file": "cavity_checkpoint.pt", "default_re": 5000.0, "bounds": (1000.0, 10000.0)},
    "Backward Facing Step (BFS)": {"file": "bfs_checkpoint.pt", "default_re": 3000.0, "bounds": (1000.0, 8000.0)},
    "NACA 0012 Airfoil": {"file": "naca_checkpoint.pt", "default_re": 4000.0, "bounds": (1000.0, 6000.0)}
}

def locate_checkpoint_file(filename: str):
    """
    Defensively checks alternative paths to stay robust across environments.
    """
    possible_paths = [
        CHECKPOINT_DIR / filename,
        Path(".") / "checkpoints" / filename,
        Path(".") / filename
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None

# ==========================================
# COGNITIVE PIPELINE EXECUTION ENGINE
# ==========================================
def agent_tool_reconstruct_field(case_name: str, reynolds: float):
    """
    Loads real PyTorch tensors, runs inference from engine.py, 
    and handles dictionary keys unpacking smoothly.
    """
    config = CASE_MAPPING[case_name]
    checkpoint_path = locate_checkpoint_file(config["file"])
    
    if not checkpoint_path:
        raise FileNotFoundError(f"Missing weight model target file: {config['file']}. Please confirm it is inside a 'checkpoints' folder in your repo.")
        
    # 1. Invoke raw neural backend loader from engine.py
    trainer = load_checkpoint(checkpoint_path, device=device)
    
    # 2. Run core prediction and capture the output dictionary safely
    prediction_data = predict_and_reconstruct(trainer, reynolds)
    
    u = prediction_data["u"]
    v = prediction_data["v"]
    xy = prediction_data["xy"]
    
    # Split coordinates array matrix (N, 2) into clear axes vectors
    x_coords = xy[:, 0]
    y_coords = xy[:, 1]
    
    # 3. High-Fidelity Dense Grid Resampling (Fixed from point scatter plots)
    grid_x = np.linspace(x_coords.min(), x_coords.max(), 250)
    grid_y = np.linspace(y_coords.min(), y_coords.max(), 150)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
    
    # Linearly interpolate unstructured points into regular matrices
    grid_U = griddata((x_coords, y_coords), u, (grid_X, grid_Y), method='linear')
    grid_V = griddata((x_coords, y_coords), v, (grid_X, grid_Y), method='linear')
    
    # Clear null values safely near geometric edge cuts
    grid_U = np.nan_to_num(grid_U, nan=0.0)
    grid_V = np.nan_to_num(grid_V, nan=0.0)
    
    r_modes = getattr(trainer.pod, 'r', 30)
    
    return grid_x, grid_y, grid_U, grid_V, r_modes

# ==========================================
# WORKBENCH WORKSPACE LAYOUT
# ==========================================
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.header("💬 Multimodal Agent Interface")
    st.info("Input natural statements to execute model variations and evaluate flow structures.")
    
    user_prompt = st.text_area(
        "Engineering Directive Input:", 
        value="Execute rapid fluid field inference on the Cylinder Wake Vortex domain with a velocity profile parameter matching Reynolds number 250.",
        height=100
    )
    
    process_btn = st.button("🚀 Parse & Execute Agentic Pipeline")
    
    st.subheader("🤖 Agent Reasoning & Tool Execution Trace")
    if process_btn:
        # Defaults configuration parsing values fallback
        target_case = "Cylinder Wake Vortex"
        target_re = 200.0
        
        # Determine selection target match based on keyword prompt strings 
        for key in CASE_MAPPING.keys():
            if key.split()[0].lower() in user_prompt.lower():
                target_case = key
                break
                
        config = CASE_MAPPING[target_case]
        
        # Regex replacement scanning logic to capture numerical integers/floats
        nums = [float(s) for s in user_prompt.replace(',',' ').split() if s.replace('.','',1).isdigit()]
        if nums:
            target_re = nums[0]
        else:
            target_re = config["default_re"]
            
        # Hard constraint threshold boundary validator validation
        bounds = config["bounds"]
        if target_re < bounds[0] or target_re > bounds[1]:
            st.error(f"🚫 **Guardrail Exception:** Extracted value Re={target_re} falls outside trained parameter spectrum bounds {bounds}.")
            st.stop()

        st.markdown(f"""
        <div class='agent-bubble'>
        <strong>Step 1: NLP Intent Parsing & Bound Evaluation Passed</strong><br>
        • Parsed Target Asset Domain: <span class='code-span'>{target_case}</span><br>
        • Extracted System Variable Vector: <span class='code-span'>Re = {target_re}</span><br>
        • Status: Parameters verified inside safe operational thresholds.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='agent-bubble' style='background-color: #faf5ff; border-color: #e9d5ff;'>
        <strong>Step 2: Activating Tool Token Dispatch</strong><br>
        • Mapping execution to functional tool call macro: <span class='code-span'>agent_tool_reconstruct_field()</span><br>
        • Fetching neural tensors from weights checkpoint file and calculating matrices...
        </div>
        """, unsafe_allow_html=True)
        
        try:
            gx, gy, gU, gV, active_modes = agent_tool_reconstruct_field(target_case, target_re)
            
            st.markdown(f"""
            <div class='agent-bubble' style='background-color: #eff6ff; border-color: #bfdbfe;'>
            <strong>Step 3: Linear Grid Spline Resampling & Output Generation Complete</strong><br>
            • Reconstructed spatial arrays configurations using <span class='code-span'>{active_modes}</span> active components.<br>
            • Unstructured dataset layers translated into high-density grid fields matrix successfully.
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📝 Synthesized Engineering Report")
            st.markdown(f"""
            <div class='report-box'>
            <strong>AUTOMATED ANALYSIS CASE DOSSIER: {target_case.upper()}</strong><br>
            • <strong>Input Conditions:</strong> Flow velocity profile index set at Reynolds parameter = {target_re:.2f}<br>
            • <strong>Reduced Latent Dimensions:</strong> {active_modes} primary energy components active.<br>
            • <strong>Performance Index:</strong> Tensor space calculated instantly in milliseconds, replacing long multi-hour traditional Navier-Stokes operations.
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state['results'] = (gx, gy, gU, gV, target_case, target_re)
            
        except Exception as e:
            st.error(f"💥 **Pipeline Interruption Error:** {str(e)}")
    else:
        st.write("Awaiting engineering commands stream input...")

with col_right:
    st.header("📊 Scientific Graphics Canvas")
    
    if 'results' in st.session_state and process_btn:
        gx, gy, gU, gV, t_case, t_re = st.session_state['results']
        
        # High-Fidelity Plotly Continuous Heat Contours for U
        fig_u = go.Figure(data=[
            go.Contour(
                x=gx, y=gy, z=gU,
                colorscale="RdBu_r",
                line_width=0.2,
                contours=dict(coloring='heatmap', showlines=True),
                colorbar=dict(title="U Velocity Component", titleside="top")
            )
        ])
        fig_u.update_layout(
            title=f"AI Contour Field (Horizontal U Fluid Direction) | Re = {t_re}",
            xaxis_title="X Coordinates Matrix Grid", yaxis_title="Y Coordinates Matrix Grid",
            margin=dict(l=30, r=30, t=50, b=30), height=380, plot_bgcolor="white"
        )
        st.plotly_chart(fig_u, use_container_width=True)
        
        # High-Fidelity Plotly Continuous Heat Contours for V
        fig_v = go.Figure(data=[
            go.Contour(
                x=gx, y=gy, z=gV,
                colorscale="Balance",
                line_width=0.2,
                contours=dict(coloring='heatmap', showlines=True),
                colorbar=dict(title="V Velocity Component", titleside="top")
            )
        ])
        fig_v.update_layout(
            title=f"AI Contour Field (Vertical V Fluid Direction) | Re = {t_re}",
            xaxis_title="X Coordinates Matrix Grid", yaxis_title="Y Coordinates Matrix Grid",
            margin=dict(l=30, r=30, t=50, b=30), height=380, plot_bgcolor="white"
        )
        st.plotly_chart(fig_v, use_container_width=True)
        
    else:
        st.warning("Canvas ready. Real-time scientific contours will paint here upon executing agent requests.")
