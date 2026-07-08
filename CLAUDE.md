# POD-FCDNN Surrogate Modeling App — Project Context

## What this project is
A Streamlit dashboard for a POD (Proper Orthogonal Decomposition) + FCDNN
(Fully Connected Deep Neural Network) surrogate model for CFD flow fields.
It's an MS thesis project (Computer Science, NUST SEECS) — the original
research/model code came from a colleague; my assigned work is entirely on
the application layer: visuals, GUI, graphics, and speed. Not doing new CFD
or ML research — the thesis contribution is the app itself.

Four benchmark cases, each with a pretrained checkpoint in `checkpoints/`:
- **Cavity** (lid-driven cavity, parameterized by Reynolds number)
- **Cylinder** (flow over a cylinder, parameterized by Re)
- **Backward Facing Step / BFS** (parameterized by Re)
- **NACA0012** (airfoil, parameterized by angle of attack α, not Re)

## How the underlying model works (engine.py — mostly untouched)
1. POD: SVD on training snapshots reduces each full flow field (tens of
   thousands to ~440k mesh nodes × u,v,p) down to a handful of coefficients
   (r=4 for Cavity/Cylinder/BFS, r=7 for NACA0012).
2. FCDNN: a small network (width 128, 4 hidden layers) maps the scalar
   parameter (log(Re) or α) → those r POD coefficients.
3. Reconstruction: coefficients × POD basis → full u, v, p field.

Key functions in `engine.py`:
- `load_checkpoint(path)` → returns a trainer object with `.pod` (mean, Phi,
  svals, xy, N, r) and `.model`
- `predict_and_reconstruct(trainer, param)` → returns dict with u, v, p, xy
- `compute_errors(truth, pred)` → MAE, L2, L2_rel%, max error
- `infer_architecture(state_dict)` → introspects width/depth/out_dim since
  these aren't stored explicitly in the checkpoint
- `subsample_indices(n_points, cap=45000)` → fixed reproducible subsample for
  large meshes (BFS has 442k nodes — full Delaunay triangulation on all of
  them is slow; capping keeps rendering fast without visible quality loss)
- `load_snapshot_uvp_from_buffer(buffer)` → parses an uploaded ground-truth
  `.dat`/`.csv` snapshot (file-like object, for the Validation tab)

**Bug fixed**: `save_checkpoint()` used to write a nested `pod_dict`, but
`load_checkpoint()` expects flat keys (`pod_mean`, `pod_phi`, etc.). Existing
checkpoints load fine (saved differently originally), but retraining +
re-saving would have broken loading. Now fixed to write flat keys.

## Current app.py structure
Modern dark-themed Streamlit dashboard:
- **Hero header** + custom CSS (Space Grotesk/Inter fonts, gradient text,
  hidden Streamlit chrome, card-styled metrics) — injected via
  `st.markdown(..., unsafe_allow_html=True)`
- **Sidebar**: case selector (with emoji icons), Re/α slider, rendering
  options (grid resolution, LIC streak length, vector overlay toggle),
  Predict button
- **KPI row**: mesh nodes, POD modes, network width/depth — always visible,
  cheap to compute (no prediction needed)
- **Tabs**:
  1. 🌈 **Flow Visualization** — the hero render. Uses Line Integral
     Convolution (LIC, via the `lic` pip package) composited with a jet
     colormap by velocity magnitude — matches classic CFD post-processing
     tool aesthetics (ParaView/Ansys-style). Rendered with matplotlib,
     displayed via `st.pyplot`.
  2. 🔎 **Field Explorer** — interactive Plotly contour panels for
     pressure, u, v, |V|, dark theme (`plotly_dark`), optional quiver vector
     overlay, CSV export of predicted field.
  3. 📊 **Model Diagnostics** — POD energy spectrum (cumulative %), singular
     value decay (log scale), network architecture info.
  4. ✅ **Validate vs Ground Truth** — upload a real snapshot, get per-field
     error metrics (MAE/L2/L2_rel%/max) + truth/prediction/difference
     contour comparison. Handles node-count mismatches via interpolation.

## Rendering pipeline / performance strategy
Scattered CFD mesh nodes → smooth field images via:
1. `Delaunay` triangulation of node xy coordinates (scipy.spatial) — cached
   per case via `st.cache_resource` since geometry doesn't change with
   Re/α, only field *values* do. This is the expensive step (~6.5s for BFS
   at full 442k nodes), so it's cached and mesh is subsampled to 45k nodes
   cap for large cases.
2. `LinearNDInterpolator` onto a regular grid for contour/LIC rendering.
3. Hole/exterior masking: KD-tree nearest-neighbor distance check — grid
   points far from any real mesh node (e.g. inside the cylinder body, or
   outside a non-rectangular domain) get masked to NaN so they don't show
   fake interpolated data.
4. Predictions persist in `st.session_state` across tab switches / widget
   changes so results aren't lost by touching an unrelated control.

## Visual style decisions made so far
- Reference images the user liked: jet/rainbow colormap CFD visualizations
  with streamlines (classic textbook/ParaView look), and one 3D glossy
  render (acknowledged as literally a 3D ray-traced render — not
  replicable from 2D data without misrepresenting it, so we extracted the
  achievable technique instead: LIC).
- Went with **dark theme** overall (`.streamlit/config.toml` base="dark",
  primary color cyan `#22d3ee`) to match the CFD-tool aesthetic in
  reference images.
- `jet` colormap used deliberately (not `turbo`) to match the classic CFD
  tool look the user referenced, despite `jet` being perceptually
  non-uniform — this was an explicit user preference, not an oversight.

## Known open items / next steps (not yet done)
- LIC has only been generated as static preview images for all 4 cases so
  far; wired into the live app for Cavity/Cylinder specifically tested,
  should verify BFS and NACA0012 render correctly + performantly in the
  actual running app (not just standalone scripts).
- Performance benchmarking of the live Streamlit app end-to-end (predict
  button → LIC render) hasn't been measured for all 4 cases yet — BFS is
  the largest mesh (442k nodes) and most at risk of feeling slow even after
  subsampling.
- Possible future asks the user has mentioned wanting: animated Re-sweep
  GIF/video export, streamline density tuning, "export publication-quality
  figure" button, further dashboard polish.
- User is NOT looking to add CFD/ML research contributions (e.g. UQ,
  nonlinear ROM, neural operators) — that direction was discussed and
  explicitly declined in favor of pure app/UI/visualization work. Don't
  reintroduce research-scope suggestions unless asked.

## Environment
- `pip install -r requirements.txt` (includes streamlit, numpy, pandas,
  torch, matplotlib, plotly, scikit-learn, scipy, lic)
- Run: `streamlit run app.py`
- `pip install --break-system-packages` may be needed in some sandboxed
  Linux environments (not typically needed on a normal local machine).
