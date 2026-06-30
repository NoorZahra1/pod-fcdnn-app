"""
POD-FCDNN Engine: Core machine learning and data processing logic.

This module provides:
- Data discovery and snapshot loading
- Proper Orthogonal Decomposition (POD)
- Fully Connected Deep Neural Network (FCDNN) training
- Inference and field reconstruction
"""

import re
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import dataclasses
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader


# ============================================================================
# Data Discovery & Loading
# ============================================================================

RE_FILE_PATTERN = re.compile(r"Re[\s_\-]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def discover_cases(base_dir: Path, ext: str = ".dat") -> Tuple[np.ndarray, list]:
    """
    Discover snapshot files in a directory by Reynolds number.
    
    Expects filenames like: "Re 1000.dat", "Re3000.dat", "RE_5000.dat", etc.
    
    Args:
        base_dir: Parent directory containing snapshot files
        ext: File extension to search for (default: ".dat")
    
    Returns:
        Tuple of:
        - Re_values: (M,) array of Reynolds numbers, sorted
        - file_paths: List of Path objects to snapshot files
    
    Raises:
        RuntimeError: If no matching files found in directory
    """
    base_dir = Path(base_dir)
    cases = []
    
    for fp in sorted(base_dir.iterdir()):
        if not fp.is_file():
            continue
        if fp.suffix.lower() != ext.lower():
            continue
        
        match = RE_FILE_PATTERN.search(fp.stem)
        if not match:
            continue
        
        Re = float(match.group(1))
        cases.append((Re, fp))
    
    if not cases:
        raise RuntimeError(
            f"No snapshot files found in {base_dir}. "
            f"Expected files like 'Re 1000{ext}'."
        )
    
    cases.sort(key=lambda x: x[0])
    Re_values = np.array([c[0] for c in cases], dtype=np.float64)
    file_paths = [c[1] for c in cases]
    
    return Re_values, file_paths


def load_snapshot_uvp(dat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single CFD snapshot file with velocity and pressure data.
    
    Expected columns: nodenumber, x-coordinate, y-coordinate, 
                      absolute-pressure, x-velocity, y-velocity
    
    Supports both comma-separated and whitespace-separated files.
    
    Args:
        dat_path: Path to the .dat snapshot file
    
    Returns:
        Tuple of:
        - xvec: (3N,) array = [u1, u2, ..., uN, v1, v2, ..., vN, p1, p2, ..., pN]
        - xy: (N, 2) array of [x, y] coordinates for each node
    
    Raises:
        ValueError: If required columns are missing
    """
    dat_path = Path(dat_path)
    
    # Try comma-separated first; fallback to whitespace
    df = pd.read_csv(dat_path)
    if df.shape[1] == 1:
        df = pd.read_csv(dat_path, delim_whitespace=True)
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    required_cols = [
        "nodenumber", "x-coordinate", "y-coordinate",
        "absolute-pressure", "x-velocity", "y-velocity"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {dat_path.name}: {missing}\n"
            f"Found: {list(df.columns)}"
        )
    
    # Sort by node number for consistency
    df = df.sort_values("nodenumber", kind="mergesort").reset_index(drop=True)
    
    u = df["x-velocity"].to_numpy(dtype=np.float64)
    v = df["y-velocity"].to_numpy(dtype=np.float64)
    p = df["absolute-pressure"].to_numpy(dtype=np.float64)
    xy = df[["x-coordinate", "y-coordinate"]].to_numpy(dtype=np.float64)
    
    xvec = np.hstack([u, v, p])  # (3N,)
    return xvec, xy


def load_training_snapshots(base_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all training snapshots from a directory.
    
    Args:
        base_dir: Directory containing Re*.dat files
    
    Returns:
        Tuple of:
        - X: (M, 3N) snapshot matrix (M snapshots, 3N state variables)
        - Re_values: (M,) Reynolds numbers
        - xy: (N, 2) spatial coordinates (from first snapshot)
    """
    Re_values, file_paths = discover_cases(base_dir, ext=".dat")
    
    X_list = []
    xy_ref = None
    
    for fp in file_paths:
        xvec, xy = load_snapshot_uvp(fp)
        if xy_ref is None:
            xy_ref = xy
        X_list.append(xvec)
    
    X = np.vstack(X_list)  # (M, 3N)
    return X, Re_values, xy_ref


# ============================================================================
# POD (Proper Orthogonal Decomposition)
# ============================================================================

@dataclass
class PODModel:
    """Container for POD decomposition data."""
    mean: np.ndarray      # (D,) - mean snapshot
    Phi: np.ndarray       # (D, r) - POD basis modes
    svals: np.ndarray     # (r,) - singular values
    xy: np.ndarray        # (N, 2) - spatial node coordinates
    N: int                # number of nodes
    r: int                # number of retained modes
    
    @property
    def energy_fraction(self) -> np.ndarray:
        """Cumulative energy fraction captured by each mode."""
        energy = self.svals ** 2
        energy_frac = energy / energy.sum()
        return np.cumsum(energy_frac)


def fit_pod(X: np.ndarray, r: int, xy: np.ndarray) -> PODModel:
    """
    Compute POD decomposition via SVD.
    
    Args:
        X: (M, D) snapshot matrix
        r: Number of modes to retain
        xy: (N, 2) spatial coordinates (N = D/3 for u,v,p)
    
    Returns:
        PODModel with fitted basis and singular values
    
    Raises:
        ValueError: If snapshot format is invalid
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    r_eff = min(r, Vt.shape[0])
    
    Phi = Vt.T[:, :r_eff]
    svals = S[:r_eff].copy()
    
    D = X.shape[1]
    if D % 3 != 0:
        raise ValueError("Snapshot dimension must be 3N for [u,v,p].")
    
    N = D // 3
    if xy.shape[0] != N:
        raise ValueError(f"Coordinate mismatch: expected {N} nodes, got {xy.shape[0]}.")
    
    return PODModel(mean=mean, Phi=Phi, svals=svals, xy=xy, N=N, r=r_eff)


def pod_project(pod: PODModel, X: np.ndarray) -> np.ndarray:
    """
    Project snapshots onto POD basis.
    
    Args:
        pod: Fitted PODModel
        X: (M, D) snapshot matrix
    
    Returns:
        A: (M, r) POD coefficient matrix
    """
    Xc = X - pod.mean
    return Xc @ pod.Phi


def pod_reconstruct_field(pod: PODModel, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct velocity and pressure fields from POD coefficients.
    
    Args:
        pod: Fitted PODModel
        coeffs: (r,) or (M, r) POD coefficients
    
    Returns:
        Tuple of:
        - u: x-velocity field
        - v: y-velocity field
        - p: absolute pressure field
    """
    if coeffs.ndim == 1:
        coeffs = coeffs[np.newaxis, :]
    
    X_recon = coeffs @ pod.Phi.T + pod.mean  # (M, 3N)
    
    N = pod.N
    u = X_recon[:, :N]
    v = X_recon[:, N:2*N]
    p = X_recon[:, 2*N:3*N]
    
    return u.squeeze(), v.squeeze(), p.squeeze()


# ============================================================================
# Neural Network (FCDNN)
# ============================================================================

class FCDNN(nn.Module):
    """
    Fully Connected Deep Neural Network for POD coefficient prediction.
    
    Input: log(Reynolds) -> Output: POD coefficients
    """
    
    def __init__(self, out_dim: int, width: int = 128, depth: int = 4):
        """
        Args:
            out_dim: Output dimension (number of POD modes)
            width: Hidden layer width
            depth: Number of hidden layers
        """
        super().__init__()
        layers = [nn.Linear(1, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 1) -> (batch, out_dim)"""
        return self.net(x)


class PODFCDNNTrainer:
    """Trainer for POD-FCDNN surrogate model."""
    
    def __init__(
        self,
        pod: PODModel,
        Re_values: np.ndarray,
        pod_coeffs: np.ndarray,
        nn_width: int = 128,
        nn_depth: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        """
        Initialize trainer.
        
        Args:
            pod: Fitted PODModel
            Re_values: (M,) Reynolds numbers
            pod_coeffs: (M, r) POD coefficients
            nn_width: NN hidden layer width
            nn_depth: NN depth
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            device: "cpu" or "cuda"
        """
        self.pod = pod
        self.device = torch.device(device)
        self.history = {"loss": []}
        
        # Prepare training data
        self.x_train = np.log(Re_values).reshape(-1, 1).astype(np.float32)
        self.y_train = pod_coeffs.astype(np.float32)
        
        # Normalize
        self.x_mean = float(self.x_train.mean())
        self.x_std = float(self.x_train.std() + 1e-8)
        self.y_mean = self.y_train.mean(axis=0)
        self.y_std = self.y_train.std(axis=0) + 1e-8
        
        x_norm = (self.x_train - self.x_mean) / self.x_std
        y_norm = (self.y_train - self.y_mean) / self.y_std
        
        self.x_tensor = torch.tensor(x_norm, device=self.device)
        self.y_tensor = torch.tensor(y_norm, device=self.device)
        
        # Model
        self.model = FCDNN(
            out_dim=pod.r,
            width=nn_width,
            depth=nn_depth
        ).to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
    
    def train(
        self,
        epochs: int,
        batch_size: int = 32,
        verbose: bool = True,
        callback=None
    ) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Print loss every 100 epochs
            callback: Optional callback(epoch, loss) for progress tracking
        
        Returns:
            Dictionary with training history
        """
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            self.history["loss"].append(avg_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}  Loss: {avg_loss:.6e}")
            
            if callback:
                callback(epoch + 1, avg_loss)
        
        return self.history
    
    def predict_coefficients(self, Re: float) -> np.ndarray:
        """
        Predict POD coefficients for a given Reynolds number.
        
        Args:
            Re: Reynolds number
        
        Returns:
            (r,) array of POD coefficients
        """
        x_in = np.log(np.array([[Re]], dtype=np.float32))
        x_norm = (x_in - self.x_mean) / self.x_std
        
        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(torch.tensor(x_norm, device=self.device))
            y_norm = y_norm.cpu().numpy()
        
        # Denormalize
        coeffs = y_norm * self.y_std + self.y_mean
        return coeffs.ravel()


# ============================================================================
# High-Level Inference & Error Metrics
# ============================================================================

def predict_and_reconstruct(
    trainer: PODFCDNNTrainer,
    Re_query: float
) -> Dict[str, Any]:
    """
    Predict flow fields at a given Reynolds number.
    
    Args:
        trainer: Trained PODFCDNNTrainer instance
        Re_query: Query Reynolds number
    
    Returns:
        Dictionary containing:
        - "u": x-velocity field (N,)
        - "v": y-velocity field (N,)
        - "p": pressure field (N,)
        - "xy": spatial coordinates (N, 2)
        - "Re": actual Re used
    """
    coeffs = trainer.predict_coefficients(Re_query)
    u, v, p = pod_reconstruct_field(trainer.pod, coeffs)
    
    return {
        "u": u,
        "v": v,
        "p": p,
        "xy": trainer.pod.xy,
        "Re": Re_query
    }


def compute_errors(
    ref: np.ndarray,
    pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute error metrics between reference and predicted fields.
    
    Args:
        ref: Reference field (N,)
        pred: Predicted field (N,)
    
    Returns:
        Dictionary with error metrics:
        - "MAE": Mean Absolute Error
        - "L2": L2 norm error
        - "L2_rel": Relative L2 error (%)
        - "max": Maximum absolute error
    """
    error = np.abs(pred - ref)
    ref_norm = np.linalg.norm(ref)
    pred_error_norm = np.linalg.norm(pred - ref)
    
    return {
        "MAE": float(np.mean(error)),
        "L2": float(pred_error_norm),
        "L2_rel": float(100 * pred_error_norm / (ref_norm + 1e-12)),
        "max": float(np.max(error)),
    }


# ============================================================================
# Serialization (Updated for PyTorch 2.6+ weights_only=True Compliance)
# ============================================================================

def save_checkpoint(trainer: PODFCDNNTrainer, path: Path) -> None:
    """Save trained model and POD data using primitive types for maximum safety."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state": trainer.model.state_dict(),
        # Convert the PODModel dataclass cleanly to a standard python dict of arrays
        "pod_dict": dataclasses.asdict(trainer.pod),
        "x_mean": trainer.x_mean,
        "x_std": trainer.x_std,
        "y_mean": trainer.y_mean,
        "y_std": trainer.y_std,
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, device: str = "cpu") -> PODFCDNNTrainer:

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    pod = PODModel(
        mean=checkpoint["pod_mean"],
        Phi=checkpoint["pod_phi"],
        svals=checkpoint["pod_svals"],
        xy=checkpoint["pod_xy"],
        N=checkpoint["pod_N"],
        r=checkpoint["pod_r"]
    )

    model = FCDNN(
        out_dim=pod.r
     
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    trainer = PODFCDNNTrainer.__new__(
        PODFCDNNTrainer
    )

    trainer.pod = pod
    trainer.model = model
    trainer.device = torch.device(device)

    trainer.x_mean = checkpoint["x_mean"]
    trainer.x_std = checkpoint["x_std"]

    trainer.y_mean = checkpoint["y_mean"]
    trainer.y_std = checkpoint["y_std"]

    return trainer








