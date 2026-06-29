from __future__ import annotations
import gc
import time
import logging
from typing import Final, TypeAlias
import numpy as np
from scipy.linalg import eigh as scipy_eigh

# Type Aliases for code clarity
MatrixBatch: TypeAlias = np.ndarray
FloatMatrix: TypeAlias = np.ndarray

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BCIBenchmark")

# ------------------------------------------------------------------------------
# 1. CORE ALGORITHMS & BASELINES IMPLEMENTATION
# ------------------------------------------------------------------------------

def compute_affine_invariant_riemannian_mean(covariances: MatrixBatch, max_iter: int = 30, tol: float = 1e-5) -> FloatMatrix:
    """Computes the true Fréchet/Riemannian mean of a set of SPD matrices."""
    P_mean = np.mean(covariances, axis=0)
    for _ in range(max_iter):
        vals, vecs = np.linalg.eigh(P_mean)
        vals = np.maximum(vals, 1e-12)
        p_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        p_sqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        
        tangent_sum = np.zeros_like(P_mean, dtype=np.float64)
        for C in covariances:
            transformed = (p_inv_sqrt @ C @ p_inv_sqrt + (p_inv_sqrt @ C @ p_inv_sqrt).T) / 2.0
            t_vals, t_vecs = np.linalg.eigh(transformed)
            log_transformed = t_vecs @ np.diag(np.log(np.maximum(t_vals, 1e-12))) @ t_vecs.T
            tangent_sum += log_transformed
            
        tangent_mean = tangent_sum / len(covariances)
        e_vals, e_vecs = np.linalg.eigh(tangent_mean)
        P_new = p_sqrt @ (e_vecs @ np.diag(np.exp(e_vals)) @ e_vecs.T) @ p_sqrt
        
        if np.linalg.norm(P_new - P_mean, ord="fro") / np.linalg.norm(P_mean, ord="fro") < tol:
            P_mean = P_new
            break
        P_mean = P_new
    return P_mean


class CommonSpatialPatterns:
    """Standard Common Spatial Patterns (CSP) spatial filter baseline implementation."""
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_ = None

    def fit(self, covariances: MatrixBatch, labels: np.ndarray) -> CommonSpatialPatterns:
        # Estimate class-conditional compound covariance matrices
        cov_0 = compute_affine_invariant_riemannian_mean(covariances[labels == 0])
        cov_1 = compute_affine_invariant_riemannian_mean(covariances[labels == 1])
        # Solve generalized eigenvalue problem
        vals, vecs = scipy_eigh(cov_0, cov_0 + cov_1)
        ix = np.argsort(np.abs(vals - 0.5))[::-1]
        self.filters_ = vecs[:, ix[:self.n_components]].T
        return self

    def transform(self, covariances: MatrixBatch) -> np.ndarray:
        # Map out variance log-features
        feats = np.zeros((len(covariances), self.n_components))
        for idx, C in enumerate(covariances):
            projected = self.filters_ @ C @ self.filters_.T
            feats[idx] = np.log(np.maximum(np.diag(projected), 1e-12))
        return feats


# ------------------------------------------------------------------------------
# 2. EVALUATION BENCHMARK SUITE RUNNER
# ------------------------------------------------------------------------------

class BCIPipelineBenchmarkingSuite:
    """Profiles computational throughput across CSP, MDM, Tangent Space, and your Loopless method."""
    
    def __init__(self, warmups: int = 5, repetitions: int = 30):
        self.warmups: Final[int] = warmups
        self.repetitions: Final[int] = repetitions
        # Dictionary structure: { Pipeline_Name: { Channel_Count: Average_Execution_Time_ms } }
        self.metrics: dict[str, dict[int, float]] = {
            "CSP + LDA Feature Map": {},
            "Traditional Tangent Space (TS)": {},
            "Minimum Distance to Mean (MDM)": {},
            "Your Loopless O(C2) Kernel": {}
        }

    @staticmethod
    def generate_synthetic_dataset(n_trials: int, n_channels: int) -> tuple[MatrixBatch, np.ndarray, FloatMatrix]:
        """Generates balanced target label metrics and realistic spatial covariance vectors."""
        np.random.seed(42)
        covs = np.zeros((n_trials, n_channels, n_channels), dtype=np.float64)
        for idx in range(n_trials):
            A = np.random.randn(n_channels, n_channels)
            covs[idx] = A @ A.T + np.eye(n_channels) * 0.2
        labels = np.array([0 if i < n_trials // 2 else 1 for i in range(n_trials)])
        ref_matrix = compute_affine_invariant_riemannian_mean(covs)
        return covs, labels, ref_matrix

    def run_evaluation(self, channel_scenarios: list[int], n_trials: int = 100) -> None:
        """Executes hardware latency monitoring loops across explicit pipeline blocks."""
        for C in channel_scenarios:
            covs, labels, ref_matrix = self.generate_synthetic_dataset(n_trials, C)
            
            # Pre-compile/Cache parameters for evaluation targets
            csp = CommonSpatialPatterns(n_components=min(4, C)).fit(covs, labels)
            mock_lda_weights = np.random.randn(C, C)
            
            # Compute tangent mappings setup
            vals, vecs = np.linalg.eigh(ref_matrix)
            p_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12))) @ vecs.T
            triu_idx = np.triu_indices(C)

            # --- PIPELINE 1: CSP FEATURE EXTRACTION ---
            self._profile_pipeline("CSP + LDA Feature Map", C, lambda: csp.transform(covs))

            # --- PIPELINE 2: TRADITIONAL TANGENT SPACE PROJECTING ---
            def traditional_ts():
                out = np.zeros((n_trials, C * (C + 1) // 2))
                for i in range(n_trials):
                    transformed = p_inv_sqrt @ covs[i] @ p_inv_sqrt
                    t_vals, t_
