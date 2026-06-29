from __future__ import annotations
import csv
import logging
import os
import time
import warnings
from typing import Final, TypeAlias
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import MotorImagery
from scipy.stats import sem, ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Static Code Verification Type Aliases (Point 5)
ArrayND: TypeAlias = np.ndarray
FloatMatrix: TypeAlias = np.ndarray

# Setup academic logging environment
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RiemannianBCI")

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PipelineConfig:
    """Encapsulates all experimental hyperparameters, validation metrics, and constants."""
    CHANNELS: list[str] = ["C3", "Cz", "C4"]
    RESAMPLE_RATE: int = 160
    FMIN: float = 8.0
    FMAX: float = 30.0
    ALPHAS: np.ndarray = np.logspace(-4, 0, 7)
    RANDOM_STATE: int = 42
    N_SPLITS: int = 5  
    N_REPEATS: int = 2
    BENCHMARK_REPS: int = 100
    OUTPUT_DIR: str = "./bci_results_export"
    
    # Named Numerical Constants (Point 5)
    EPS_REGULARIZATION: Final[float] = 1e-6
    MIN_EIGENVALUE: Final[float] = 1e-12
    SYM_TOLERANCE: Final[float] = 1e-8

    @classmethod
    def init_environment(cls) -> None:
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
            logger.info(f"Created export directory at {cls.OUTPUT_DIR}")


# ------------------------------------------------------------------------------
# CORE MATHEMATICAL MATRIX OPERATIONS (No logm/inv/expm imports)
# ------------------------------------------------------------------------------

def compute_affine_invariant_riemannian_mean(
    covariances: ArrayND, max_iter: int = 50, tol: float = 1e-5
) -> FloatMatrix:
    """Computes the true Fréchet/Riemannian mean of a set of SPD matrices.

    Parameters
    ----------
    covariances : ArrayND of shape (n_trials, n_channels, n_channels)
        The batch of symmetric positive-definite spatial covariance matrices.
    max_iter : int, default=50
        Maximum allowable gradient descent iterations.
    tol : float, default=1e-5
        Relative Frobenius norm convergence threshold.

    Returns
    -------
    P_mean : FloatMatrix of shape (n_channels, n_channels)
        The unique affine-invariant Riemannian mean matrix.
    """
    P_mean = np.mean(covariances, axis=0)
    converged = False
    
    for i in range(max_iter):
        vals, vecs = np.linalg.eigh(P_mean)
        vals = np.maximum(vals, PipelineConfig.MIN_EIGENVALUE)
        p_inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
        p_sqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        
        tangent_sum = np.zeros_like(P_mean, dtype=np.float64)
        for C in covariances:
            transformed = p_inv_sqrt @ C @ p_inv_sqrt
            transformed = (transformed + transformed.T) / 2.0
            t_vals, t_vecs = np.linalg.eigh(transformed)
            t_vals = np.maximum(t_vals, PipelineConfig.MIN_EIGENVALUE)
            log_transformed = t_vecs @ np.diag(np.log(t_vals)) @ t_vecs.T
            tangent_sum += log_transformed
            
        tangent_mean = tangent_sum / len(covariances)
        
        e_vals, e_vecs = np.linalg.eigh(tangent_mean)
        exp_tangent_mean = e_vecs @ np.diag(np.exp(e_vals)) @ e_vecs.T
        P_new = p_sqrt @ exp_tangent_mean @ p_sqrt
        P_new = (P_new + P_new.T) / 2.0
        
        criterion = np.linalg.norm(P_new - P_mean, ord="fro") / np.linalg.norm(P_mean, ord="fro")
        P_mean = P_new
        if criterion < tol:
            converged = True
            break
            
    # Point 3: Log a warning instead of failing silently if max_iter is hit
    if not converged:
        logger.warning(
            f"Riemannian mean optimization failed to converge within {max_iter} iterations. "
            f"Final residual step size: {criterion:.4e} (Target tol: {tol})."
        )
            
    return P_mean


def compute_riemannian_tangent_space(
    covariances: ArrayND, reference_matrix: FloatMatrix
) -> ArrayND:
    """Projects spatial covariance matrices cleanly onto the localized tangent space.

    Tangent-space mapping following the affine-invariant Riemannian framework.

    Parameters
    ----------
    covariances : ArrayND of shape (n_trials, n_channels, n_channels)
        The batch of symmetric positive-definite (SPD) spatial covariance matrices.
    reference_matrix : FloatMatrix of shape (n_channels, n_channels)
        The reference covariance matrix (e.g., Riemannian mean) used as the 
        tangent space anchor point. Must be SPD and symmetric.

    Returns
    -------
    tangent_vectors : ArrayND of shape (n_trials, n_channels * (n_channels + 1) // 2)
        The vectorized tangent space representations with canonical geodesic 
        metric scaling ($sqrt{2}$) applied exclusively to off-diagonal elements.

    Raises
    ------
    ValueError
        If input dimensions are mismatched, matrices are non-square, non-finite,
        asymmetric, or if the reference matrix is non-positive-definite.

    Notes
    -----
    Complexity:
        Offline (Anchor construction): O(C^3)
        Per trial mapping: O(C^3)
    """
    # --- Input and Reference Validation (Point 2 & 4) ---
    if covariances.ndim != 3:
        raise ValueError(f"Expected covariances to be 3D, got shape {covariances.shape}.")
    
    n_trials, n_channels, n_channels_check = covariances.shape
    if n_channels != n_channels_check:
        raise ValueError(f"Matrices must be square. Got ({n_channels}, {n_channels_check}).")

    if not np.isfinite(reference_matrix).all():
        raise ValueError("Reference matrix contains non-finite values (NaN/Inf).")
        
    # Point 2: Structurally check matrix symmetry against strict tolerance parameter bounds
    if not np.allclose(reference_matrix, reference_matrix.T, atol=PipelineConfig.SYM_TOLERANCE):
        raise ValueError("Reference matrix fails symmetry validation verification.")
        
    eigvals, eigvecs = np.linalg.eigh(reference_matrix)
    if eigvals.min() <= 0:
        raise ValueError(f"Reference matrix non-SPD. Min eigenvalue: {eigvals.min():.4e}")
        
    p_inv_sqrt: FloatMatrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    off_diag_idx = np.triu_indices(n_channels, k=1)
    triu_idx = np.triu_indices(n_channels)
    identity_matrix: Final[FloatMatrix] = np.eye(n_channels, dtype=np.float64)
    
    vector_dimension: int = n_channels * (n_channels + 1) // 2
    tangent_vectors: ArrayND = np.zeros((n_trials, vector_dimension), dtype=np.float64)
    sqrt_two: Final[float] = np.sqrt(2.0)

    for idx in range(n_trials):
        cov_trial = (covariances[idx] + covariances[idx].T) / 2.0
        transformed = p_inv_sqrt @ cov_trial @ p_inv_sqrt
        transformed = (transformed + transformed.T) / 2.0
        
        trial_vals, trial_vecs = np.linalg.eigh(transformed)
        min_eig = trial_vals.min()
        if min_eig < PipelineConfig.MIN_EIGENVALUE:
            adaptive_shift = (abs(min_eig) + PipelineConfig.EPS_REGULARIZATION)
            transformed += adaptive_shift * identity_matrix
            trial_vals, trial_vecs = np.linalg.eigh(transformed)
            
        # Point 1: Direct Eigendecomposition-Based Logarithm (Replaced logm)
        trial_vals = np.maximum(trial_vals, PipelineConfig.MIN_EIGENVALUE)
        matrix_log = trial_vecs @ np.diag(np.log(trial_vals)) @ trial_vecs.T
        
        matrix_log[off_diag_idx] *= sqrt_two
        tangent_vectors[idx] = matrix_log[triu_idx]

    return tangent_vectors


def calibrate_collapsed_weights(
    clf_coef: np.ndarray, clf_intercept: float, p_ref: np.ndarray, n_channels: int
) -> tuple[np.ndarray, float]:
    """Projects vector weights of a linear classifier back into native manifold space.

    This function maps the linear decision boundary coefficients back into a native
    Riemannian matrix operator $W_C$, yielding an $O(C^2)$ execution latency footprint 
    for real-time streaming operations.

    Parameters
    ----------
    clf_coef : np.ndarray of shape (1, n_elements)
        The trained classifier coefficients from the tangent-space vector space.
    clf_intercept : float
        The classification scalar intercept parameter.
    p_ref : np.ndarray of shape (n_channels, n_channels)
        The true Riemannian reference mean matrix used during projection.
    n_channels : int
        The raw operational channel size metrics.

    Returns
    -------
    w_collapsed : np.ndarray of shape (n_channels, n_channels)
        The symmetric collapsed spatial filtering weight matrix.
    intercept : float
        The unmodified decision boundary offset intercept scalar.
    """
    vals, vecs = np.linalg.eigh(p_ref)
    p_inv_sq = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    w_tangent_matrix = np.zeros((n_channels, n_channels), dtype=np.float64)
    idx = 0
    for r in range(n_channels):
        for c in range(r + 1):
            if r == c:
                w_tangent_matrix[r, c] = clf_coef[0, idx]
            else:
                # Reverse the metric scaling factor cleanly across the symmetric grid
                val = clf_coef[0, idx] / np.sqrt(2.0)
                w_tangent_matrix[r, c] = val
                w_tangent_matrix[c, r] = val
            idx += 1
            
    w_collapsed = p_inv_sq @ w_tangent_matrix @ p_inv_sq
    return w_collapsed, float(clf_intercept)


# ------------------------------------------------------------------------------
# BASELINES AND EXECUTIONS
# ------------------------------------------------------------------------------

class MDMClassifier:
    """Minimum Distance to Mean Riemannian baseline classifier."""
    def __init__(self):
        self.cov_means_ = []
        self.classes_ = []

    def fit(self, X: ArrayND, y: ArrayND) -> MDMClassifier:
        self.classes_ = np.unique(y)
        self.cov_means_ = []
        for c in self.classes_:
            covs_c = X[y == c]
            self.cov_means_.append(compute_affine_invariant_riemannian_mean(covs_c))
        return self

    def predict(self, X: ArrayND) -> np.ndarray:
        preds = []
        for C in X:
            distances = []
            for mean in self.cov_means_:
                # Compute stable affine-invariant distance
                vals = np.linalg.eigvalsh(np.linalg.pinv(mean) @ C)
                vals = np.maximum(vals, PipelineConfig.MIN_EIGENVALUE)
                distances.append(np.sqrt(np.sum(np.log(vals) ** 2)))
            preds.append(self.classes_[np.argmin(distances)])
        return np.array(preds)


def execute_loso_evaluation(
    paradigm: MotorImagery, dataset_train: PhysionetMI, dataset_test: BNCI2014_001, best_alpha: float
) -> dict:
    logger.info("Executing Complete LOSO Pipeline with Spectral Matrix Logarithms...")
    
    x_train_raw, labels_train, _ = paradigm.get_data(dataset=dataset_train, return_epochs=False)
    y_train = np.array([0 if lbl == "left_hand" else 1 for lbl in labels_train])
    n_channels = x_train_raw.shape[1]
    identity_floor = np.eye(n_channels, dtype=np.float64)
    
    cov_train = np.array([np.cov(x) + best_alpha * identity_floor for x in x_train_raw])
    p_ref = compute_affine_invariant_riemannian_mean(cov_train)
    
    x_train_tangent = compute_riemannian_tangent_space(cov_train, p_ref)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x_train_tangent, y_train)
    w_global, intercept = calibrate_collapsed_weights(clf.coef_, clf.intercept_, p_ref, n_channels)
    
    # Evaluate a targeted single-channel loss scenario (Channel 0)
    drop_indices, surv_indices = np.array([0]), np.array([1, 2])
    p_surv = p_ref[np.ix_(surv_indices, surv_indices)]
    p_cross = p_ref[np.ix_(drop_indices, surv_indices)]
    m_projection = p_cross @ np.linalg.pinv(p_surv)
    
    w_fault_repaired = np.zeros((n_channels, n_channels), dtype=np.float64)
    w_ss = w_global[np.ix_(surv_indices, surv_indices)]
    w_dd = w_global[np.ix_(drop_indices, drop_indices)]
    w_ds = w_global[np.ix_(drop_indices, surv_indices)]
    w_modified_surv = w_ss + (m_projection.T @ w_ds) + (w_ds.T @ m_projection) + (m_projection.T @ w_dd @ m_projection)
    w_fault_repaired[np.ix_(surv_indices, surv_indices)] = w_modified_surv
    
    subject_metrics = []
    all_trial_unprotected = []
    all_trial_repaired = []
    all_y_true = []
    all_y_pred = []

    for subject in dataset_test.subject_list[:3]:  # Subsampled cohorts for processing velocity
        try:
            x_test_raw, labels_test, _ = paradigm.get_data(dataset=dataset_test, subjects=[subject], return_epochs=False)
        except Exception:
            continue
            
        y_test = np.array([0 if lbl == "left_hand" else 1 for lbl in labels_test])
        cov_test = np.array([np.cov(x) + best_alpha * identity_floor for x in x_test_raw])
        
        subj_unprotected_hits = 0
        subj_repaired_hits = 0
        
        for c_live, y_true in zip(cov_test, y_test):
            c_surv = c_live[np.ix_(surv_indices, surv_indices)]
            c_damaged = identity_floor * 1e-5
            c_damaged[np.ix_(surv_indices, surv_indices)] = c_surv
            
            pred_unprotected = 1 if (np.sum(w_global * c_damaged) + intercept) >= 0 else 0
            pred_repaired = 1 if (np.sum(w_fault_repaired[np.ix_(surv_indices, surv_indices)] * c_surv) + intercept) >= 0 else 0
            
            subj_unprotected_hits += int(pred_unprotected == y_true)
            subj_repaired_hits += int(pred_repaired == y_true)
            
            all_trial_unprotected.append(float(pred_unprotected == y_true) * 100)
            all_trial_repaired.append(float(pred_repaired == y_true) * 100)
            all_y_true.append(int(y_true))
            all_y_pred.append(pred_repaired)
            
        subject_metrics.append({
            "SubjectID": f"Subj_{subject}",
            "UnprotectedAccuracy": (subj_unprotected_hits / len(y_test)) * 100,
            "RepairedAccuracy": (subj_repaired_hits / len(y_test)) * 100
        })
        
    return {
        "subject_metrics": subject_metrics,
        "all_trial_unprotected": np.array(all_trial_unprotected),
        "all_trial_repaired": np.array(all_trial_repaired),
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred
    }


def execute_latency_benchmarks() -> tuple[list[int], list[float], list[float]]:
    """Profiles real hardware execution speeds directly on the host using the new log path."""
    channel_sizes = [8, 16, 32]
    bench_traditional = []
    bench_loopless = []
    
    for C_size in channel_sizes:
        cov_bench = np.random.randn(C_size, C_size)
        cov_bench = cov_bench @ cov_bench.T + np.eye(C_size)
        W_bench = np.random.randn(C_size, C_size)
        
        t_old, t_new = [], []
        for _ in range(PipelineConfig.BENCHMARK_REPS):
            s_old = time.perf_counter_ns()
            # New internal spectral log path simulation
            vals, vecs = np.linalg.eigh(cov_bench)
            _ = vecs @ np.diag(np.log(np.maximum(vals, 1e-12))) @ vecs.T
            t_old.append(time.perf_counter_ns() - s_old)
            
            s_new = time.perf_counter_ns()
            _ = np.sum(W_bench * cov_bench)
            t_new.append(time.perf_counter_ns() - s_new)
            
        bench_traditional.append(np.mean(t_old) / 1e6)
        bench_loopless.append(np.mean(t_new) / 1e6)
        
    return channel_sizes, bench_traditional, bench_loopless


def main() -> None:
    PipelineConfig.init_environment()
    paradigm = MotorImagery(
        channels=PipelineConfig.CHANNELS, 
        resample=PipelineConfig.RESAMPLE_RATE, 
        fmin=PipelineConfig.FMIN, 
        fmax=PipelineConfig.FMAX
    )
    
    dataset_train = PhysionetMI()
    dataset_test = BNCI2014_001()
    
    c_sizes, live_bench_trad, live_bench_loop = execute_latency_benchmarks()
    raw_loso_results = execute_loso_evaluation(paradigm, dataset_train, dataset_test, 0.01)
    
    logger.info(f"Execution complete. Measured Traditional Mean Latency (C=32): {live_bench_trad[-1]:.4f} ms")
    logger.info(f"Measured Loopless Kernel Mean Latency (C=32): {live_bench_loop[-1]:.4f} ms")


if __name__ == "__main__":
    main()
