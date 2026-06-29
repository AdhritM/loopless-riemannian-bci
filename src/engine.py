from __future__ import annotations
from typing import Final, TypeAlias
import numpy as np
from scipy.linalg import logm

# Static Code Verification Type Aliases
ArrayND: TypeAlias = np.ndarray
FloatMatrix: TypeAlias = np.ndarray

# Named Numerical Constants (Point 5)
EPS_REGULARIZATION: Final[float] = 1e-6
MIN_EIGENVALUE: Final[float] = 1e-12


def compute_riemannian_tangent_space(
    covariances: ArrayND, reference_matrix: FloatMatrix
) -> ArrayND:
    # --- 1. Input Validation (Point 3) ---
    if covariances.ndim != 3:
        raise ValueError(
            f"Expected 'covariances' to be 3-dimensional (trials, channels, channels), "
            f"but got shape {covariances.shape}."
        )
    
    n_trials, n_channels, n_channels_check = covariances.shape
    if n_channels != n_channels_check:
        raise ValueError(
            f"Covariance matrices must be square. Got shape ({n_channels}, {n_channels_check})."
        )
        
    if reference_matrix.shape != (n_channels, n_channels):
        raise ValueError(
            f"Dimension mismatch: 'reference_matrix' shape {reference_matrix.shape} "
            f"must match covariance spatial dimensions ({n_channels}, {n_channels})."
        )

    # --- 2. Stable Eigendecomposition-Based Inverse Square Root (Point 1 & 4) ---
    # Replaced sqrtm + inv with eigh for Symmetric Positive-Definite (SPD) efficiency
    eigvals, eigvecs = np.linalg.eigh(reference_matrix)
    
    # Floor negative/tiny eigenvalues before inversion to maintain stability
    eigvals = np.maximum(eigvals, MIN_EIGENVALUE)
    p_inv_sqrt: FloatMatrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # --- 3. Pre-compute and Cache Index Arrays (Point 6 & 7) ---
    diag_idx = np.diag_indices(n_channels)
    off_diag_idx = np.triu_indices(n_channels, k=1)
    triu_idx = np.triu_indices(n_channels)  # Cached for final vector extraction
    
    vector_dimension: int = n_channels * (n_channels + 1) // 2
    tangent_vectors: ArrayND = np.zeros((n_trials, vector_dimension))
    sqrt_two: Final[float] = np.sqrt(2.0)

    # --- 4. Main Processing Loop ---
    for idx in range(n_trials):
        # Symmetric congruence transformation
        transformed: FloatMatrix = p_inv_sqrt @ covariances[idx] @ p_inv_sqrt
        
        # Enforce strict structural symmetry
        transformed = (transformed + transformed.T) / 2.0
        
        # Conditional SPD Correction (Point 2)
        # Only shift spectrum if the matrix breaches the floor
        min_eig = np.min(np.linalg.eigvalsh(transformed))
        if min_eig < MIN_EIGENVALUE:
            transformed += EPS_REGULARIZATION * np.eye(n_channels)
            
        # Compute matrix logarithm
        matrix_log: FloatMatrix = logm(transformed).real
        
        # Verify finite outputs immediately (Point 9)
        if not np.isfinite(matrix_log).all():
            raise ValueError(
                f"Numerical failure: Matrix logarithm contains NaN or Inf at trial index {idx}."
            )

        # --- 5. Explicit Metric Scaling & Vectorization (Point 7) ---
        # Scale off-diagonals cleanly without touching or correcting the diagonal afterwards
        matrix_log[off_diag_idx] *= sqrt_two
        
        # Pack elements into the tangent array using ordered upper-triangle indexing
        tangent_vectors[idx] = matrix_log[triu_idx]

    return tangent_vectors
