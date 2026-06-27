"""
Core Riemannian Submanifold Processing and Trace Projection Architecture.
"""

from __future__ import annotations
from typing import Final, TypeAlias
import numpy as np
from scipy.linalg import inv, logm, sqrtm

# Static Code Verification Type Aliases
ArrayND: TypeAlias = np.ndarray
FloatMatrix: TypeAlias = np.ndarray

def compute_riemannian_tangent_space(
    covariances: ArrayND, reference_matrix: FloatMatrix
) -> ArrayND:
    """Projects spatial covariance matrices cleanly onto the localized tangent space."""
    n_trials, n_channels, _ = covariances.shape
    vector_dimension: int = n_channels * (n_channels + 1) // 2
    tangent_vectors: ArrayND = np.zeros((n_trials, vector_dimension))

    # Compute symmetric congruence anchor points
    p_sqrt: FloatMatrix = sqrtm(reference_matrix).real
    p_inv_sqrt: FloatMatrix = inv(p_sqrt).real

    triu_idx: tuple[ArrayND, ArrayND] = np.triu_indices(n_channels)
    sqrt_two: Final[float] = np.sqrt(2.0)

    for idx in range(n_trials):
        # Enforce strict structural symmetry and an eigenvalue floor
        transformed: FloatMatrix = p_inv_sqrt @ covariances[idx] @ p_inv_sqrt
        transformed = (transformed + transformed.T) / 2.0 + 1e-6 * np.eye(n_channels)
        matrix_log: FloatMatrix = logm(transformed).real

        # Scale off-diagonals to preserve canonical geodesic metric distances
        matrix_log[triu_idx] *= sqrt_two
        np.fill_diagonal(matrix_log, np.diagonal(matrix_log) / sqrt_two)

        tangent_vectors[idx] = matrix_log[triu_idx]

    return tangent_vectors
