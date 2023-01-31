from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class SubSysResult:
    H_object: np.ndarray
    psi_object: PsiObject
    B: np.ndarray
    co: float



@dataclass(frozen=False)
class PsiObject:
    """Data structure for Hessian matrix.
        Eigenvalues and Eigenvectors are automatically calculated and sorted.
    Args:
        params: Parameter vector at which the Hessian matrix was measured
        hessian_matrix: Hessian matrix evaluated at params as a 2D numpy array.
    """

    psi: np.ndarray
    psi_s: Optional[np.ndarray] = None
    psi_e: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
