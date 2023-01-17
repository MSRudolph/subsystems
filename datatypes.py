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


# @dataclass()
# class HamiltonianObject:
#     """Data structure for Hessian matrix.
#         Eigenvalues and Eigenvectors are automatically calculated and sorted.
#     Args:
#         params: Parameter vector at which the Hessian matrix was measured
#         hessian_matrix: Hessian matrix evaluated at params as a 2D numpy array.
#     """

#     H: np.ndarray
#     H_s: Optional[np.ndarray] = None
#     H_e: Optional[np.ndarray] = None
#     eigvals_H: np.ndarray = field(init=False)
#     P: np.ndarray = field(init=False)
#     t_dec: float = field(init=False)

#     def __post_init__(self):
#         self.eigvals_H, self.P = np.linalg.eigh(self.H)
#         self.t_dec = 1/max(np.abs(self.eigvals_H))



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

    # def get_psi(self):
    #     if self.A is not None:
    #         return np.tensordot(self.A@self.psi_s, self.psi_e, axes=0).flatten()
    #     elif self.psi is None:
    #         self.psi = np.tensordot(self.psi_s,self.psi_e, axes=0).flatten()
    #         return self.psi
    #     else:
    #         return self.psi


