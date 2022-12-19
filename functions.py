import numpy as np
import scipy
from qutip import Qobj, tensor


# optimization routine
def opt(B: np.ndarray, training_dt: np.ndarray, psi: np.ndarray, eigvals_H: np.ndarray, P: np.ndarray, qS: int, qE: int, thresh=1e-11, maxiter=1e5, print_every: int = 500):
    #     psi=psi0()

    co = cost(B, psi, eigvals_H, P, training_dt, qS, qE)
    print("Starting Guess purity", co)

    i = 0
    alpha = 0
    prev_cost = 1
#     thresh = 1e-11 #cost threshold (stopping criterion)
#     thresh = 2e-7
    while (co > thresh) and (i < maxiter):
        i += 1
        B = update_B(B, training_dt, psi, eigvals_H, P, alpha, qS, qE)
        co = cost(B, psi, eigvals_H, P, training_dt, qS, qE)

        cost_ratio = co/prev_cost
        fac = 1.1
        secondThresh = thresh  # 1.e-9
        if (cost_ratio > (1-1.e-5) and co < secondThresh):
            print("Stuck, but cost is quite low so might as well quit now. Cost=", co)
            return B, co

        # the following adapts alpha given the cost ratio for the previous and this step
        if (alpha < 1.e-10):
            alpha = 0
        if (cost_ratio > 1):  # increase alpha
            if (alpha == 0):
                alpha = 1.e-5
            alpha = alpha*fac
        else:  # decrease alpha
            alpha = alpha/fac

        if (alpha > 500):  # or (cost_ratio>(1-1.e-9) and co>thresh):
            print("This isn't working; let's try starting with a new guess")
            B, s, vh = np.linalg.svd(np.random.rand(2**(qS+qE), 2**(qS+qE))-0.5 + 1j *
                                     np.random.rand(2**(qS+qE), 2**(qS+qE))-0.5)  # start with an inital guess
            alpha = 0
        prev_cost = co

        if (i % print_every == 0):
            print("COST", co, "after", i, "iterations ", "alpha=", alpha)

    print("Total iterations:", i)
    print(training_dt, ":", co)

    return B, co


# for checking if you've found a solution that generalizes beyond the training times
def check(B: np.ndarray, times: np.ndarray, psi: float, eigvals_H, P, qS: int, qE: int):

    def get_purity_global(ket: np.ndarray, system_qubit_index: int):
        dims = [[2**(qS), 2**(qE)], [2**(qS), 2**(qE)]]
        rho = Qobj(np.outer(ket, ket.conj()), dims=dims)
        red_rho = rho.ptrace(system_qubit_index)
        purity = np.trace(red_rho.full()@red_rho.full())
        return purity

    # if times is None:
    #     times = np.logspace(np.log10(t_dec)-1,np.log10(t_dec)+4,50)

    prt = []
    for t in times:
        psi_t = B@eH(t, eigvals_H, P, qS, qE)@psi
        prt.append(get_purity_global(psi_t, 0).real)

    return prt


def prep_training(get_H_function, qS: int, qE: int, random_state=False, seed=None):
    H, H_s, H_e = get_H_function()

    eigvals_H, P = np.linalg.eigh(H)
    t_dec = 1/max(np.abs(eigvals_H))

    if random_state or H_s is None:
        psi = rand_state(2**(qS+qE), seed=seed)
    else:
        psi_S = np.linalg.eigh(H_s)[1][:, 0]
        psi_E = rand_state(2**qE, seed=seed)
        psi = np.ndarray.flatten(np.tensordot(psi_S, psi_E, axes=0))

    B = get_random_unitary(qS, qE, seed=seed)

    return B, H, eigvals_H, P, t_dec, psi


def get_random_unitary(qS: int, qE: int, seed=None):
    if seed is not None:
        np.random.seed(seed)
    B, s, vh = np.linalg.svd(np.random.rand(2**(qS+qE), 2**(qS+qE))-0.5 + 1j *
                             np.random.rand(2**(qS+qE), 2**(qS+qE))-0.5)  # start with an inital guess
    return B


def rand_state(dims, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.rand(seed)
    state = np.random.rand(dims) + 1j*np.random.rand(dims)
    return state/np.linalg.norm(state)


def eH(t: float, eigvals_H: np.ndarray, P: np.ndarray, qS: int, qE: int) -> np.ndarray:

    H_D = np.zeros((2**(qS+qE), 2**(qS+qE))).astype(dtype=np.complex128)
    np.fill_diagonal(H_D, np.exp(-1j*eigvals_H*t))
    Ut = P@H_D@P.conj().T
    return Ut


def cost(B: np.ndarray, psi: np.ndarray, eigvals_H: np.ndarray, P: np.ndarray, dt: np.ndarray, qS: int, qE: int) -> float:
    co = 0
    for t in dt:
        phi = B@eH(t, eigvals_H, P, qS, qE)@psi
        phi = np.reshape(phi, (2**qS, 2**qE))
        rho_pt = phi@phi.T.conj()
        co += 1-np.trace(rho_pt@rho_pt)
    return co/len(dt)


def updateB(A, B, alpha, qS, qE):
    psiSA = A@psiS
    psi = np.ndarray.flatten(np.tensordot(psiE, psiSA, axes=0))
    E = np.zeros((2**(qE+qS), 2**(qS+qE))).astype(dtype=np.complex128)
    for Ut in Utt:
        phi = B@Ut@psi
        phi = np.reshape(phi, [2**qS, 2**qE], order='F')
        rho = phi@phi.conj().T

        phit = rho.conj()@phi.conj()
        chi = Ut@psi
        phit = np.reshape(phit, [1, 2**(qS+qE)], order='F')

        chi = np.reshape(chi, [2**(qS+qE), 1], order='F')
        E = E+chi@phit

    u, x, v = svd((E+alpha*B.conj().T), lapack_driver='gesvd')
    B = v.conj().T@u.conj().T
    return B


def update_B(B: np.ndarray, training_dt: np.ndarray, psi: np.ndarray, eigvals_H: np.ndarray, P: np.ndarray, alpha: float, qS: int, qE: int):
    E = np.zeros((2**(qS+qE), 2**(qS+qE)))
    for t in training_dt:
        phi = B @ eH(t, eigvals_H, P, qS, qE) @ psi
        phi = np.reshape(phi, (2**qS, 2**qE))
        rho = phi@phi.T.conj()

        phit = rho.conj().T@phi.conj()
        chi = eH(t, eigvals_H, P, qS, qE) @ psi
        phit = phit.flatten()
        E = E+np.outer(chi, phit)

    u, x, v = scipy.linalg.svd((E+alpha*B.conj().T), lapack_driver='gesvd')
    B_new = v.conj().T@u.conj().T
    return B_new
