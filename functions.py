import numpy as np
import scipy
from qutip import Qobj
from datatypes import PsiObject, HamiltonianObject
from typing import Optional


# optimization routine
def opt(B: np.ndarray, training_dt: np.ndarray, psi_object: PsiObject, H_object: HamiltonianObject, qS: int, qE: int, thresh=1e-11, maxiter=1e5, print_every: int = 500):

    eigvals_H = H_object.eigvals_H
    P = H_object.P
    co = cost(B, psi_object.get_psi(), eigvals_H, P, training_dt, qS, qE)
    print("Starting Guess purity", co)


    train_state = False
    if psi_object.A is not None:
        train_state = True

    i = 0
    beta = 0
    alpha = 0

    prev_cost = 1
#     thresh = 1e-11 #cost threshold (stopping criterion)
#     thresh = 2e-7
    while (co > thresh) and (i < maxiter):
        i += 1
        
        if train_state:
            psi_object.A = update_A(B, training_dt, psi_object, H_object.eigvals_H, H_object.P, beta, qS, qE)
        B = update_B(B, training_dt, psi_object.get_psi(), H_object.eigvals_H, H_object.P, beta, qS, qE)
        
        co = cost(B, psi_object.get_psi(), eigvals_H, P, training_dt, qS, qE)

        cost_ratio = co/prev_cost
        fac = 1.1
        secondThresh = thresh  # 1.e-9
        if (cost_ratio > (1-1.e-5) and co < secondThresh):
            print("Stuck, but cost is quite low so might as well quit now. Cost=", co)
            return B, co

        # the following adapts alpha given the cost ratio for the previous and this step
        if (alpha < 1.e-10):
            alpha = 0
        if (beta < 1.e-10):
            beta = 0
        if (cost_ratio > 1):  # increase alpha
            if (alpha == 0) and train_state:
                alpha = 1.e-5
            if (beta == 0):
                beta = 1.e-5
            alpha = alpha*fac
            beta = beta*fac
        else:  # decrease alpha
            alpha = alpha/fac
            beta = beta/fac

        if (alpha > 500):  # or (cost_ratio>(1-1.e-9) and co>thresh):
            print("This isn't working; let's try starting with a new guess")
            psi_object.A = get_random_unitary(qS=qS, qE=0) # start with an inital guess
            beta = 0
        
        if (beta > 500):  # or (cost_ratio>(1-1.e-9) and co>thresh):
            print("This isn't working; let's try starting with a new guess")
            B = get_random_unitary(qS=qS, qE=qE) # start with an inital guess
            beta = 0

        prev_cost = co

        if (i % print_every == 0):
            print("COST", co, "after", i, "iterations ", "alpha=", alpha, "beta=", beta)

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


def prep_training(get_H_function, qS: int, qE: int, random_state: bool=False, train_state: bool = False, seed: Optional[int] = None):
    H_object = get_H_function()

    if train_state:
        A = get_random_unitary(qS=qS, qE=0)
    else:
        A = None

    if random_state or H_object.H_s is None:
        psi_s = rand_state(2**qS, seed=seed)

    else:
        psi_s = np.linalg.eigh(H_object.H_s)[1][:, 0]
    
    psi_e = rand_state(2**qE, seed=seed)
    psi = np.ndarray.flatten(np.tensordot(psi_s, psi_e, axes=0))
    psi_object = PsiObject(psi=psi, psi_s=psi_s, psi_e=psi_e, A=A)

    B = get_random_unitary(qS, qE, seed=seed)

    return B, H_object, psi_object


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


def update_B(B: np.ndarray, training_dt: np.ndarray, psi: np.ndarray, eigvals_H: np.ndarray, P: np.ndarray, beta: float, qS: int, qE: int):
    E = np.zeros((2**(qS+qE), 2**(qS+qE)))
    for t in training_dt:
        Ut = eH(t, eigvals_H, P, qS, qE)
        chi = Ut @ psi
        phi = B @ chi
        phi = np.reshape(phi, (2**qS, 2**qE))
        rho = phi@phi.T.conj()

        phit = rho.conj().T@phi.conj()
        
        phit = phit.flatten()
        E = E+np.outer(chi, phit)

    u, x, v = scipy.linalg.svd((E+beta*B.conj().T), lapack_driver='gesvd')
    B_new = v.conj().T@u.conj().T
    return B_new


def update_A(B: np.ndarray, training_dt: np.ndarray, psi_object: PsiObject, eigvals_H: np.ndarray, P: np.ndarray, alpha: float, qS: int, qE: int):
    psi_s = psi_object.psi_s
    psi_e = psi_object.psi_e
    A = psi_object.A
    psiSA = A@psi_s
    
    psi = np.ndarray.flatten(np.tensordot(psiSA, psi_e, axes=0))

    E = np.zeros((2**qS, 2**qS)).astype(dtype=np.complex128)

    for t in training_dt:
        Ut = eH(t, eigvals_H, P, qS, qE)
        phi = B@Ut@psi
        phi = np.reshape(phi, [2**qS,2**qE]) #, order='F')
        rho = phi@phi.conj().T

        phit = rho.conj()@phi.conj()
        phit = phit.flatten()  # np.reshape(phit,[1,2**(qS+qE)]) # , order='F')
        phit = phit@B@Ut
        phit = np.reshape(phit,[2**qS, 2**qE]) # , order = 'F')
        phit = phit@psi_e
        E = E + np.outer(phit, psi_s)
    u,x,v = scipy.linalg.svd((E+alpha*A.conj().T),lapack_driver='gesvd')
    A = v.conj().T@u.conj().T
    return A
