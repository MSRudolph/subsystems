import numpy as np
import scipy
import qutip
from qutip import Qobj
from datatypes import PsiObject#, HamiltonianObject
from hamiltonians import get_decoherence_time
from typing import Optional
from functools import partial
import tensornetwork as tn
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


# optimization routine
def opt(B: np.ndarray, training_dt: np.ndarray, psi_object: PsiObject, H: np.ndarray, qS: int, qE: int, train_state: bool = False, thresh=1e-11, maxiter=1e5, print_every: int = 500):

    # eigvals_H = H_object.eigvals_H
    # P = H_object.P
    co = cost(psi_object, B, H, training_dt, qS, qE)
    print("Starting Guess purity", co)

    i = 0
    beta = len(training_dt) #0
    alpha = len(training_dt)  #0

    prev_cost = 1
#     thresh = 1e-11 #cost threshold (stopping criterion)
#     thresh = 2e-7
    while (co > thresh) and (i < maxiter):
        i += 1
        
        if train_state:
            psi_object.A = update_A(psi_object, B, H, training_dt, beta, qS, qE)
        B = update_B(psi_object, B, H, training_dt, beta, qS, qE)
        
        co = cost(psi_object, B, H, training_dt, qS, qE)

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
            alpha = alpha*fac
            beta = beta*fac
            if (alpha == 0) and train_state:
                alpha = 1.e-5
            if (beta == 0):
                beta = 1.e-5
        else:  # decrease alpha
            alpha = alpha/fac
            beta = beta/fac

        if (alpha > 500):  # or (cost_ratio>(1-1.e-9) and co>thresh):
            print("This isn't working; let's try starting with a new guess")
            psi_object.A = get_random_unitary(int(np.log2(np.shape(psi_object.A)[0]))) # start with an inital guess
            alpha = 0
        
        if (beta > 500):  # or (cost_ratio>(1-1.e-9) and co>thresh):
            print("This isn't working; let's try starting with a new guess")
            B = get_random_unitary(qS+qE) # start with an inital guess
            beta = 0

        prev_cost = co

        if (i % print_every == 0) or (i<=10):
            print("COST", co, "after", i, "iterations ", "alpha=", alpha, "beta=", beta)

    print("Total iterations:", i)
    print(training_dt, ":", co)

    return B, co


# for checking if you've found a solution that generalizes beyond the training times
def check(psi_object: PsiObject, B: np.ndarray, H:np.ndarray, times: np.ndarray, qS: int, qE: int):
    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return [float(jax_cost(A, B, H, psi_object.psi, [t], qS, qE, act_entire_state)) for t in times]

# def jax_check(A, B, H, psiS, psiE, dt):
#     return [tensor_cost(A, B, H, psiS, psiE, [t]) for t in dt]


def prep_training(get_H_function, qS: int, qE: int, random_state: bool=True, A_acts_globally: bool = True, A_is_identity: bool=True, seed: Optional[int] = None):
    np.random.seed(seed)
    H, H_s, H_e = get_H_function()
    t_dec = get_decoherence_time(H)
    H /= t_dec*1
    if H_s is not None:
        H_s /= t_dec*1
    if H_e is not None:
        H_e /= t_dec*1

    np.random.seed(seed)
    if A_is_identity:
        if A_acts_globally:
            size = (2**(qS+qE), 2**(qS+qE))
        else:
            size = (2**qS, 2**qS)
        A = np.eye(*size)
    else:
        if A_acts_globally:
            A = get_random_unitary(qS+qE, seed=seed)
        else:
            A = get_random_unitary(qS, seed=seed)

    if random_state:
        psi_s = None
        psi_e = None
        psi = rand_state(2**(qS+qE), seed=seed)
        
    else:
        psi_s = rand_state(2**qS, seed=seed)
        psi_e = rand_state(2**qE, seed=seed)
        psi = np.ndarray.flatten(np.tensordot(psi_e, psi_s, axes=0))
    
    psi_object = PsiObject(psi=np.reshape(psi, [2**qE, 2**qS]), psi_s=psi_s, psi_e=psi_e, A=A)

    B = get_random_unitary(qS+qE, seed=seed)

    return B, H, psi_object


def get_random_unitary(n: int, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # B, s, vh = np.linalg.svd(np.random.normal(size=(2**n, 2**n)) + 1j * np.random.normal(size=(2**n, 2**n)))
    return qutip.rand_unitary_haar(2**n, seed=seed).data.todense()


def rand_state(dims, seed=None) -> np.ndarray:
    if seed is not None:
        np.random.rand(seed)
    state = np.random.normal(size=dims) + 1j*np.random.normal(size=dims)
    return np.array(qutip.rand_unitary_haar(dims, seed=seed).data.todense()[0]) # state/np.linalg.norm(state)


def eH(t: float, H: np.ndarray) -> np.ndarray:
    return jax.scipy.linalg.expm(-1j*H*t)


def cost(psi_object:PsiObject, B: np.ndarray, H: np.ndarray, dt: np.ndarray, qS: int, qE: int) -> float:
    A = psi_object.A
    # psi_s = psi_object.psi_s
    # psi_e = psi_object.psi_e
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return np.real(jax_cost(A, B, H, psi_object.psi, dt, qS, qE, act_entire_state))


@partial(jax.jit, static_argnames=["qS", "qE", "act_entire_state"])
def jax_cost(A, B, H, psi, dt, qS:int, qE:int, act_entire_state: bool):
    
    if act_entire_state:
        Apsi = act_A_on_psi_global(jax.numpy.reshape(A, [2**qE, 2**qS, 2**qE, 2**qS]), psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax") 
    else:
        Apsi = act_A_on_psi_system(A, psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax") 

    tB = tn.Node(np.reshape(B, [2**qE,2**qS, 2**qE,2**qS]), axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")

    co = 0.0
    for t in dt:
        Ut = jax.scipy.linalg.expm(-1j*H*t)
        tUt = tn.Node(np.reshape(Ut, [2**qE,2**qS, 2**qE,2**qS]), axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")

        tApsi["S"] ^ tUt["Sin"]
        tApsi["E"] ^ tUt["Ein"]
        tchi = tn.contract_between(tApsi, tUt, output_edge_order=[tUt["Eout"], tUt["Sout"]], axis_names=["E", "S"])

        tchi["S"] ^ tB["Sin"]
        tchi["E"] ^ tB["Ein"]
        tphi1 = tn.contract_between(tchi, tB, output_edge_order=[tB["Eout"], tB["Sout"]], axis_names=["E", "S"])

        tphi2 = tphi1.copy()
        tphi_conj1 = tphi1.copy(conjugate=True)
        tphi_conj2 = tphi_conj1.copy()

        tphi1["E"] ^ tphi_conj1["E"]
        tphi2["E"] ^ tphi_conj2["E"]
        tphi1["S"] ^ tphi_conj2["S"]
        tphi2["S"] ^ tphi_conj1["S"]

        node = tn.contractors.auto([tphi1, tphi2, tphi_conj1, tphi_conj2])
        co += 1 - jax.numpy.real(node.tensor)
        
    return co /len(dt)


@jax.jit
def act_A_on_psi_global(A: np.ndarray, psi: np.ndarray):
    tpsi = tn.Node(psi, axis_names=["E", "S"], backend="jax")
    tA = tn.Node(A, axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")
    tA["Sin"] ^ tpsi["S"]
    tA["Ein"] ^ tpsi["E"]
    tpsi = tn.contractors.auto([tA, tpsi], output_edge_order=[tA["Eout"], tA["Sout"]])
    return tpsi.tensor


@jax.jit
def act_A_on_psi_system(A: np.ndarray, psi: np.ndarray):
    tpsi = tn.Node(psi, axis_names=["E", "S"], backend="jax")
    tA = tn.Node(A, axis_names=["Sout", "Sin"], backend="jax")
    tA["Sin"] ^ tpsi["S"]
    tpsi = tn.contractors.auto([tA, tpsi], output_edge_order=[tpsi["E"], tA["Sout"]])
    return tpsi.tensor


def update_B(psi_object:PsiObject, B: np.ndarray, H: np.ndarray, training_dt: np.ndarray, beta: float, qS: int, qE: int):
    A = psi_object.A
    # psi_s = psi_object.psi_s
    # psi_e = psi_object.psi_e
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return jax_updateB(A, B, H, psi_object.psi, training_dt, beta, qS, qE, act_entire_state)


@partial(jax.jit, static_argnames=["qS", "qE", "act_entire_state"])
def jax_updateB(A, B, H, psi, dt, beta, qS, qE, act_entire_state:bool):
    # tpsiS = tn.Node(psiS, axis_names=["S"], backend="jax")
    # tpsiE = tn.Node(psiE, axis_names=["E"], backend="jax")
    
    if act_entire_state:
        Apsi = act_A_on_psi_global(jax.numpy.reshape(A, [2**qE, 2**qS, 2**qE, 2**qS]), psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax") 
    else:
        Apsi = act_A_on_psi_system(A, psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax") 
    
    tB = tn.Node(np.reshape(B, [2**qE,2**qS, 2**qE,2**qS]), axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")

    E = jax.numpy.zeros((2**(qE+qS), 2**(qS+qE)), dtype=jax.numpy.complex128)
    for t in dt:
        Ut = jax.scipy.linalg.expm(-1j*H*t)
        tUt = tn.Node(np.reshape(Ut, [2**qE,2**qS, 2**qE,2**qS]), axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")

        tApsi["S"] ^ tUt["Sin"]
        tApsi["E"] ^ tUt["Ein"]
        tchi = tn.contract_between(tApsi, tUt, output_edge_order=[tUt["Eout"], tUt["Sout"]], axis_names=["E", "S"])

        tchi["S"] ^ tB["Sin"]
        tchi["E"] ^ tB["Ein"]
        tphi1 = tn.contract_between(tchi, tB, output_edge_order=[tB["Eout"], tB["Sout"]], axis_names=["E", "S"])

        tphi_conj1 = tphi1.copy(conjugate=True)
        tphi_conj2 = tphi_conj1.copy()

        tphi1["E"] ^ tphi_conj1["E"]
        tphi1["S"] ^ tphi_conj2["S"]

        E += tn.contractors.auto([tphi1, tchi, tphi_conj1, tphi_conj2], 
                            output_edge_order=[tchi["E"], tchi["S"], tphi_conj2["E"], tphi_conj1["S"]]).tensor.reshape((2**(qE+qS), 2**(qS+qE)))
    

    u,x,v = jax.numpy.linalg.svd((E+beta*B.conj().T))
    B = v.conj().T@u.conj().T
    return B


def update_A(psi_object: PsiObject, B: np.ndarray, H: np.ndarray, training_dt: np.ndarray, alpha: float, qS: int, qE: int):
    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return jax_updateA(A, B, H, psi_object.psi, training_dt, alpha, qS, qE, act_entire_state)

@partial(jax.jit, static_argnames=["qS", "qE", "act_entire_state"])
def jax_updateA(A, B, H, psi, dt, alpha, qS, qE, act_entire_state):
    # tpsiS = tn.Node(psiS, axis_names=["S"], backend="jax")
    # tpsiE = tn.Node(psiE, axis_names=["E"], backend="jax")
    tpsi = tn.Node(psi, axis_names=["E", "S"], backend="jax")

    if act_entire_state:
        Apsi = act_A_on_psi_global(jax.numpy.reshape(A, [2**qE, 2**qS, 2**qE, 2**qS]), psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax") 
        E = jax.numpy.zeros((2**(qE+qS), 2**(qE+qS)), dtype=jax.numpy.complex128)
    else:
        Apsi = act_A_on_psi_system(A, psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax") 
        E = jax.numpy.zeros((2**(qS), 2**(qS)), dtype=jax.numpy.complex128)
        

    tB = tn.Node(np.reshape(B, [2**qE,2**qS, 2**qE,2**qS]), axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")

    for t in dt:
        Ut = jax.scipy.linalg.expm(-1j*H*t)
        tUt = tn.Node(np.reshape(Ut, [2**qE,2**qS, 2**qE,2**qS]), axis_names=["Eout", "Sout", "Ein", "Sin"], backend="jax")
      
        tApsi["S"] ^ tUt["Sin"]
        tApsi["E"] ^ tUt["Ein"]
        tchi = tn.contract_between(tApsi, tUt, output_edge_order=[tUt["Eout"], tUt["Sout"]], axis_names=["E", "S"])

        tchi["S"] ^ tB["Sin"]
        tchi["E"] ^ tB["Ein"]
        tphi1 = tn.contract_between(tchi, tB, output_edge_order=[tB["Eout"], tB["Sout"]], axis_names=["E", "S"])

        tphi_conj1 = tphi1.copy(conjugate=True)
        tphi_conj2 = tphi_conj1.copy()

        tphi1["E"] ^ tphi_conj1["E"]
        tphi1["S"] ^ tphi_conj2["S"]

        tphi_conj2["E"] ^ tB["Eout"]
        tphi_conj1["S"] ^ tB["Sout"]

        tB["Ein"] ^ tUt["Eout"]
        tB["Sin"] ^ tUt["Sout"]

        
        if act_entire_state:
            E += tn.contractors.auto(
                [tphi1, tphi_conj1, tphi_conj2, tB, tUt, tpsi], 
                output_edge_order=[tpsi["E"], tpsi["S"], tUt["Ein"], tUt["Sin"]]
                ).tensor.reshape((2**(qE+qS), 2**(qE+qS)))
        else:
            tpsi["E"] ^ tUt["Ein"]
            E += tn.contractors.auto([tphi1, tphi_conj1, tphi_conj2, tB, tUt, tpsi], 
                                output_edge_order=[tpsi["S"], tUt["Sin"]]).tensor


    u,x,v = jax.numpy.linalg.svd((E+alpha*A.conj().T))
    A = v.conj().T@u.conj().T
    return A
