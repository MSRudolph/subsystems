import numpy as np
import qutip
from datatypes import PsiObject
from hamiltonians import get_decoherence_time
from typing import Optional
from functools import partial
import tensornetwork as tn
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


# optimization routine
def opt(
    B: np.ndarray,
    training_dt: np.ndarray,
    psi_object: PsiObject,
    H: np.ndarray,
    qS: int,
    qE: int,
    train_state: bool = False,
    thresh=1e-11,
    maxiter=1e5,
    print_every: int = 500,
):
    
    """
    The main optimization routine maximizing the purity of the of the system and the evironment.

    Args:
        B (np.ndarray): Initial guess for the unitary B. This is the choice of measurement basis.
        training_dt (np.ndarray): Array of training times.
        psi_object (PsiObject): Object containing the initial state and the state prep unitary A.
        H (np.ndarray): Hamiltonian of the combined system and environment.
        qS (int): Number of qubits in the system.
        qE (int): Number of qubits in the environment.
        train_state (bool, optional): Whether to optimize the state prep unitary A. Defaults to False.
        thresh (float, optional): Convergence threshold for the cost function. Defaults to 1e-11.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1e5.
        print_every (int, optional): Frequency of printing the cost function. Defaults to 500.
    Returns:
        np.ndarray: Optimized unitary B.
        float: Final value of the cost function.
    """

    co = cost(psi_object, B, H, training_dt, qS, qE)
    print("Starting Guess purity", co)

    i = 0

    # initialize lagging factors
    # alpha is for A, beta is for B
    alpha = len(training_dt)  # 0
    beta = len(training_dt)  # 0

    prev_cost = 1
    while (co > thresh) and (i < maxiter):
        i += 1

        if train_state:
            psi_object.A = update_A(psi_object, B, H, training_dt, alpha, qS, qE)
        B = update_B(psi_object, B, H, training_dt, beta, qS, qE)

        co = cost(psi_object, B, H, training_dt, qS, qE)

        cost_ratio = co / prev_cost
        fac = 1.1

        # adapt the lagging factors
        if alpha < 1.0e-10:
            alpha = 0
        if beta < 1.0e-10:
            beta = 0
        if cost_ratio > 1:  # increase the lagging factors
            alpha = alpha * fac
            beta = beta * fac
            if (alpha == 0) and train_state:
                alpha = 1.0e-5
            if beta == 0:
                beta = 1.0e-5
        else:  # decrease the lagging factors
            alpha = alpha / fac
            beta = beta / fac

        prev_cost = co

        if (i % print_every == 0) or (i <= 10):
            print("COST", co, "after", i, "iterations ", "alpha=", alpha, "beta=", beta)

    print("Total iterations:", i)
    print(training_dt, ":", co)

    return B, co


# for checking if you've found a solution that generalizes beyond the training times
def check(
    psi_object: PsiObject,
    B: np.ndarray,
    H: np.ndarray,
    times: np.ndarray,
    qS: int,
    qE: int,
):
    """
    Tests the cost function (the purity) on a set of times outside the training set.
    Args:
        psi_object (PsiObject): Object containing the initial state and the state prep unitary A.
        B (np.ndarray): Measurement unitary.
        H (np.ndarray): Hamiltonian of the combined system and environment.
        times (np.ndarray): Array of times to test the cost function on.
        qS (int): Number of qubits in the system.
        qE (int): Number of qubits in the environment.
    """
    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return [
        float(jax_cost(A, B, H, psi_object.psi, [t], qS, qE, act_entire_state))
        for t in times
    ]


def jax_check(
    psi_object: PsiObject,
    B: np.ndarray,
    H: np.ndarray,
    times: np.ndarray,
    qS: int,
    qE: int,
):
    ### lower-level jax version of check function

    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return [
        jax_cost(A, B, H, psi_object.psi, [t], qS, qE, act_entire_state) for t in times
    ]


def prep_training(
    get_H_function,
    qS: int,
    qE: int,
    random_state: bool = True,
    A_acts_globally: bool = True,
    A_is_identity: bool = True,
    seed: Optional[int] = None,
):
    """
    Prepare the Hamiltonian, initial state and initial guess for B and A.
    Args:
        get_H_function (function): Function that returns the Hamiltonian and optionally H_s and H_e.
        qS (int): Number of qubits in the system.
        qE (int): Number of qubits in the environment.
        random_state (bool, optional): Whether to use a random initial state. Defaults to True.
        A_acts_globally (bool, optional): Whether A acts on the entire system+environment or just the system. Defaults to True.
        A_is_identity (bool, optional): Whether A is the identity matrix. Defaults to True.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
    Returns:
        B (np.ndarray): Initial guess for the unitary B.
        H (np.ndarray): Hamiltonian of the combined system and environment.
        psi_object (PsiObject): Object containing the initial state and the state prep unitary A.
    """
    np.random.seed(seed)
    H, H_s, H_e = get_H_function()
    t_dec = get_decoherence_time(H)
    H /= t_dec * 1
    if H_s is not None:
        H_s /= t_dec * 1
    if H_e is not None:
        H_e /= t_dec * 1

    np.random.seed(seed)
    if A_is_identity:
        if A_acts_globally:
            size = (2 ** (qS + qE), 2 ** (qS + qE))
        else:
            size = (2**qS, 2**qS)
        A = np.eye(*size)
    else:
        if A_acts_globally:
            A = get_random_unitary(qS + qE, seed=seed)
        else:
            A = get_random_unitary(qS, seed=seed)

    if random_state:
        psi_s = None
        psi_e = None
        psi = rand_state(2 ** (qS + qE), seed=seed)

    else:
        psi_s = None
        psi_e = None
        states = [rand_state(2) for _ in range(qS + qE)]
        psi = 1
        for st in states:
            psi = np.tensordot(psi, st, axes=0)
        psi = psi.flatten()

    psi_object = PsiObject(
        psi=np.reshape(psi, [2**qE, 2**qS]), psi_s=psi_s, psi_e=psi_e, A=A
    )

    B = get_random_unitary(qS + qE, seed=seed)
    # B = np.eye(2 ** (qS + qE), 2 ** (qS + qE))

    return B, H, psi_object


def get_random_unitary(n: int, seed=None):
    # helper function to get a random unitary

    # Note: For Haar random unitaries we should use randn() instead of rand()
    # but this has bad optimization properties

    if seed is not None:
        np.random.seed(seed)

    return np.linalg.svd(
        np.random.rand(2**n, 2**n) + 0.1j * np.random.rand(2**n, 2**n)
    )[0]


def rand_state(dims, seed=None) -> np.ndarray:
    # helper function to get a random state

    if seed is not None:
        np.random.rand(seed)
    state = np.random.normal(size=dims) + 1j * np.random.normal(size=dims)
    return np.array(
        qutip.rand_unitary_haar(dims, seed=seed).data.todense()[0]
    )


def eH(t: float, H: np.ndarray) -> np.ndarray:
    # the time evolution operator for time t and Hamiltonian H
    return jax.scipy.linalg.expm(-1j * H * t)


def cost(
    psi_object: PsiObject,
    B: np.ndarray,
    H: np.ndarray,
    dt: np.ndarray,
    qS: int,
    qE: int,
) -> float:
    """
    Compute the cost function (the purity).
    Args:
        psi_object (PsiObject): Object containing the initial state and the state prep unitary A.
        B (np.ndarray): Measurement unitary.
        H (np.ndarray): Hamiltonian of the combined system and environment.
        dt (np.ndarray): Array of times to compute the cost function on.
        qS (int): Number of qubits in the system.
        qE (int): Number of qubits in the environment.
    """
    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return np.real(jax_cost(A, B, H, psi_object.psi, dt, qS, qE, act_entire_state))


@partial(jax.jit, static_argnames=["qS", "qE", "act_entire_state"])
def jax_cost(A, B, H, psi, dt, qS: int, qE: int, act_entire_state: bool):
    # lower-level jax version of cost function
    # loops over times in python, everything else in jax

    if act_entire_state:
        Apsi = act_A_on_psi_global(
            jax.numpy.reshape(A, [2**qE, 2**qS, 2**qE, 2**qS]), psi
        )
    else:
        Apsi = act_A_on_psi_system(A, psi)

    co = 0.0
    for t in dt:
        co += jax_cost_part(Apsi, B, H, t, qS, qE)

    return co / len(dt)


@partial(jax.jit, static_argnames=["qS", "qE"])
def jax_cost_part(Apsi, B, H, t, qS, qE):
    # computes the cost for a single time t

    tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax")

    tB = tn.Node(
        np.reshape(B, [2**qE, 2**qS, 2**qE, 2**qS]),
        axis_names=["Eout", "Sout", "Ein", "Sin"],
        backend="jax",
    )

    Ut = jax.scipy.linalg.expm(-1j * H * t)
    tUt = tn.Node(
        np.reshape(Ut, [2**qE, 2**qS, 2**qE, 2**qS]),
        axis_names=["Eout", "Sout", "Ein", "Sin"],
        backend="jax",
    )

    tApsi["S"] ^ tUt["Sin"]
    tApsi["E"] ^ tUt["Ein"]
    tchi = tn.contract_between(
        tApsi, tUt, output_edge_order=[tUt["Eout"], tUt["Sout"]], axis_names=["E", "S"]
    )

    tchi["S"] ^ tB["Sin"]
    tchi["E"] ^ tB["Ein"]
    tphi1 = tn.contract_between(
        tchi, tB, output_edge_order=[tB["Eout"], tB["Sout"]], axis_names=["E", "S"]
    )

    tphi2 = tphi1.copy()
    tphi_conj1 = tphi1.copy(conjugate=True)
    tphi_conj2 = tphi_conj1.copy()

    tphi1["E"] ^ tphi_conj1["E"]
    tphi2["E"] ^ tphi_conj2["E"]
    tphi1["S"] ^ tphi_conj2["S"]
    tphi2["S"] ^ tphi_conj1["S"]

    node = tn.contractors.auto([tphi1, tphi2, tphi_conj1, tphi_conj2])
    return 1 - jax.numpy.real(node.tensor)



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


def update_B(
    psi_object: PsiObject,
    B: np.ndarray,
    H: np.ndarray,
    training_dt: np.ndarray,
    beta: float,
    qS: int,
    qE: int,
):
    # updates the measurement unitary B

    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return jax_updateB(
        A, B, H, psi_object.psi, training_dt, beta, qS, qE, act_entire_state
    )


@partial(jax.jit, static_argnames=["qS", "qE", "act_entire_state"])
def jax_updateB(A, B, H, psi, dt, beta, qS, qE, act_entire_state: bool):
    # lower-level jax version of updateB

    if act_entire_state:
        Apsi = act_A_on_psi_global(
            jax.numpy.reshape(A, [2**qE, 2**qS, 2**qE, 2**qS]), psi
        )
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax")
    else:
        Apsi = act_A_on_psi_system(A, psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax")

    tB = tn.Node(
        np.reshape(B, [2**qE, 2**qS, 2**qE, 2**qS]),
        axis_names=["Eout", "Sout", "Ein", "Sin"],
        backend="jax",
    )

    E = jax.numpy.zeros((2 ** (qE + qS), 2 ** (qS + qE)), dtype=jax.numpy.complex128)
    for t in dt:
        Ut = jax.scipy.linalg.expm(-1j * H * t)
        tUt = tn.Node(
            np.reshape(Ut, [2**qE, 2**qS, 2**qE, 2**qS]),
            axis_names=["Eout", "Sout", "Ein", "Sin"],
            backend="jax",
        )

        tApsi["S"] ^ tUt["Sin"]
        tApsi["E"] ^ tUt["Ein"]
        tchi = tn.contract_between(
            tApsi,
            tUt,
            output_edge_order=[tUt["Eout"], tUt["Sout"]],
            axis_names=["E", "S"],
        )

        tchi["S"] ^ tB["Sin"]
        tchi["E"] ^ tB["Ein"]
        tphi1 = tn.contract_between(
            tchi, tB, output_edge_order=[tB["Eout"], tB["Sout"]], axis_names=["E", "S"]
        )

        tphi_conj1 = tphi1.copy(conjugate=True)
        tphi_conj2 = tphi_conj1.copy()

        tphi1["E"] ^ tphi_conj1["E"]
        tphi1["S"] ^ tphi_conj2["S"]

        E += tn.contractors.auto(
            [tphi1, tchi, tphi_conj1, tphi_conj2],
            output_edge_order=[tchi["E"], tchi["S"], tphi_conj2["E"], tphi_conj1["S"]],
        ).tensor.reshape((2 ** (qE + qS), 2 ** (qS + qE)))

    u, x, v = jax.numpy.linalg.svd((E + beta * B.conj().T))
    B = v.conj().T @ u.conj().T
    return B


def update_A(
    psi_object: PsiObject,
    B: np.ndarray,
    H: np.ndarray,
    training_dt: np.ndarray,
    alpha: float,
    qS: int,
    qE: int,
):
    # updates the state prep unitary A
    A = psi_object.A
    act_entire_state = not (A.shape == (2**qS, 2**qS))
    return jax_updateA(
        A, B, H, psi_object.psi, training_dt, alpha, qS, qE, act_entire_state
    )


@partial(jax.jit, static_argnames=["qS", "qE", "act_entire_state"])
def jax_updateA(A, B, H, psi, dt, alpha, qS, qE, act_entire_state):
    # lower-level jax version of updateA

    tpsi = tn.Node(psi, axis_names=["E", "S"], backend="jax")

    if act_entire_state:
        Apsi = act_A_on_psi_global(
            jax.numpy.reshape(A, [2**qE, 2**qS, 2**qE, 2**qS]), psi
        )
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax")
        E = jax.numpy.zeros(
            (2 ** (qE + qS), 2 ** (qE + qS)), dtype=jax.numpy.complex128
        )
    else:
        Apsi = act_A_on_psi_system(A, psi)
        tApsi = tn.Node(Apsi, axis_names=["E", "S"], backend="jax")
        E = jax.numpy.zeros((2 ** (qS), 2 ** (qS)), dtype=jax.numpy.complex128)

    tB = tn.Node(
        np.reshape(B, [2**qE, 2**qS, 2**qE, 2**qS]),
        axis_names=["Eout", "Sout", "Ein", "Sin"],
        backend="jax",
    )

    for t in dt:
        Ut = jax.scipy.linalg.expm(-1j * H * t)
        tUt = tn.Node(
            np.reshape(Ut, [2**qE, 2**qS, 2**qE, 2**qS]),
            axis_names=["Eout", "Sout", "Ein", "Sin"],
            backend="jax",
        )

        tApsi["S"] ^ tUt["Sin"]
        tApsi["E"] ^ tUt["Ein"]
        tchi = tn.contract_between(
            tApsi,
            tUt,
            output_edge_order=[tUt["Eout"], tUt["Sout"]],
            axis_names=["E", "S"],
        )

        tchi["S"] ^ tB["Sin"]
        tchi["E"] ^ tB["Ein"]
        tphi1 = tn.contract_between(
            tchi, tB, output_edge_order=[tB["Eout"], tB["Sout"]], axis_names=["E", "S"]
        )

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
                output_edge_order=[tpsi["E"], tpsi["S"], tUt["Ein"], tUt["Sin"]],
            ).tensor.reshape((2 ** (qE + qS), 2 ** (qE + qS)))
        else:
            tpsi["E"] ^ tUt["Ein"]
            E += tn.contractors.auto(
                [tphi1, tphi_conj1, tphi_conj2, tB, tUt, tpsi],
                output_edge_order=[tpsi["S"], tUt["Sin"]],
            ).tensor

    u, x, v = jax.numpy.linalg.svd((E + alpha * A.conj().T))
    A = v.conj().T @ u.conj().T
    return A
