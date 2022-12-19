import numpy as np
import qutip
from itertools import product, permutations

PAULIS = {
    'I': qutip.identity(2).full(),
    'X': qutip.sigmax().full(),
    'Y': qutip.sigmay().full(),
    'Z': qutip.sigmaz().full()
}


def get_random_H(n_qubits: int, seed=None):
    if seed is not None:
        np.random.rand(seed)
    mat = np.random.normal(size=(2**n_qubits, 2**n_qubits)) + \
        np.random.normal(size=(2**n_qubits, 2**n_qubits))*1j
    return mat*mat.conj().T, None, None


def get_random_tensor_H(qS: int, qE: int, seed=None):
    if seed is not None:
        np.random.rand(seed)
    H_s = qutip.Qobj(get_random_H(qS, seed=seed)[0])
    H_e = qutip.Qobj(get_random_H(qE, seed=seed)[0])
    Htensor = qutip.tensor(H_s, H_e)
    return Htensor.full(), H_s, H_e


def get_interpolated_H(t: float, qS: int, qE: int, seed=None):
    Htensor, H_s, H_e = get_random_tensor_H(qS, qE, seed)

    Hrandom = get_random_H(qS, qE, seed)

    return (Htensor*t + Hrandom*(1-t)), H_s, H_e


def get_central_spin_H(qS: int, qE: int):
    # central spin H

    def tensor_product_list(list_Mats):
        # tensor product all the elements of the list
        temp = np.kron(list_Mats[0], list_Mats[1])
        for i in range(1, len(list_Mats)-1):
            temp = np.kron(temp, list_Mats[i+1])
        return temp
    p_list = ['X']
    [p_list.append('I') for i in range(qE-1)]

    int_terms = set(permutations(p_list))

    H_int = np.zeros((2**(qS+qE), 2**(qS+qE))).astype(dtype=np.complex128)
    for i, term in enumerate(int_terms):
        this_term_mats = [PAULIS[key] for key in term]
        product_of_this_term = tensor_product_list(this_term_mats)
        if (i == 1):
            H_term = np.kron(product_of_this_term, PAULIS['X'])
        elif (i == 2):
            H_term = np.kron(product_of_this_term, PAULIS['Z'])
        else:
            H_term = np.kron(product_of_this_term, PAULIS['Z'])

        H_int += np.random.rand()*H_term
    p_list = ['Y']
    [p_list.append('I') for i in range(qE)]
    self_terms = permutations(p_list)

    H_self = np.zeros((2**(qS+qE), 2**(qS+qE))).astype(dtype=np.complex128)
    for terms in set(self_terms):
        # print(terms)
        mat_list = []
        for this_qubit in terms:
            mat_list.append(PAULIS[this_qubit])
        H_self += tensor_product_list(mat_list)

    return H_int + H_self, None, None
