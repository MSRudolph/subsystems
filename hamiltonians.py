import numpy as np
import qutip
from itertools import permutations, product
# from datatypes import HamiltonianObject


PAULIS = {
    'I': qutip.identity(2).full(),
    'X': qutip.sigmax().full(),
    'Y': qutip.sigmay().full(),
    'Z': qutip.sigmaz().full()
}


def get_random_H(n_qubits: int, seed=None) -> np.ndarray:
    np.random.seed(seed)
    mat = np.random.normal(size=(2**n_qubits, 2**n_qubits)) + \
        np.random.normal(size=(2**n_qubits, 2**n_qubits))*1j
    return 1/4*mat*mat.conj().T, None, None


def get_random_tensor_H(qS: int, qE: int, seed=None) -> np.ndarray:
    np.random.seed(seed)
    H_s = qutip.Qobj(get_random_H(qS, seed=seed)[0])
    H_e = qutip.Qobj(get_random_H(qE, seed=seed)[0])
    Htensor = qutip.tensor(H_e, H_s)
    return Htensor.full(), H_s, H_e


def get_interpolated_H(t: float, qS: int, qE: int, seed=None) -> np.ndarray:
    Htensor, H_s, H_e = get_random_tensor_H(qS, qE, seed)

    Hrandom, _, _ = get_random_H(qS+qE, seed)

    return (Htensor*t + Hrandom*(1-t)), H_s, H_e


def get_central_spin_H(qS: int, qE: int, randomized: bool = False, int_term_factor=1) -> np.ndarray:
    # central spin H

    p_list = ['X']
    [p_list.append('I') for i in range(qE-1)]

    int_terms = set(permutations(p_list))

    H_int = np.zeros((2**(qS+qE), 2**(qS+qE))).astype(dtype=np.complex128)
    for i, term in enumerate(int_terms):
        this_term_mats = [PAULIS[key] for key in term]
        product_of_this_term = tensor_product_list(this_term_mats)
        # print(i, term)
        if (i == 1):
            H_term = np.kron(product_of_this_term, PAULIS['X'])
        elif (i == 2):
            H_term = np.kron(product_of_this_term, PAULIS['Z'])
        else:
            H_term = np.kron(product_of_this_term, PAULIS['Z'])

        
        coeff = np.random.rand() if randomized else 1. 
        H_int += int_term_factor * coeff * H_term

    # print(int_terms)
    p_list = ['Y']
    [p_list.append('I') for i in range(qE)]
    self_terms = set(permutations(p_list))

    H_self = np.zeros((2**(qS+qE), 2**(qS+qE))).astype(dtype=np.complex128)
    for terms in set(self_terms):
        #print(terms)
        mat_list = []
        for this_qubit in terms:
            mat_list.append(PAULIS[this_qubit])
        H_self += tensor_product_list(mat_list)

    # print(self_terms)
    return H_int + H_self, None, None


def get_decoherence_time(H: np.ndarray)-> float:
    eigvals_H, P = np.linalg.eigh(H)
    return eigvals_H[-1]


def pauli_decomp(H, thresh=1.e-10, printt=False):
    N = int(np.log2(len(H)))
    paulis_comb = product(PAULIS, repeat=N)#product from itertools is awesome. Gives cartesian product for my dictionary

    pauli_coeffs_list=[]#list of all the coefficients
    pauli_terms_list =[]#list of the terms to go with the coeffs, like Z\otimesZ\otimesX etc
    for term in paulis_comb:
        this_term_mats = [PAULIS[key] for key in term]#'term' is each of the terms in the Pali sum like  Z\otimesZ\otimesX. Get the corresonding matrices from the dictionary upstairs
        this_coeff = HS( tensor_product_list(this_term_mats),H )/(2**N)
        pauli_coeffs_list.append(this_coeff)
        term_str = ''
        for els in list(term):
            term_str+=els
        pauli_terms_list.append(term_str )#the variable 'term' is a tuple so hard to store (for what's to come later), so just convert it to a string like 'ZZX' and append to list
        if (printt==True and np.abs(this_coeff)>thresh):
            print(term, this_coeff.real)
    return pauli_coeffs_list, pauli_terms_list

def tensor_product_list(list_Mats):
    # tensor product all the elements of the list
    temp = np.kron(list_Mats[0], list_Mats[1])
    for i in range(1, len(list_Mats)-1):
        temp = np.kron(temp, list_Mats[i+1])
    return temp

def HS(M1, M2):
    return (np.dot(M1.conjugate().transpose(), M2)).trace()