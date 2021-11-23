##################################################################
##################################################################
### ******** QUANTUM STATES, OPERATORS AND PROJECTORS ******** ###
##################################################################
##################################################################


import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg


################
### *************** STATES *************** ###
################

# 0 and 1
StU = scipy.sparse.csr_matrix(np.array([1,0],dtype=complex)).transpose()
StD = scipy.sparse.csr_matrix(np.array([0,1],dtype=complex)).transpose()

# + and -
StP = scipy.sparse.csr_matrix(np.array(1/np.sqrt(2)*np.array([1,1],dtype=complex))).transpose()
StM = scipy.sparse.csr_matrix(np.array(1/np.sqrt(2)*np.array([1,-1],dtype=complex))).transpose()

### Multi-qUbit states
def state(state_list):
    # Input: list of states of the form above (can include coefficients)
    # Output: resulting state
    tmp = state_list[0]
    for ii in range(len(state_list)-1):
        tmp = scipy.sparse.kron(tmp, state_list[ii+1], format = "csr")
    
    return tmp

###  Density matrix from a state 
def dmat(input_state):
    # Input: state
    # Output: density matrix
    tmp = scipy.sparse.kron(input_state.conjugate(), input_state, format = "csr")
    Num = np.sqrt(tmp.get_shape()[0])
    Num = int(Num)
        
    return scipy.sparse.csr_matrix.reshape(tmp, (Num,Num))


################
### *************** OPERATORS *************** ###
################

### Paulis and raising/lowering
I_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,1]],dtype=complex))
X_mat = scipy.sparse.csr_matrix(np.array([[0,1],[1,0]],dtype=complex))
Y_mat = scipy.sparse.csr_matrix(np.array([[0,-1j],[1j,0]],dtype=complex))
Z_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,-1]],dtype=complex))
P_mat = (X_mat + 1j*Y_mat) / 2
M_mat = (X_mat - 1j*Y_mat) / 2

### Clifford
H_mat = scipy.sparse.csr_matrix(1/np.sqrt(2)*np.array([[1,1],[1,-1]],dtype=complex))
S_mat = scipy.sparse.csr_matrix(np.array([[1,0],[0,1j]],dtype=complex))

### Tensor product between different operators
def tensor(mm):
    # Input: list of matrices
    # Output: tensor product of matrices
    if len(mm) == 0:
        return matrix([])
    elif len(mm) == 1:
        return mm[0]
    else:
        return scipy.sparse.kron(mm[0],tensor(mm[1:]),format="csr")

### First method to write down a Pauli in high dimensions
def pauli_gen(which_pauli , index , dim):
    # Input1: letter identifying the Pauli: "X", "Y", "Z", "P", "M"
    # Input2: index of the qubit to which the Pauli is applied (starting from 0)
    # Input3: total number of qubits
    # Output: corresponding matrix
    
    tmp = [I_mat for i in range(dim)]
    if which_pauli == "X":
        tmp[index] = X_mat
    elif which_pauli == "Y":
        tmp[index] = Y_mat
    elif which_pauli == "Z":
        tmp[index] = Z_mat
    elif which_pauli == "P":
        tmp[index] = P_mat
    elif which_pauli == "M":
        tmp[index] = M_mat
    else:
        if which_pauli != "I":
            raise Exception("Problem with pauli_gen")
        
    return tensor(tmp)

### Second method to write down a Pauli in high dimensions
def pauli_ltm(pauli_list):
    # Input: Pauli list in the form ["I","X","Y","Z","P","M",...]
    # Output: corresponding matrix
    
    dim = len(pauli_list)
    tmp = np.array([])
    
    for i in range(dim):
        if pauli_list[i] == "I":
            tmp = np.append(tmp,I_mat)
        elif pauli_list[i] == "X":
            tmp = np.append(tmp,X_mat)
        elif pauli_list[i] == "Y":
            tmp = np.append(tmp,Y_mat)
        elif pauli_list[i] == "Z":
            tmp = np.append(tmp,Z_mat)
        elif pauli_list[i] == "P":
            tmp = np.append(tmp,P_mat)
        elif pauli_list[i] == "M":
            tmp = np.append(tmp,M_mat)
        else:
            raise Exception("Problem with pauli_ltm")
        
    return tensor(tmp)


#Apply  (list of) operators to a state:
def op_state(operator_list,input_state):
    # Input1: list of matrices
    # Input1: state
    # Output: resulting state
    if len(operator_list) == 0:
        return input_state
    elif len(operator_list) == 1:
        return operator_list[0].dot(input_state)
    else:
        return operator_list[-1].dot(op_state(operator_list[:-1] , input_state))


################
### *************** OPERATORS *************** ###
################

def st_proj(out_st,in_st):
    # Input: states for the projector
    # Output: resulting operator
    proj_in = in_st.conjugate().T
    proj_out = out_st

    return scipy.sparse.kron(proj_out,proj_in,format="csr")



################
### TESTS ###
################
if __name__=="__main__":

    ################
    ### CHECK THAT STATES AND OPERATORS ARE WELL DEFINED ###
    ################

    testL = state([StU,StD,StP,StM]).conjugate().transpose()

    testR1 = state([StD,StD,StP,StM])
    testR2 = state([StU,StU,StP,StM])
    testR3 = state([StU,StD,StM,StM])
    testR4 = state([StU,StD,StP,StP])
    testR5 = state([StD,StU,StM,StP])

    Op1 = [pauli_ltm(["P","I","I","I"])]
    Op2 = [pauli_ltm(["I","M","I","I"])]
    Op3 = [pauli_ltm(["I","I","Z","I"])]
    Op4 = [pauli_ltm(["I","I","I","Z"])]
    Op5 = [pauli_ltm(["P","M","Z","Z"])]

    print(testL.dot(op_state(Op1,testR1)).toarray())
    print(testL.dot(op_state(Op2,testR2)).toarray())
    print(testL.dot(op_state(Op3,testR3)).toarray())
    print(testL.dot(op_state(Op4,testR4)).toarray())
    print(testL.dot(op_state(Op5,testR5)).toarray())

    print(pauli_ltm(["X","I","Y","Z"]))