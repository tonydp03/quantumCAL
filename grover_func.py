##############################################
##############################################
### ******** FUNCTIONS FOR GROVER ******** ###
##############################################
##############################################


from cmath import log10
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import random
import math
from q_utilities import *
from scipy.special import gamma
from scipy.integrate import trapz
import sys

################
### *************** CHANGE OF COORDINATES *************** ###
################

### From cartesian to decimal
def cart_to_dec(point, all_X, all_Y):
    # Input1 : point in the form of the (cartesian) dataset: (x, y, z, belonging layer)
    # Input2&3 : all the X and Y coordinates (respectively) in the layer
    # Output: point in the decimal form (x_dec, y_dec, z, belonging layer)
    
    Nx = len(all_X)
    Ny = len(all_Y)
    DimX, DimY = n_qubits(Nx,Ny)
    
    # the last coordinate of point is the layer where the point belongs:
    Zq = int(point[3])
    # spit out the correct result if the layer is inactive:
    X = point[0]
    Y = point[1]
    
    # if the point is inactive
    if (np.isnan(X)) or (np.isnan(Y)):
        Xq = 2**DimX - 1
        Yq = 2**DimY - 1
        
        return np.array([Xq, Yq, point[2], Zq])
    # if it is not inactive    
    else:
        Xq = np.where(X == np.sort(all_X))[0][0]
        Yq = np.where(Y == np.sort(all_Y))[0][0]
        
        return np.array([Xq, Yq, point[2], Zq])

### From decimal to cartesian
def dec_to_cart(point, dataset):
    # Input1 : point in the decimal form of the dataset: (x, y, z, belonging layer)
    # Input2&3&4 : all the X and Y and Z coordinates (respectively)
    # Output: point in the cartesian form (x, y, z, belonging layer)
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    
    Nx = len(all_X)
    Ny = len(all_Y)
    DimX, DimY = n_qubits(Nx,Ny)
    
    #return inactive in case the point is inactive
    if (point[0] == 2**DimX - 1) & (point[1] == 2**DimY - 1):
        return np.array([math.nan, math.nan, point[2], point[3]])
    else:
        return np.array([np.sort(all_X)[int(point[0])], np.sort(all_Y)[int(point[1])], point[2], point[3]])


################
### *************** QUBIT FORM OF A STATE - SINGLE LAYER *************** ###
################

### Determine the number of qubits required for a given dataset
def n_qubits(Nx,Ny):
    # Inputs: number of elements in the x, y and z dimensions
    # Output: number of required qubits for Grover
    
    DimX = np.log2(Nx)
    DimY = np.log2(Ny)
    #if there is only one point in either the X or Y dimension, add a qubit nevertheless:
    only_point = False
    if DimX == 0:
        DimX = 1
        only_point = True
    if DimY == 0:
        DimY = 1
        only_point = True
    #if there are too many points, or either DimX or DimY is equal to 0, add a qubit in X for the inactive layer
    if (DimX == np.round(DimX)) & (DimY == np.round(DimY)) & (only_point == False):
        DimX = DimX + 1
    else:
        DimX = np.ceil(DimX)
    DimY = np.ceil(DimY)
    
    return int(DimX), int(DimY)

### Quantum state for a single point (decimal to qubit)
def dec_to_qubit(point, all_X, all_Y):
    # Inputs 1: point in the decimal form (x, y, z, belonging layer)
    # Input 2&3: all the X and Y coordinates (respectively) in the layer
    # Output: state vector (if form = "qubit") or dec array (if form = "dec")
    
    # Finding dimensions:
    Nx = len(all_X)
    Ny = len(all_Y)
    DimX, DimY = n_qubits(Nx,Ny)
    
    # Defining the X and the Y states into its qubit form
    StX = bin(int(point[0]))[2:].zfill(DimX)[::-1]
    StY = bin(int(point[1]))[2:].zfill(DimY)[::-1]
    
    if (len(StX) == 0) or (len(StY) == 0):
        print("Error in defining the state on a single layer!")
    else:
        #tableaus to be given into the state function
        tabX = [np.mod(int(StX[i])+1,2)*StD + np.mod(int(StX[i]),2)*StU for i in range(len(StX))]
        tabY = [np.mod(int(StY[i])+1,2)*StD + np.mod(int(StY[i]),2)*StU for i in range(len(StY))]
        #states
        StXq = state(tabX)
        StYq = state(tabY)
        #kron of the states
        out = scipy.sparse.kron(StXq, StYq, format = "csr")
        
        return out

### Quantum state for a single point (qubit to decimal)
def qubit_to_dec(point, dataset, form = "dec"):
    # Inputs 1: point in the qubit form [qubit point , z, belonging layer]
    # Input 2: dataset fed into grover
    # Output: point corresponding in decimal form (if form = "qubit") or cartesian form (if form = "cart")
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    
    # Finding dimensions:
    Nx = len(all_X)
    Ny = len(all_Y)
    DimX, DimY = n_qubits(Nx,Ny)
    
    qubit_state = point[0]
    layer = point[2]
    z_coord = point[1]
    
    # Find position of the biggest element in "point"
    # index_max = int(np.where(qubit_state.toarray() == np.max(qubit_state.toarray()))[0][0])
    index_max = int(qubit_state.argmax())
    
    # Recover the qubit form
    dec_form = bin(2**(DimX + DimY) - 1 - index_max)[2:].zfill(DimX+DimY)
    dec_x = dec_form[0:DimX]
    dec_x = int("0b"+dec_x[::-1],2)
    dec_y = dec_form[DimX:DimX+DimY+1]
    dec_y = int("0b"+dec_y[::-1],2)
      
    #return
    if form == "dec":
        return np.array([dec_x, dec_y, z_coord, layer])
    elif form == "cart":
        return dec_to_cart([dec_x, dec_y, z_coord, layer], dataset)


################
### *************** QUBIT FORM OF A STATE - MANY LAYERS *************** ###
################

### Quantum state for multiple points (decimal to qubit)
def full_dec_to_qubit(in_vec, dataset):
    # Input 1: array of points in the decimal form [[dec point , z, belonging layer],...,]
    # Input 2: dataset fed into grover
    # Output: vector state of a trajectory
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    
    ## Finding dimensions:
    #Nx = len(all_X)
    #Ny = len(all_Y)
    #DimX, DimY = n_qubits(Nx,Ny)
    
    # Sort (first) and generate qubit states for each point in in_vec
    sorted_vec = sorted(in_vec, key = lambda el: el[2])
    all_states = [dec_to_qubit(sorted_vec[i], all_X, all_Y) for i in range(len(sorted_vec))]

    out_state = state([all_states[i] for i in range(len(all_states))])
    
    return out_state

### Quantum state for multiple points (qubit to decimal)
def full_qubit_to_dec(in_vec, dataset, form = "dec"):
    # Input 1: full qubit state (vector state of a trajectory)
    # Input 2: dataset fed into grover
    # Output: points corresponding in decimal form (if form = "dec") or cartesian form (if form = "cart")
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    all_Z_index = dataset[3]
    
    # Finding dimensions:
    Nx = len(all_X)
    Ny = len(all_Y)
    DimX, DimY = n_qubits(Nx,Ny)
    DimTot = DimX + DimY
    
    # Find position of the biggest element in "in_vec"
    index_max = int(abs(in_vec).argmax())
        
    # Recover the dec form
    dec_form = bin((2**DimTot)**len(all_Z) - 1 - index_max)[2:].zfill(DimTot * len(all_Z))
    
    # Find the dec form for all layers (first), and then the x and y components
    dec_form_layer = [dec_form[DimTot*i:DimTot*(i+1)] for i in range(len(all_Z))]
    dec_x_tmp = [dec_form_layer[i][0:DimX] for i in range(len(all_Z))]
    dec_x = [int("0b"+dec_x_tmp[i][::-1],2) for i in range(len(all_Z))]
    dec_y_tmp = [dec_form_layer[i][DimX:DimX+DimY+1] for i in range(len(all_Z))]
    dec_y = [int("0b"+dec_y_tmp[i][::-1],2) for i in range(len(all_Z))]
    
    # Plug these results together to find the state in dec form:
    #dec_fin = [[dec_x[i] ,dec_y[i] ,all_Z[i] , i] for i in range(len(all_Z))]
    
    # Output if requested in "dec" form
    if form == "dec":
        return [np.array([dec_x[i] ,dec_y[i] ,all_Z[i] , all_Z_index[i]]) for i in range(len(all_Z))]
    elif form == "cart":
        dec_fin = [[dec_x[i] ,dec_y[i] ,all_Z[i] , all_Z_index[i]] for i in range(len(all_Z))]
        return [dec_to_cart(dec_fin[i], dataset) for i in range(len(all_Z))]
    else:
        print("Which form do you want the output in the function full_qubit_to_dec?")


def pca(data, weights):
    xyz = data.T
    assert xyz.shape[0] == 3, "Input data shape should be 3, check your input shape"
    N = len(weights)
    barycenter = [0.,0.,0.]
    covM = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    covMW = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    sorted_index = [0,0,0]
    sorted_eigenvalue = [0,0,0]
    sorted_eigenvectors = [0,0,0]

    weight_sum = np.sum(weights)
    weight_sum2 = 0

    # Compute the barycenter
    # barycenter = np.average(data, weights, axis=0)
    for i in range(N):

        weight = weights[i] #computing the weight
        barycenter[0] += xyz[0][i] * weight
        barycenter[1] += xyz[1][i] * weight
        barycenter[2] += xyz[2][i] * weight
    # print(barycenter)
    barycenter /= weight_sum 

    #Compute the covariance matrix
    covM = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    for i in range(N):
        weight = weights[i]/weight_sum

        weight_sum2 += weight * weight            
        for x in range(3):
            for y in range(3):
                covM[x][y] += weight * (xyz[x][i] - barycenter[x]) * (xyz[y][i] - barycenter[y])
    
    covMW = covM * 1./(1-weight_sum2) #weighted covariance matrix
    #get eigenvalues and eigenvectors and sort them
    try:
        eigen_values , eigen_vectors = np.linalg.eigh(covMW)
        norm = np.linalg.norm(eigen_values)
        vectors = eigen_values/norm
        sorted_index = np.argsort(vectors)[::-1]
        # sorted_eigenvalue = vectors[sorted_index]
        sorted_eigenvectors = eigen_vectors[:,sorted_index].T
    except:
        print("Not Converged")
        print(N)
        sorted_index = [0,0,0]
        # sorted_eigenvalue = [0,0,0]
        sorted_eigenvectors = [0,0,0]
        barycenter = []

    return barycenter, sorted_eigenvectors 

def line_fit(data, weights):

    # calculate the mean of the points, i.e. the 'center' of the cloud, and the pca vector
    datamean, vv = pca(np.array(data), weights)

    # find the length of the segment such that it encompasses all the considered points.
    max_dist = np.linalg.norm(data[-1] - data[0])
    
    # calculate the line segment
    linepts = vv[0] * np.mgrid[-max_dist:max_dist:2j][:, np.newaxis]

    # shift by the mean to get the line in the right place
    linepts += datamean
    
    # linepts_xy = linepts
    # linepts_etaphi = np.array([xyz_to_etaphiz(linepts_xy[0,0], linepts_xy[0,1], linepts_xy[0,2]),
    #                           xyz_to_etaphiz(linepts_xy[-1,0], linepts_xy[-1,1], linepts_xy[-1,2])])

    return linepts, vv[0]

def dist_from_line(point, linepts):
    
    Av = linepts[0]
    Bv = linepts[1]
    Pv = point

    # find the point Vv = Bv + (v-1) Av on the line that is perpendicular to the chosen point outside
    v = ((Av[0]-Bv[0]) * (Av[0]-Bv[0]+Pv[0]) + (Av[1]-Bv[1]) * (Av[1]-Bv[1]+Pv[1]) + (Av[2]-Bv[2]) * (Av[2]-Bv[2]+Pv[2])) / (Av[0]**2 + Av[1]**2 - Av[0]*Bv[0] - Av[1]*Bv[1] + Av[2]*(Av[2]-Bv[2]))
    Vv = Bv + (v-1)*Av
    
    # return the norm of the vector point - Vv
    return np.linalg.norm(Pv-Vv)

def pval_fit(data, weights):
    linepts, _ = line_fit(data, weights)
    sum = 0
    for point in data:
        sum += dist_from_line(point, linepts)**2
    sum = np.sqrt(sum)
    # print('||||||| Sum: {}'.format(sum))
    dofs = len(data) - 2
    prob = p_val(sum, dofs)
    # print('||||||| PVAL: {}'.format(prob))
    return prob

def chi_squared(x, dofs):
    return (x**(dofs/2-1) * np.exp(-x/2)) / (2**(dofs/2) * scipy.special.gamma(dofs/2)) * np.heaviside(x, 0)

def p_val(xi_val, dofs, check=False):
    if(check):
        N_points = 10**3
        if (dofs == 1):
            x_int = np.logspace(-10, 3, N_points)
        else: 
            x_int = np.arange(dofs/10**10, 10**2*dofs, 1/N_points)
        norm = trapz(chi_squared(x_int, dofs), x_int)
        print('||||| NORM X^2', norm)
        if(np.abs(1-norm)> 0.01):
            sys.exit()

    N_points = 10**3
    if (dofs == 1):
        x_int = np.logspace(-10, np.log10(xi_val), N_points)
    else:
        x_int = np.arange(dofs/10**10, xi_val, xi_val/N_points)
    
    F_approx = trapz(chi_squared(x_int, dofs), x_int)
    
    return F_approx




################
### TESTS ###
################
if __name__=="__main__":

    ################
    ### CHECK THAT THE COORDINATE TRANSFORM WORKS FOR SINGLE POINT ###
    ################

    Nx = random.randint(1,50)
    Ny = random.randint(1,50)
    Nz = random.randint(1,50)
    Ntry = 100

    all_X = np.sort(np.array([random.random() for i in range(Nx)]))
    all_Y = np.sort(np.array([random.random() for i in range(Ny)]))
    all_Z = np.sort(np.array([random.random() for i in range(Nz)]))

    for i in range(Ntry):

        point = np.array([random.choice(all_X),random.choice(all_Y),random.choice(all_Z)])
        if random.random() < 0.5/Ntry:
            point[0] = math.nan
            point[1] = math.nan
        point = np.append(point,np.where(point[2] == all_Z))
        point_dec = cart_to_dec(point, all_X, all_Y)
        
        qubit_state = dec_to_qubit(point_dec, all_X, all_Y)
        point_rec_cart = qubit_to_dec([qubit_state,point[2],point[3]], [all_X,all_Y,all_Z], form = "cart")
        point_rec_qubit = qubit_to_dec([qubit_state,point[2],point[3]], [all_X,all_Y,all_Z], form = "dec")
        
        if not np.isnan(point[0]):
            if (max(np.abs(point_rec_cart-point))>10**-10) or (max(np.abs(point_rec_qubit-point_dec))>10**-10):
                print("error when there is no nan")
        else:
            if (max(np.abs(point_rec_cart[2:]-point[2:])) >10**-10) or (max(np.abs(point_rec_qubit-point_dec))>10**-10):
                print("error when there is nan")


    ################
    ### CHECK THAT THE COORDINATE TRANSFORM WORKS FOR MULTIPLE POINTS ###
    ################

    # Choose number of iterations:
    Ntry = 100

    # Choosing random dimensions
    Nx = random.randint(1,3)
    Ny = random.randint(1,7)
    Nz = random.randint(1,5)

    # Calculating the dimensions for the vector
    DimX, DimY = n_qubits(Nx,Ny)

    if Nx == 1 and Ny == 1:
        Ny = 2

    # Generating random data:
    all_X = np.sort(np.array([random.random() for i in range(Nx)]))
    all_Y = np.sort(np.array([random.random() for i in range(Ny)]))
    all_Z = np.sort(np.array([random.random() for i in range(Nz)]))

    dataset = [all_X, all_Y, all_Z]


    for j in range(Ntry):
        point_list_cart = []
        for i in range(Nz):
            point_list_cart.append([random.choice(all_X),random.choice(all_Y),all_Z[i],i])
            if random.random() < 20*0.5 / (Nz*Ntry):
                point_list_cart[i][0] = math.nan
                point_list_cart[i][1] = math.nan
        
        # Find the decimal form of the point:
        point_list_dec = [cart_to_dec(point_list_cart[i], dataset[0], dataset[1]) for i in range(Nz)]    
        # Find the corresponding qubit vector:
        point_qubit = full_dec_to_qubit(point_list_dec, dataset)
        # Reconstruct the decimal form from the qubit state:
        rec_point_dec = full_qubit_to_dec(point_qubit, dataset, form = "dec")
        # Reconstruct the cartesian form from the qubit state:
        rec_point_cart = full_qubit_to_dec(point_qubit, dataset, form = "cart")
        
        # Check that we find the correct state:
        dev_dec = np.max(np.abs(np.array(rec_point_dec)-np.array(point_list_dec)))
        dev_cart = 0
        num_el = 0
        
        for i in range(Nz):
            if ((not np.isnan(point_list_cart[i][0])) and (not np.isnan(rec_point_cart[i][0]))):
                dev_cart = dev_cart + np.max(np.abs(rec_point_cart[i]-point_list_cart[i]))
                num_el = num_el + 1
                
            elif (np.isnan(point_list_cart[i][0]) and not np.isnan(rec_point_cart[i][0])) or (np.isnan(rec_point_cart[i][0]) and not np.isnan(point_list_cart[i][0])):
                dev_cart = dev_cart + 1000
                num_el = num_el + 1
            elif ((np.isnan(point_list_cart[i][0])) and (np.isnan(rec_point_cart[i][0]))):
                num_el = num_el + 1
            
        if num_el != Nz:
            print("Missing some elements in the cartesian comparison!!!")
        if dev_dec >10**-10 or dev_cart >10**-10:
            print("Error with the coordinates conversions!!!")
            print("dev_dec = " + str(dev_dec))
            print("dev_cart = " + str(dev_cart))
    
    print("Tests done!")