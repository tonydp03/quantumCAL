##############################################
##############################################
### ******** OPERATORS FOR GROVER ******** ###
##############################################
##############################################


import numpy as np
import itertools
import random
import math
from grover_func import *


################
### *************** DISCRIMINATING OPERATOR - THE "BLACK BOX" *************** ###
################

### Finding distance between points
def f_dist(point_1, point_2, dataset, input_form):
    # Input 1 & 2: points in "input_form" encoding. 
    # Input 3: dataset fed into grover
    # Input 4: form used for the input. Can be "qubit", "dec", or "cart" for the qubit, decimal or cartesian form, respectively
    # Output: distance between the two points
    
    # Obtain the points in the cartesian form:
    if input_form == "qubit":
        p_1 = qubit_to_dec(point_1, dataset, form = "cart")
        p_2 = qubit_to_dec(point_2, dataset, form = "cart")
    elif input_form == "dec":
        p_1 = dec_to_cart(point_1, dataset)
        p_2 = dec_to_cart(point_2, dataset)
    elif input_form == "cart":
        p_1 = point_1
        p_2 = point_2
    else:
        print("Error in f_dist. Which input are you using?")
    
    # Set infinite distance if at least one of the two points is absent ("nan"):
    if np.isnan(p_1[0]) or np.isnan(p_1[1]) or np.isnan(p_2[0]) or np.isnan(p_2[1]):
        return float('inf')
    
    # calculate distance in case the points are there:
    z_dist = np.abs(p_1[2] - p_2[2])
    xy_dist = np.sqrt((p_1[0] - p_2[0])**2 + (p_1[1] - p_2[1])**2)
    
    # Return the desired value:
    return xy_dist/z_dist

### Finding distances in tracksters
def f_dist_t(trackster, dataset, input_form, point_other_layers = np.array([]), out_length = False):
    # Input 1: points in a trackster in "input_form". 
    # Input 2: dataset fed into grover
    # Input 3: form used for the input. Can be "qubit", "dec", or "cart" for the qubit, decimal or cartesian form, respectively
    # Input 4: points from other layer - RIGHT NOW UNUSED, WE MIGHT WANT TO INCLUDE IN THE FUTURE
    # Input 5: decide which output to gove. If "False", it return the distances. If "True", it returns the number of holes in the tracksters
    
    # Output: max distances of two points in a trackster. Outputs are four: continuous distance, with a missing layer, starting/finishing at some point and starting/finishing at some point with a missing layer
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    
    # Calculating the dimensions for the system:
    DimX, DimY = n_qubits(len(all_X),len(all_Y))
    DimZ = len(all_Z)
    
    # If input_form == "qubit", translate the trackster into a more manageable form:
    if input_form == "qubit":
        track = full_qubit_to_dec(trackster, dataset, form = "cart")
        in_form = "cart"
    elif input_form == "dec":
        track = [dec_to_cart(trackster[i], dataset) for i in range(DimZ)]
        in_form = "cart"
    elif input_form == "cart":
        track = trackster.copy()
        in_form = input_form
    else:
        print("Error with the input form in f_dist_t")
    
    # If out_length = True, count the number of holes and spit out that number:
    if out_length:
        n_holes = 0
        for point in track:
            if np.isnan(point[0]):
                n_holes = n_holes + 1
        return n_holes
    
    # Initialize the vectors containing the distances of interest, that the function will return
    dist_0 = 0.
    dist_1 = 0.
    dist_2 = 0.
    dist_3 = 0.
    
    # Finding the ditances between all points in the trackster:
    all_dist_0 = []
    for i in range(DimZ - 1):
        all_dist_0.append(f_dist(track[i], track[i + 1], dataset, in_form))
        
    # Define dist_0, which is the max of all_dist
    dist_0 = max(all_dist_0)
    
    # Find all missing layers:
    inactive_points = np.argwhere(np.isnan(track))[:,0]
    inactive_points = list(dict.fromkeys(inactive_points))
    active_points = [i for i in range(DimZ) if i not in inactive_points]
    if len(inactive_points) > 1:
        inactive_points.sort()
        inactive_points.reverse()
    # Define the points in the track that remain after removing all inactive ones
    remaining_points = track.copy()
    if len(inactive_points) > 0:
        for i in inactive_points:
            remaining_points.pop(i)
    # distances between inactive points
    dist_inactive = []
    if len(inactive_points) > 1:
        for i in range(len(inactive_points)-1):
            dist_inactive.append(np.abs(inactive_points[i]-inactive_points[i+1]))
    #distances between active points
    dist_active = []
    if len(active_points) > 1:
        for i in range(len(active_points)-1):
            dist_active.append(np.abs(active_points[i] - active_points[i+1]))
            
    # Define dist_1, which is the max distance when removing missing layers in between
    # First, if there are inactive points, find all cases in which the missing layer is a hole in a good event:
    if (not (0 in inactive_points or DimZ - 1 in inactive_points)) and (not 1 in dist_inactive) and (math.isinf(dist_0)):
        all_dist_1 = []
        for i in range(len(remaining_points) - 1):
            all_dist_1.append(f_dist(remaining_points[i], remaining_points[i + 1], dataset, in_form))
        
        # Define dist_1, which is the max of all_dist_1
        dist_1 = max(all_dist_1)
    else:
        dist_1 = float('inf')
    
    # Define dist_2, which is the max distance when removing missing layers at the beginning or the end of the trackster:   
    if len(dist_active) == 0:
        dist_2 = float('inf')
    else:
        if (np.abs(1 - max(np.abs(dist_active))) < 10**-10) and (len(remaining_points) > 2) and (math.isinf(dist_0)):
            all_dist_2 = []
            for i in range(len(remaining_points) - 1):
                all_dist_2.append(f_dist(remaining_points[i], remaining_points[i + 1], dataset, in_form))
            
            # Define dist_2, which is the max of all_dist_2
            dist_2 = max(all_dist_2)
        else: 
            dist_2 = float('inf')
    
    # Define dist_3, which is the max distance when removing missing layers at the beginning or the end of the trackster, AND we include holes within:
    if len(dist_active) == 0:
        dist_3 = float('inf')
    else:
        if (2 - max(np.abs(dist_active)) > -10**-1) and (len(remaining_points) > 2) and (math.isinf(dist_1)) and (math.isinf(dist_0)) and (math.isinf(dist_2)):
            all_dist_3 = []
            for i in range(len(remaining_points) - 1):
                all_dist_3.append(f_dist(remaining_points[i], remaining_points[i + 1], dataset, in_form))
                
            # Define dist_2, which is the max of all_dist_2
            dist_3 = max(all_dist_3)
        else: 
            dist_3 = float('inf')
    
    
    
    return dist_0, dist_1, dist_2, dist_3

### Given a set of points, collect them layer by layer
def points_layer_collection(all_points, dataset, add_inactive = True, input_form = "cart", output_form = "dec"):
    # Input 1: all points to be considered in the search.
    # Input 2: dataset fed into grover
    # Input 3: option to add the inactive points
    # Input 4: input form - currently is ONLY "cart"
    # Input 5: output form - currently is ONLY "dec"
    # Output: [[all points in layer 0], [all points in layer 1],...,[all points in layer -1]]
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    all_Z_indices = dataset[3]
    
    #Dimensions of the system:
    # DimX, DimY = n_qubits(len(all_X),len(all_Y))
    # DimZ = len(all_Z)
    
    #Find all layers
    layers_0 = []
    for i in range(len(all_points)):
        layers_0.append(all_points[i][3])
    
    layers_1 = list(dict.fromkeys(layers_0))
    layers_1.sort()
        
    #Add inactive points if desired
    # all_points_out = all_points.tolist().copy()
    all_points_out = all_points.copy()
    if add_inactive:
        for index in range(len(all_Z_indices)):
            all_points_out.append(np.array([math.nan, math.nan, all_Z[int(index)], all_Z_indices[index]]))
            
    #Use "output_form" coordinates:
    if output_form == "dec":
        all_points_dec = []
        for point in all_points_out:
            all_points_dec.append(cart_to_dec(point, all_X, all_Y))
    else:
        return "ERROR!"
    
    #Collect the points in all_points depending on their layers
    return [[point for point in all_points_dec if point[3] == index] for index in all_Z_indices]

### Superposition between a set of input points
def l_sup(all_points_ordered, dataset, input_form = "dec", output_form = "qubit", tracksters_to_be_removed = []):
    # Input 1: all points to be considered in the search. Each element in "all_points_ordered" contains all the points of a given layer
    # Input 2: dataset fed into grover
    # Input 3: input form - currently is ONLY "dec"
    # Input 4: output form - choose whether outputting the state in "dec" or "qubit" form
    # Output: superposition state in qubit form
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    
    #Dimensions of the system:
    DimX, DimY = n_qubits(len(all_X),len(all_Y))
    DimZ = len(all_Z)
    
    #Find and return all possible tracksters 
    all_tracksters = [list(p) for p in itertools.product(*all_points_ordered)]
    
    #Remove tracksters that the user wants NOT to search over
    if len(tracksters_to_be_removed) != 0:
        for track in tracksters_to_be_removed:
            remove_array_from_list(all_tracksters, track)
    
    #If there are no points left, return the empty array
    if len(all_tracksters) == 0:
        return []
    
    #Return in the desired form:
    if output_form == "dec":
        return all_tracksters
    elif output_form == "qubit":
        return np.sum([full_dec_to_qubit(trackster, dataset) for trackster in all_tracksters])/np.sqrt(len(all_tracksters))

### Marking operator: the black box!
def remove_array_from_list(full_list, list_to_remove):
    for index in range(len(full_list)):
        if np.array_equal(full_list[index], list_to_remove):
            full_list.pop(index)
            break

def black_box(thresholds, all_points_ordered, dataset, input_form = "dec", output_form = "Operator", tracksters_to_be_removed = [], Printing = False):
    # Input 1: thresholds to be used. First one is when there are no missing layers in between, second one when there are. 
    # Input 2: all points to be considered in the search. Each element in "all_points_ordered" contains all the points of a given layer
    # Input 3: dataset fed into grover
    # Input 4: input form - currently is ONLY "dec"
    # Input 5: output form - Decides whether spit out the operator ()"Operator") or the list of the points that have been found ("List")
    # Output: operator that gives a phase every time the thresholds are satisfied
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    
    #Define the thresholds to be used:
    thrs_0 = thresholds[0]
    thrs_1 = thresholds[1]
    thrs_2 = thresholds[2]
    thrs_3 = thresholds[3]
    
    #Dimensions of the system:
    DimX, DimY = n_qubits(len(all_X),len(all_Y))
    DimZ = len(all_Z)
    
    #Find all possible tracksters in dec form:
    all_tracksters_dec = l_sup(all_points_ordered, dataset, input_form = input_form, output_form = "dec")
    
    #Remove tracksters that the user wants NOT to search over
    if len(tracksters_to_be_removed) != 0:
        for track in tracksters_to_be_removed:
            remove_array_from_list(all_tracksters_dec, track)
    
    #If there are no points left, return the identity
    if len(all_tracksters_dec) == 0:
        if output_form == "Operator":
            return pauli_gen("I" , 0 , (DimX + DimY)*DimZ)
        elif output_form == "List":
            return []
    
    #Find all distances:
    all_dists = []
    for track in all_tracksters_dec:
        all_dists.append(f_dist_t(track, dataset, "dec"))
    
    #Initialize the vector containing the tracksters that satisfy the criteria:
    good_track = []
    counter = 0
    
    #Find all points that have no inactive layer and satisfy the threshold
    for (i,dist) in enumerate(all_dists):
        if list(dist)[0] < thrs_0:
            good_track.append(all_tracksters_dec[i])
            counter = counter + 1
    
    #Assuming there are no points as the previous ones, we proceed by finding points that have an inactive component from a particle skipping a layer
    if counter == 0:
        for (i,dist) in enumerate(all_dists):
            if list(dist)[1] < thrs_1:
                good_track.append(all_tracksters_dec[i])
                counter = counter + 1
                
    #Assuming there are no points as the previous ones, we proceed by finding tracksters of particles either generated or absorbed at some point
    if counter == 0:
        for (i,dist) in enumerate(all_dists):
            if list(dist)[2] < thrs_2:
                good_track.append(all_tracksters_dec[i])
                counter = counter + 1
        
    #Assuming there are no points as the previous ones, we proceed by finding tracksters of particles either generated or absorbed at some point, WITH a missing layer somewhere
    if counter == 0:
        for (i,dist) in enumerate(all_dists):
            if list(dist)[3] < thrs_3:
                good_track.append(all_tracksters_dec[i])
                counter = counter + 1  
    
    #Mark only the longest tracksters (we choose the one with less holes first!)
    n_holes_vec = []
    
    for track in good_track:
        n_holes_vec.append(f_dist_t(track, dataset, input_form, out_length = True))
    
    if len(n_holes_vec) != 0:
        min_num = min(n_holes_vec)
        indices_min = [i for i, v in enumerate(n_holes_vec) if v == min_num]
    
        fin_track = [good_track[i] for i in indices_min]
    else:
        fin_track = good_track
        
    # Build the desired discerning operator:
    if len(fin_track) == 0:
        if output_form == "Operator":
            return pauli_gen("I" , 0 , (DimX + DimY)*DimZ)
        elif output_form == "List":
            return fin_track
    else: 
        #States to be marked:
        st_to_mark = np.sum([full_dec_to_qubit(track, dataset) for track in fin_track])/np.sqrt(len(fin_track))
        #st_to_mark = [full_dec_to_qubit(track, dataset) for track in fin_track]
        #op_proj = np.sum([st_proj(st,st) for st in st_to_mark])
        if output_form == "Operator":
            bb_op = pauli_gen("I" , 0 , (DimX + DimY)*DimZ) - 2 * st_proj(st_to_mark,st_to_mark)
            #bb_op = pauli_gen("I" , 0 , (DimX + DimY)*DimZ) - 2 * op_proj
            if Printing:
                print("Unitarity of the black box operator (0 is good):", abs(bb_op.conjugate().transpose().dot(bb_op) - pauli_gen("I" , 0 , (DimX + DimY)*DimZ)).max())
            return bb_op
        elif output_form == "List":
            return fin_track
 
### Grover routine
def Grover(thresholds, all_points_ordered, dataset, input_form = "dec", output_form = "dec", tracksters_to_be_removed = [], Printing = False):
    # Input 1: thresholds to be used. First one is when there are no missing layers in between, second one when there are. 
    # Input 2: all points to be considered in the search. Each element in "all_points_ordered" contains all the points of a given layer
    # Input 3: dataset fed into grover
    # Input 4: input form - currently is ONLY "dec"
    # Input 5: output form - Decides whether spit out the qubit state ("qubit") or the dec state ("dec")
    # Output: state found by grover
    
    all_X = dataset[0]
    all_Y = dataset[1]
    all_Z = dataset[2]
    all_Z_indices = dataset[3]

    
    #Define the thresholds to be used:
    thrs_0 = thresholds[0]
    thrs_1 = thresholds[1]
    thrs_2 = thresholds[2]
    thrs_3 = thresholds[3]
    
    #Dimensions of the system:
    DimX, DimY = n_qubits(len(all_X),len(all_Y))
    DimZ = len(all_Z)
    
    # if Printing==True:
    #     print('DimX, DimY: ', DimX, DimY)

    #If there are not enough layers, return an empty array:
    if len(all_Z_indices) < 4:
        return []
    
    #Do the Grover search until we find all points:
    #Find all possible tracksters:
    all_tracksters_dec = l_sup(all_points_ordered, dataset, input_form = input_form, output_form = "dec", tracksters_to_be_removed = tracksters_to_be_removed)
    all_tracksters_qubit = l_sup(all_points_ordered, dataset, input_form = input_form, output_form = "qubit", tracksters_to_be_removed = tracksters_to_be_removed)
    
    #Define the relevant operators:
    #black box
    bb = black_box(thresholds, all_points_ordered, dataset, input_form = input_form, output_form = "Operator", tracksters_to_be_removed = tracksters_to_be_removed, Printing = Printing)
    bb_list = black_box(thresholds, all_points_ordered, dataset, input_form = input_form, output_form = "List", tracksters_to_be_removed = tracksters_to_be_removed, Printing = Printing)
    #state projector
    st_proj_op = 2*st_proj(all_tracksters_qubit,all_tracksters_qubit) - pauli_gen("I" , 0 , (DimX + DimY)*DimZ)
    if Printing:
        print("Grover box: number of points still to be found:", len(black_box(thresholds, all_points_ordered, dataset, input_form = input_form, output_form = "List", tracksters_to_be_removed = tracksters_to_be_removed)))
        print("Unitarity of the state projector (0 is good): ", abs(st_proj_op.conjugate().transpose().dot(st_proj_op) - pauli_gen("I" , 0 , (DimX + DimY)*DimZ)).max())
    
    #Number of iterations:
    if len(bb_list) != 0:
        Niter = np.round(np.pi / (4 * np.arcsin(np.sqrt(len(bb_list)/len(all_tracksters_dec)))) - 1/2)
    else:
        Niter = 1
    if Printing:
        print("Total number of trackster in input: ", len(all_tracksters_dec))
        print("Number of iterations: ", Niter)
    
    #Grover loop:
    tmp_state = all_tracksters_qubit
    for i in range(int(Niter)):
        tmp_state = bb.dot(tmp_state)
        tmp_state = st_proj_op.dot(tmp_state)
    
    #Find the resulting state:
    res_state = full_qubit_to_dec(tmp_state, dataset, form = output_form)
    #print("distances of resulting state = ",f_dist_t(res_state, dataset, "dec"))
    return res_state



################
### TESTS ###
################
if __name__=="__main__":

    ################
    ### CHECK THAT THE DISTANCES BETWEEN POINTS ARE WELL DEFINED ###
    ################

    # Choose number of iterations:
    Ntry = 100

    # Choosing random dimensions
    Nx = random.randint(1,3)
    Ny = random.randint(1,8)
    Nz = random.randint(2,4)

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
        rand_index = np.random.randint(0,Nz - 1)
        
        point_list_cart.append([random.choice(all_X),random.choice(all_Y),all_Z[rand_index],rand_index])
        point_list_cart.append([random.choice(all_X),random.choice(all_Y),all_Z[rand_index + 1],rand_index + 1])
        if random.random() < 20*0.5 / Ntry:
            rand_index = np.random.randint(0,2)
            point_list_cart[rand_index][0] = math.nan
            point_list_cart[rand_index][1] = math.nan

        # Define points in different coordinates:
        point_1_cart = np.array(point_list_cart[0])
        point_2_cart = np.array(point_list_cart[1])
        point_1_dec = cart_to_dec(point_1_cart, dataset[0], dataset[1])
        point_2_dec = cart_to_dec(point_2_cart, dataset[0], dataset[1])
        point_1_qubit = [dec_to_qubit(point_1_dec, dataset[0], dataset[1]), point_1_cart[2], point_1_cart[3]]
        point_2_qubit = [dec_to_qubit(point_2_dec, dataset[0], dataset[1]), point_2_cart[2], point_2_cart[3]]

        dist_cart = f_dist(point_1_cart, point_2_cart, dataset, input_form = "cart")
        dist_dec = f_dist(point_1_dec, point_2_dec, dataset, input_form = "dec")
        dist_qubit = f_dist(point_1_qubit, point_2_qubit, dataset, input_form = "qubit")

        diff_cart_dec_ = np.abs(dist_cart - dist_dec)
        diff_cart_qubit_ = np.abs(dist_cart - dist_qubit)

        if diff_cart_dec_>10**-10 or diff_cart_qubit_>10**-10:
            print("Error in calculating the distance!!!")


    ################
    ### CHECK THAT THE DISTANCES IN TRACKSTERS ARE WELL DEFINED ###
    ################

    # Choose number of iterations:
    Ntry = 100

    # Choosing random dimensions
    Nx = random.randint(1,3)
    Ny = random.randint(1,7)
    Nz = random.randint(2,5)

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
            point_list_cart.append(np.array([random.choice(all_X),random.choice(all_Y),all_Z[i],i]))
            if random.random() < 100*0.5 / (Ntry):
                point_list_cart[i][0] = math.nan
                point_list_cart[i][1] = math.nan
        
        # Find the decimal form of the point:
        point_list_dec = [cart_to_dec(point_list_cart[i], dataset[0], dataset[1]) for i in range(Nz)]    
        # Find the corresponding qubit vector:
        point_qubit = full_dec_to_qubit(point_list_dec, dataset)
        
        dist_cart = f_dist_t(point_list_cart, dataset, "cart")
        dist_dec = f_dist_t(point_list_dec, dataset, "dec")
        dist_qubit = f_dist_t(point_qubit, dataset, "qubit")
        
        if not (dist_cart == dist_dec and dist_cart == dist_qubit and dist_qubit == dist_dec):
            print("We have a prolem with the distances!")


    ################
    ### EXAMPLE OF SUPERPOSITION OF A SET OF INPUT POINTS ###
    ################

    # Choosing random dimensions
    Nx = random.randint(1,3)
    Ny = random.randint(1,7)
    Nz = random.randint(2,4)

    if Nx == 1 and Ny == 1:
        Ny = 2

    # Total number of points that we want to use:
    N_max = Nx * Ny * Nz
    N_chosen = int(np.floor(N_max/2))

    # Calculating the dimensions for the vector
    DimX, DimY = n_qubits(Nx,Ny)

    # Generating random data:
    all_X = np.sort(np.array([random.random() for i in range(Nx)]))
    all_Y = np.sort(np.array([random.random() for i in range(Ny)]))
    all_Z = np.sort(np.array([random.random() for i in range(Nz)]))
    all_Z_indices = [i for i in range(Nz)]

    dataset = [all_X, all_Y, all_Z, all_Z_indices]

    # Randomly pick some points in these data:
    point_list_cart = []
    for i in range(N_chosen):
        rand_int = random.randint(0,Nz-1)
        point_list_cart.append(np.array([random.choice(all_X),random.choice(all_Y),all_Z[int(rand_int)],rand_int]))
    #all_points_cart = list(dict.fromkeys(point_list_cart))    
    tupled_lst = set(map(tuple, point_list_cart))
    point_list_cart = list(map(list, tupled_lst))

    # Use the function "points_layer_collection" for splitting the point into the different layers:
    all_points_ordered = points_layer_collection(point_list_cart, dataset)

    # Generate all tracksters in output_form = "dec" or "qubit" form:
    all_tracksters = l_sup(all_points_ordered, dataset, output_form = "qubit")
    print(all_tracksters)


    ################
    ### CHECK THE BLACKBOX ###
    ################

    # Choosing thresholds:
    thresholds = [10.,10.,10.,10.]

    # Choosing random dimensions
    Nx = random.randint(2,3)
    Ny = random.randint(2,2)
    Nz = random.randint(4,5)

    if Nx == 1 and Ny == 1:
        Ny = 2

    # Total number of points that we want to use:
    N_max = Nx * Ny * Nz
    N_chosen = int(np.floor(N_max/2))

    # Generating random data:
    all_X = np.sort(np.array([random.random() for i in range(Nx)]))
    all_Y = np.sort(np.array([random.random() for i in range(Ny)]))
    all_Z = np.sort(np.array([random.random() for i in range(Nz)]))
    all_Z_indices = [i for i in range(Nz)]

    dataset = [all_X, all_Y, all_Z, all_Z_indices]

    # Randomly pick some points in these data:
    point_list_cart = []
    for i in range(N_chosen):
        rand_int = random.randint(0,Nz-1)
        point_list_cart.append(np.array([random.choice(all_X),random.choice(all_Y),all_Z[int(rand_int)],rand_int]))
    #all_points_cart = list(dict.fromkeys(point_list_cart))    
    tupled_lst = set(map(tuple, point_list_cart))
    point_list_cart = list(map(list, tupled_lst))

    # Use the function "points_layer_collection" for splitting the point into the different layers:
    all_points_ordered = points_layer_collection(point_list_cart, dataset)

    Tracksters_found_classically = []
    keep_going = True
    Round = 1
    while keep_going:
        print("Round: ", Round)
        bb = black_box(thresholds, all_points_ordered, dataset, input_form = "dec", output_form = "Operator", tracksters_to_be_removed = Tracksters_found_classically)
        new_tracksters = black_box(thresholds, all_points_ordered, dataset, input_form = "dec", output_form = "List", tracksters_to_be_removed = Tracksters_found_classically)
        Tracksters_found_classically = Tracksters_found_classically + new_tracksters
        if len(new_tracksters) != 0:
            for track in new_tracksters:
                tmp_vec = full_dec_to_qubit(track, dataset)
                tmp_avg = tmp_vec.conjugate().transpose().dot(op_state([bb],tmp_vec)).toarray()[0][0]
                #print(tmp_avg)
                #if np.abs(1 + tmp_avg) > 10**-10:
                if np.abs(1-2/len(new_tracksters) - tmp_avg)> 10**-10:
                    print("Error with the black box!")
                    print(tmp_avg)
                    print(1-2/len(new_tracksters))
        else:
            keep_going = False
        Round = Round + 1

    print("distances found:")
    print([f_dist_t(track, dataset, "dec") for track in Tracksters_found_classically])


    ################
    ### EXAMPLE OF GROVER ROUTINE ###
    ################

    # Choosing thresholds:
    thresholds = [5.,5.,5.,5.]

    # Choosing random dimensions
    Nx = random.randint(2,2)
    Ny = random.randint(2,2)
    Nz = random.randint(4,4)

    if Nx == 1 and Ny == 1:
        Ny = 2

    # Total number of points that we want to use:
    N_max = Nx * Ny * Nz
    N_chosen = int(np.floor(N_max/2))

    # Generating random data:
    all_X = np.sort(np.array([random.random() for i in range(Nx)]))
    all_Y = np.sort(np.array([random.random() for i in range(Ny)]))
    all_Z = np.sort(np.array([random.random() for i in range(Nz)]))
    all_Z_indices = [i for i in range(Nz)]

    dataset = [all_X, all_Y, all_Z, all_Z_indices]
    print('\n\n*************************************\n\n')
    print(dataset)
    print('\n\n*************************************\n\n')

    # Randomly pick some points in these data:
    point_list_cart = []
    for i in range(N_chosen):
        rand_int = random.randint(0,Nz-1)
        point_list_cart.append(np.array([random.choice(all_X),random.choice(all_Y),all_Z[int(rand_int)],rand_int]))
    #all_points_cart = list(dict.fromkeys(point_list_cart))    
    tupled_lst = set(map(tuple, point_list_cart))
    point_list_cart = list(map(list, tupled_lst))

    # Use the function "points_layer_collection" for splitting the point into the different layers:
    all_points_ordered = points_layer_collection(point_list_cart, dataset)
    print('\n\n*************************************\n\n')
    print(all_points_ordered)
    print('\n\n*************************************\n\n')

    #CLASSICAL
    # Classically find all points satisfying the threshold criteria:
    Tracksters_found_classically = black_box(thresholds, all_points_ordered, dataset, input_form = "dec", output_form = "List")
    keep_going = True

    while keep_going:
        tmp = black_box(thresholds, all_points_ordered, dataset, input_form = "dec", output_form = "List", tracksters_to_be_removed = Tracksters_found_classically)
        if len(tmp) != 0:
            Tracksters_found_classically = Tracksters_found_classically + tmp
        else:
            keep_going = False
            
    #Determine all distances of the classically found tracksters:
    dists_classical = [f_dist_t(track, dataset, "dec") for track in Tracksters_found_classically]

    #QUANTUM
    #Now do the same quantum.
    keep_going = True

    #First Grover search:
    tmp = [Grover(thresholds, all_points_ordered, dataset, Printing = True)]
    dist_tmp = f_dist_t(tmp[0], dataset, "dec")
    if (len(tmp) != 0) and ((dist_tmp[0] < thresholds[0]) or (dist_tmp[1] < thresholds[1]) or (dist_tmp[2] < thresholds[2]) or (dist_tmp[3] < thresholds[3])):
        Tracksters_found_quantumly = tmp
    else:
        keep_going = False
    print("distances of first point: ",dist_tmp)

    #All subsequent Grover searches:
    while keep_going:
        print("")
        print("new iteration")
        print("points found so far: ", len(Tracksters_found_quantumly))
        tmp = [Grover(thresholds, all_points_ordered, dataset, tracksters_to_be_removed = Tracksters_found_quantumly, Printing = True)]
        dist_tmp = f_dist_t(tmp[0], dataset, "dec")
        print("distance found at this iteration:", dist_tmp)
        if (len(tmp) != 0) and ((dist_tmp[0] < thresholds[0]) or (dist_tmp[1] < thresholds[1]) or (dist_tmp[2] < thresholds[2]) or (dist_tmp[3] < thresholds[3])):
            Tracksters_found_quantumly = Tracksters_found_quantumly + tmp
        else:
            keep_going = False
            
    dists_quantum = [f_dist_t(track, dataset, "dec") for track in Tracksters_found_quantumly]
