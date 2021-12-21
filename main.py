################################################################
################################################################
### ******** MAIN FILE FOR QTICL WITH GROVER SEARCH ******** ###
################################################################
################################################################


import numpy as np
import pandas as pd
import math
from grover_op import *
import argparse


def createGrid(x,y,z):
    gridstructure = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                gridstructure.append(np.array([x[i],y[j],z[k],k,0]))

    gridstructure = np.array(gridstructure)
    return gridstructure


def fillTheGrid(gridstructure, x, y, z, lxy, lz):
    for i in range(len(x)):
        gridstructure[np.where((gridstructure[:,0]+lxy/2>x[i]) & (gridstructure[:,0]-lxy/2<x[i])
                        & (gridstructure[:,1]+lxy/2>y[i]) & (gridstructure[:,1]-lxy/2<y[i]) 
                        & (gridstructure[:,2]+lz>z[i]) & (gridstructure[:,2]-lz<z[i])), 4] += 1
    return gridstructure


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./')
    args = parser.parse_args()

    # Set directory
    data_dir = args.dir

    # Import the data for Grover
    grover_data = pd.read_csv(data_dir+ "diphoton.csv")

    allID = grover_data['lcID'].values
    allEta = grover_data['lcEta'].values
    allPhi = grover_data['lcPhi'].values
    allLayer = grover_data['lcLayer'].values
    allX = grover_data['lcX'].values
    allY = grover_data['lcY'].values
    allZ = grover_data['lcZ'].values

    allTracks = grover_data['lcToTrack1'].values

    # Set the kind of coordinates to use
    CoordUsed = 'pseudo'

    # Set the thresholds
    gridThreshold = 5  #10
    distThreshold = 3.5

    # Choose maximum dimension of grover_data to be fed Grover
    grover_size = [7,8,4]

    #### Can add small padding
    minX = min(allX)
    minY = min(allY)
    minZ = min(allZ)
    maxX = max(allX)
    maxY = max(allY)
    maxZ = max(allZ)

    uniqueL = np.unique(allLayer)

    cubesZ = []
    for i in uniqueL:
        index = allLayer.tolist().index(i)
        cubesZ.append(allZ[index])

    maxDistance = 0
    minDistance = 100
    for i in range(len(cubesZ)-1):
        tempDistance = cubesZ[i+1]-cubesZ[i]
        if (tempDistance>maxDistance and tempDistance<distThreshold):
            maxDistance = tempDistance
        if (tempDistance<minDistance):
            minDistance = tempDistance

    # Define zTolerance for inserting points in the grid with respect to z value
    zTolerance = minDistance/2

    # Pad z to have integer number of cubes for grover
    Nz_rate = len(cubesZ)/grover_size[2]
    z_pad = int(np.round((1-math.modf(Nz_rate)[0])*grover_size[2]))
    for k in range(z_pad):
        cubesZ.append(cubesZ[len(cubesZ)-1]+minDistance)

    # Define xy distance
    tileL = (maxDistance*gridThreshold/(2*np.sqrt(2)))

    Nx = (maxX-minX)/tileL
    Ny = (maxY-minY)/tileL

    maxX = maxX + tileL*(1-math.modf(Nx)[0])/2
    minX = minX - tileL*(1-math.modf(Nx)[0])/2
    maxY = maxY + tileL*(1-math.modf(Ny)[0])/2
    minY = minY - tileL*(1-math.modf(Ny)[0])/2

    Nx = int(np.round((maxX-minX)/tileL))
    Ny = int(np.round((maxY-minY)/tileL))

    cubesX = []
    cubesY = []

    # Pad x and y to have integer number of cubes for grover
    Nx_rate = Nx/grover_size[0]
    Ny_rate = Ny/grover_size[1]

    Nx += int(np.round((1-math.modf(Nx_rate)[0])*grover_size[0]))
    Ny += int(np.round((1-math.modf(Ny_rate)[0])*grover_size[1]))

    # Create list of coordinates of the cubes 
    for i in range(Nx):
        cubesX.append(minX+tileL/2 + i*tileL)
    for i in range(Ny):
        cubesY.append(minY+tileL/2 + i*tileL)


    # Now create the grid with each cube represented by a list (x,y,z,0) where 0 is the cardinality (# of points in the cube)
    gridStructure = createGrid(cubesX, cubesY, cubesZ)

    ##### gridStructure elements can be accessed easily: gridStructure[0,2] gives access to z (2) of the first cube (0)
    ##### The first element indicates the cubeID, the second corresponds to the feature: 0 for x, 1 for y, 2 for z, 3 for layer and 4 for cardinality

    # Fill the grid 
    fillTheGrid(gridStructure, allX, allY, allZ, tileL, zTolerance)

    print('\n **** \n')

    # for i in range(int(len(cubesX)/grover_size[0])):
    #     for j in range(int(len(cubesY)/grover_size[1])):
    #         for k in range(int(len(cubesZ)/grover_size[2])):
    #             Grover(gridStructure[np.where((gridStructure[:,0]>=cubesX[i*grover_size[0]]) & (gridStructure[:,0]<cubesX[(i+1)*grover_size[0]]) 
    #                                     & (gridStructure[:,1]>=cubesY[j*grover_size[1]]) & (gridStructure[:,1]<cubesY[(j+1)*grover_size[1]]) 
    #                                     & (gridStructure[:,2]>=cubesZ[k*grover_size[2]]) & (gridStructure[:,2]<cubesZ[(k+1)*grover_size[2]]))])

    # Choosing thresholds:
    thresholds = [gridThreshold,gridThreshold,gridThreshold,gridThreshold]

    # # Choosing random dimensions
    # Nx = random.randint(2,2)
    # Ny = random.randint(2,2)
    # Nz = random.randint(4,4)

    # if Nx == 1 and Ny == 1:
    #     Ny = 2

    # # Total number of points that we want to use:
    # N_max = Nx * Ny * Nz
    # N_chosen = int(np.floor(N_max/2))

    # for i in range(int(len(cubesX)/grover_size[0])):
    #     for j in range(int(len(cubesY)/grover_size[1])):
    #         for k in range(int(len(cubesZ)/grover_size[2])):
    i,j,k=[7,1,4]
    condition_indices = np.where((gridStructure[:,0]>=cubesX[i*grover_size[0]]) & (gridStructure[:,0]<=cubesX[(i+1)*grover_size[0]-1]) 
                                        & (gridStructure[:,1]>=cubesY[j*grover_size[1]]) & (gridStructure[:,1]<=cubesY[(j+1)*grover_size[1]-1]) 
                                        & (gridStructure[:,2]>=cubesZ[k*grover_size[2]]) & (gridStructure[:,2]<=cubesZ[(k+1)*grover_size[2]-1]))

    gridTest = gridStructure[condition_indices]
    
    all_X = np.unique(gridTest[:,0])
    all_Y = np.unique(gridTest[:,1])
    all_Z = np.unique(gridTest[:,2])
    all_Z_indices = np.unique(gridTest[:,3])

    dataset = [all_X, all_Y, all_Z, all_Z_indices]

    temp = gridTest[np.where(gridTest[:,4]!=0)]
    occupied_cubes = [np.array(k) for k in temp]
    # print(occupied_cubes)
                # if(len(occupied_cubes)>10):
                #     print(i,j,k)
                #     print(len(occupied_cubes))
                # print(occupied_cubes)

    # Use the function "points_layer_collection" for splitting the point into the different layers:
    all_points_ordered = points_layer_collection(occupied_cubes, dataset)
    # print(all_points_ordered)

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
        print('Tracksters found quantumly: ', tmp)
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
            print('Tracksters found quantumly: ', tmp)
        else:
            keep_going = False
            
    dists_quantum = [f_dist_t(track, dataset, "dec") for track in Tracksters_found_quantumly]