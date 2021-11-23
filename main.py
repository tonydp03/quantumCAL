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
    for i in range(len(cubesX)):
        for j in range(len(cubesY)):
            for k in range(len(cubesZ)):
                gridstructure.append(np.array([cubesX[i],cubesY[j],cubesZ[k],0]))

    gridstructure = np.array(gridstructure)
    return gridstructure


def fillTheGrid(gridstructure, x, y, z, lxy, lz):
    for i in range(len(x)):
        gridstructure[np.where((gridstructure[:,0]+lxy/2>x[i]) & (gridstructure[:,0]-lxy/2<x[i])
                        & (gridstructure[:,1]+lxy/2>y[i]) & (gridstructure[:,1]-lxy/2<y[i]) 
                        & (gridstructure[:,2]+lz>z[i]) & (gridstructure[:,2]-lz<z[i])), 3] += 1
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
    ##### The first element indicates the cubeID, the second corresponds to the feature: 0 for x, 1 for y, 2 for z and 3 for cardinality

    # Fill the grid 
    fillTheGrid(gridStructure, allX, allY, allZ, tileL, zTolerance)

    print('\n **** \n')

    # for i in range(int(len(cubesX)/grover_size[0])):
    #     for j in range(int(len(cubesY)/grover_size[1])):
    #         for k in range(int(len(cubesZ)/grover_size[2])):
    #             Grover(gridStructure[np.where((gridStructure[:,0]>=cubesX[i*grover_size[0]]) & (gridStructure[:,0]<cubesX[(i+1)*grover_size[0]]) 
    #                                     & (gridStructure[:,1]>=cubesY[j*grover_size[1]]) & (gridStructure[:,1]<cubesY[(j+1)*grover_size[1]]) 
    #                                     & (gridStructure[:,2]>=cubesZ[k*grover_size[2]]) & (gridStructure[:,2]<cubesZ[(k+1)*grover_size[2]]))])
