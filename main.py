################################################################
################################################################
### ******** MAIN FILE FOR QTICL WITH GROVER SEARCH ******** ###
################################################################
################################################################


from re import L
import numpy as np
import pandas as pd
import math
from grover_op import *
import argparse
from plot_utils import *
import matplotlib.pyplot as plt
import itertools


def createGrid(x,y,z):
    gridstructure = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                gridstructure.append(np.array([x[i],y[j],z[k],k,0,[]], dtype=object))

    gridstructure = np.array(gridstructure)
    return gridstructure


def fillTheGrid(gridstructure, x, y, z, layer, energy, lxy, lz):
    for i in range(len(x)):
        condition_found = np.where((gridstructure[:,0]+lxy/2>x[i]) & (gridstructure[:,0]-lxy/2<x[i])
                        & (gridstructure[:,1]+lxy/2>y[i]) & (gridstructure[:,1]-lxy/2<y[i]) 
                        & (gridstructure[:,2]+lz>z[i]) & (gridstructure[:,2]-lz<z[i]))
        gridstructure[condition_found, 4] += 1
        gridstructure[condition_found, 5][0][0].append(np.array([x[i],y[i],z[i],layer[i],energy[i]]))

    return gridstructure

def trkIsValid(lcInTrackster, energyThrs, energyThrsCumulative):
    lcEnergy = [[lc_energy[4] for lc_energy in lc] for lc in lcInTrackster]
    energyIndices = [[i for i in range(len(energies))] for energies in lcEnergy]

    allPaths = list(itertools.product(*energyIndices))
    # print(f'\n {lcEnergy}')
    # print(f'{allPaths}')

    totDiff = []
    for path in allPaths:
        enDiff = 0
        for i,k in enumerate(path):
            if(i>0):
                enContr = np.abs((lcEnergy[i][k] - curr)/(lcEnergy[i][k] + curr))
                if(enContr < energyThrs):
                    enDiff += enContr
                else:
                    enDiff += float('inf')
            curr = lcEnergy[i][k]
            # print('CURR {} & DIFF {}'.format(curr,enDiff))
        totDiff.append(enDiff)
    # print('TotDiff {}'.format(totDiff))
    # print('MinTotDiff {}'.format(np.min(totDiff)))
    minTotDiff = np.min(totDiff)
    argMinTotDiff = np.where(totDiff == minTotDiff)[0][0]
    # print('ArgMinTotDiff {}'.format(argMinTotDiff))
    minIndices = allPaths[argMinTotDiff] 
    if(minTotDiff > energyThrsCumulative):
        return []
    else:
        # return allPaths[argMinTotDiff]
        return [lcInTrackster[i][minIndices[i]] for i in range(len(minIndices))]
        # return [lc_list[i] for lc_list in len(lcInTrackster)]

    # for i,e in enumerate(lcEnergy[:-1]):
    #     curr = e
    #     nxt = lcEnergy[i+1]
    #     for j,c in enumerate(curr):
    #         for k,n in enumerate(nxt):
    #             diff = np.abs((c-n)/(c+n))
                # print(f'{c,n,(c-n), (c+n), diff}')    
               # ABS(MAX[(E1-E2)/(E1+E2)])>EN_threshold 

    # print('LC_ENERGY: {}'.format(lcEnergy))



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
    allEnergies = grover_data['lcEnergy'].values
    allTracks = grover_data['lcToTrack1'].values

    # Set the kind of coordinates to use
    CoordUsed = 'pseudo'

    # Choose maximum dimension of grover_data to be fed Grover
    grover_size = [3,4,4]

    # Set the thresholds
    gridThreshold = 5  #10
    distThreshold = 3.5
    enThreshold = 0.8
    enThresholdCumulative = 0.5 * grover_size[2]

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


    # Now create the grid with each cube represented by a list (x,y,z,k,0) where k is the layer number and 0 is the cardinality (# of points in the cube)
    # print('Creating the grid...')
    # gridStructure = createGrid(cubesX, cubesY, cubesZ)
    # print('Grid created!')

    ##### gridStructure elements can be accessed easily: gridStructure[0,2] gives access to z (2) of the first cube (0)
    ##### The first element indicates the cubeID, the second corresponds to the feature: 0 for x, 1 for y, 2 for z, 3 for layer and 4 for cardinality

    # Fill the grid 
    # print('Filling the grid...')
    # This is super slow! Maybe we can run it once and save the grid in a csv file
    # fillTheGrid(gridStructure, allX, allY, allZ, allLayer, allEnergies, tileL, zTolerance)
    # print('Grid filled!')

    gridStructure = np.load('grid.npy', allow_pickle=True)

    # Choosing thresholds:
    thresholds = [gridThreshold,gridThreshold,gridThreshold,gridThreshold]
    xs = list()
    ys = list()
    zs = list()

    LCs_columns = ['i', 'j', 'k', 'x', 'y', 'z', 'layer', 'energy', 'TrkId']
    allTrksFoundQuantumly = pd.DataFrame([],columns=LCs_columns)
    numTrk = 0
    # allTrksFoundQuantumly.loc[len(allTrksFoundQuantumly)] = testList
    # print(allTrksFoundQuantumly)

    for i in range(int(len(cubesX)/grover_size[0])):
        for j in range(int(len(cubesY)/grover_size[1])):
            for k in range(int(len(cubesZ)/grover_size[2])):
                # i,j,k = [1,13,4]
                i,j = [2,14] # also 2,14,2-3-4-6 --- for cubes with cardinality > 1
                # print(f" Shape grid structur {gridStructure.shape} {type(gridStructure)}")
                condition_indices = np.where((gridStructure[:,0]>=cubesX[i*grover_size[0]]) & (gridStructure[:,0]<=cubesX[(i+1)*grover_size[0]-1]) 
                                        & (gridStructure[:,1]>=cubesY[j*grover_size[1]]) & (gridStructure[:,1]<=cubesY[(j+1)*grover_size[1]-1]) 
                                        & (gridStructure[:,2]>=cubesZ[k*grover_size[2]]) & (gridStructure[:,2]<=cubesZ[(k+1)*grover_size[2]-1]))

                gridTest = gridStructure[condition_indices]

                # print('**** GRID TEST***\n', gridTest.shape, type(gridTest))
                # print(f"gridStructure 6 {type(gridStructure[0,5])} {gridStructure[0,5]}")

                all_X = np.unique(gridTest[:,0])
                all_Y = np.unique(gridTest[:,1])
                all_Z = np.unique(gridTest[:,2])
                all_Z_indices = np.unique(gridTest[:,3])

                dataset = [all_X, all_Y, all_Z, all_Z_indices]

                temp = gridTest[np.where(gridTest[:,4]!=0)]
                occupied_cubes = [np.array(k) for k in temp]

                # Use the function "points_layer_collection" for splitting the point into the different layers:
                all_points_ordered = points_layer_collection(occupied_cubes, dataset)

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
                Tracksters_to_remove = [] # this is necessary otherwise dists_quantum at the end of the search will fail if no Trackster is found

                # Grover search:
                while(keep_going):
                    # tmp = [Grover(thresholds, all_points_ordered, dataset, Printing = False)]
                    print('\n ****** Iteration starts ****** \n')
                    temp = gridTest[np.where(gridTest[:,4]!=0)]
                    # print('TEMP ******** \n\n', temp)
                    occupied_cubes = [np.array(k) for k in temp]
                    print('OCCUPIED CUBES ******** \n', occupied_cubes)

                    # Use the function "points_layer_collection" for splitting the point into the different layers:
                    all_points_ordered = points_layer_collection(occupied_cubes, dataset)
                    print('ALL POINTS ORDERED ******** \n', all_points_ordered)

                    # Grover Iteration
                    print("\n **** Grover call! ****")
                    tmp = [Grover(thresholds, all_points_ordered, dataset, tracksters_to_be_removed = Tracksters_to_remove, Printing = False)]
                    print("\n **** Grover found something! ****")
                    # print('Tmp: ', tmp)
                    dist_tmp = f_dist_t(tmp[0], dataset, "dec")

                    if (len(tmp) != 0) and ((dist_tmp[0] < thresholds[0]) or (dist_tmp[1] < thresholds[1]) or (dist_tmp[2] < thresholds[2]) or (dist_tmp[3] < thresholds[3])):
                        lc_inTrk = []
                        
                        print('*** Distance conditions satisfied! ***')
                        all_condition_indices_ls = list()
                        for lc in tmp[0]:
                            # print(lc[0])
                            # print(len(gridTest[0]))
                            # if(lc[0]<len(gridTest[0])): # the LC is valid (not hole)
                            coordinates_lc = dec_to_cart(lc,dataset)
                            if (not np.isnan(coordinates_lc[0])): # the LC is valid (not hole)
                                condition_indices_lc = np.where((gridTest[:,0]==coordinates_lc[0]) & (gridTest[:,1]==coordinates_lc[1]) & (gridTest[:,2]==coordinates_lc[2]))
                                all_condition_indices_ls.append(condition_indices_lc)
                                # print(condition_indices_lc)
                                # print('****** GRID WITH COND IND ***')
                                # print(gridTest[condition_indices_lc])
                                # lc_crds = gridTest[condition_indices_lc,-1]
                                lc_coords_tmp = gridTest[condition_indices_lc,5][0]
                                # print(f"1 {gridTest[condition_indices_lc,-1]}")
                                print('GRID : ******** \n', gridTest[condition_indices_lc])
                                if(len(lc_coords_tmp) > 0):                                
                                    # print(f"2 {gridTest[condition_indices_lc,-1][0][0]}")
                                    lc_inTrk.append(gridTest[condition_indices_lc,-1][0][0])
                        print('LC_INTRK: {}'.format(lc_inTrk))
                        ##### Energy condition ###### 
                        # if condition satisfied, remove lc_inTrk from gridTest (counter-- and lc corrisponding to the energy used during check)
                        # and append tmp to the dataframe of "good" tracksters
                        checkTrk = trkIsValid(lc_inTrk, enThreshold, enThresholdCumulative)
                        # print('LC_CHECKED_TRK: {}'.format(checkTrk))
                        # print('gridTest : {}'.format(gridTest))

                        if(len(checkTrk)>0):
                            ('\n *** Energy conditions satisfied! ***')
                            # remove LCs from gridstruct (counter --)
                            for l, cond in enumerate(all_condition_indices_ls):
                                print('gridTest (pre): {}'.format(gridTest[cond]))
                                gridTest[cond,4]-=1
                                print('LC TO REMOVE: {}'.format(checkTrk[l]))                                
                                print('LC TO REMOVE: {}'.format(checkTrk[l]))
                                gridTest[cond,5][0][0].remove(checkTrk[l])
                                print('gridTest (post): {}'.format(gridTest[cond]))
        
                                #### Put into the dataframe
                                LC_forDataFrame = [i,j,k, checkTrk[l][0], checkTrk[l][1], checkTrk[l][2],checkTrk[l][3], checkTrk[l][4], numTrk]
                                allTrksFoundQuantumly.loc[len(allTrksFoundQuantumly)] = LC_forDataFrame
                            numTrk += 1 # Increase counter of num Tracksters
                            print('Tracksters found quantumly: \n', allTrksFoundQuantumly)
                        else:
                            # if condition not satisfied, dump trackster tmp
                            Tracksters_to_remove.append(tmp)
                            # print('Tracksters to remove: ', tmp)                    
                    else:
                        keep_going = False
                        print('\n *** Distance conditions NOT satisfied! No more Tracksters to be found! ***')
                    print("distances of last point: ",dist_tmp)

                #All subsequent Grover searches:
                # while keep_going:
                #     print("")
                #     print("new iteration")
                #     print("points found so far: ", len(Tracksters_to_remove))
                #     tmp = [Grover(thresholds, all_points_ordered, dataset, tracksters_to_be_removed = Tracksters_to_remove, Printing = False)]
                #     dist_tmp = f_dist_t(tmp[0], dataset, "dec")
                #     #### Valutazione trackster (considerazioni energetiche?): per ogni coppia di LC adiacenti, valuta se ABS(MAX[(E1-E2)/(E1+E2)])>EN_threshold
                #     #### ---> Scarta trackster e rimuovi da Grover (NB: se cardinalità > 1 utilizzare valore di E che minimizzi la formula sopra)
                #     #### Se ABS(MAX[(E1-E2)/(E1+E2)])<EN_threshold teniamo il trackster e rimuoviamo gli LC del trackster da all_points_ordered
                #     #### (NB: non rimuovere i missing layers = punti in cui non ci sono LC, aventi x e y massime)
                #     #### Se cardinalità > 1, non rimuovere il punto ma ridurre di 1 la cardinalità.
                #     print("distance found at this iteration:", dist_tmp)
                #     print(len(tmp))
                #     print(thresholds)
                #     if (len(tmp) != 0) and ((dist_tmp[0] < thresholds[0]) or (dist_tmp[1] < thresholds[1]) or (dist_tmp[2] < thresholds[2]) or (dist_tmp[3] < thresholds[3])):
                #         lc_inTrk = []
                #         print("Here")
                #         for lc in tmp[0]:
                #             # print(lc[0])
                #             # print(len(gridTest[0]))
                #             if(lc[0]<len(gridTest[0])): # the LC is valid (not hole)
                #                 coordinates_lc = dec_to_cart(lc,dataset)
                #                 # print('****** COORDINATES ***')
                #                 # print(coordinates_lc)
                #                 condition_indices_lc = np.where((gridTest[:,0]==coordinates_lc[0]) & (gridTest[:,1]==coordinates_lc[1]) & (gridTest[:,2]==coordinates_lc[2]))
                #                 # print(condition_indices_lc)
                #                 print('****** GRID WITH COND IND ***')
                #                 print(gridTest[condition_indices_lc])
                #                 lc_crds = gridTest[condition_indices_lc,-1]
                #                 lc_coords_tmp = gridTest[condition_indices_lc,-1][0]
                #                 # print(f"1 {gridTest[condition_indices_lc,-1]}")
                #                 # print(gridTest[condition_indices_lc,-1][0])
                #                 if(len(lc_coords_tmp) > 0):                                
                #                     # print(f"2 {gridTest[condition_indices_lc,-1][0][0]}")
                #                     lc_inTrk.append(gridTest[condition_indices_lc,-1][0][0])
                        
                #         print('LC_INTRK: {}'.format(lc_inTrk))
                #         trkIsValid(lc_inTrk, enThreshold, enThresholdCumulative)

                #         ##### HERE GOES THE ENERGY CONDITION ######
                #         #  
                #         # This will become the trackster to dump
                #         Tracksters_to_remove = Tracksters_to_remove + tmp
                #         print('Tracksters to remove: ', tmp)
                #     else:
                #         keep_going = False
                # break
            break
        break            
    #             dists_quantum = [f_dist_t(track, dataset, "dec") for track in Tracksters_to_remove]
    #             print(dists_quantum)
    # plots3DwithProjection(xs,ys,zs)