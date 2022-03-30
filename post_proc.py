###############################################################
###############################################################
### **** POST-PROCESSING FUNCTIONS FOR TRACKSTER MERGE **** ###
###############################################################
###############################################################


import numpy as np
import pandas as pd
import math

from plot_utils import *
import matplotlib.pyplot as plt
from grover_func import *

#'x', 'y', 'z', 'layer', 'energy', 'LCID', 'TrkId'

def trkIsValid(lcInTrackster, energyThrs, energyThrsCumulative, pThrs):
    lcEnergy = [[lc_energy[4] for lc_energy in lc] for lc in lcInTrackster]
    energyIndices = [[i for i in range(len(energies))] for energies in lcEnergy]

    allPaths = list(itertools.product(*energyIndices))
    lcXYZ = [[lcInfo[:3] for lcInfo in lc] for lc in lcInTrackster]

    totEnDiff = []
    totPVal = []
    for path in allPaths:
        enDiff = 0
        points = [lcXYZ[i][k] for i,k in enumerate(path)]
        energies = [lcEnergy[i][k] for i,k in enumerate(path)]
        pval = pval_fit(points, energies)
        if(pval > pThrs):
            pval = float('inf')
            # pval = 1
        for i,k in enumerate(path):
            if(i>0):
                enContr = np.abs((lcEnergy[i][k] - curr)/(lcEnergy[i][k] + curr))
                if(enContr < energyThrs):
                    enDiff += enContr
                else:
                    enDiff += float('inf')
            curr = lcEnergy[i][k]
        if(enDiff > energyThrsCumulative*len(energies)): # we multiply the energyThrsCumulative by the number of LCs
            totEnDiff.append(float('inf'))
        else:
            totEnDiff.append(enDiff/len(path))
        totPVal.append(pval)
    
    totDiff = [np.sqrt(totEnDiff[i] * totPVal[i]) for i in range(len(allPaths))]
    minTotDiff = np.min(totDiff)
    argMinTotDiff = np.where(totDiff == minTotDiff)[0][0]
    minIndices = allPaths[argMinTotDiff] 

    if(np.isinf(minTotDiff)):
        return []
    else:
        return [lcInTrackster[i][minIndices[i]] for i in range(len(minIndices))]


def closestDistanceBetweenLines(line1, line2, clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distances XY and Z
    '''

    a0 = line1[0]
    a1 = line1[1]
    b0 = line2[0]
    b1 = line2[1]

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                    #     return a0,b0,np.linalg.norm(a0-b0), 0.0
                        distXY = np.sqrt((a0[0]-b0[0])**2 + (a0[1]-b0[1])**2)
                        distZ = np.abs(a0[2]-b0[2])
                        return a0,b0,distXY,distZ                
                    # return a0,b1,np.linalg.norm(a0-b1)
                    distXY = np.sqrt((a0[0]-b1[0])**2 + (a0[1]-b1[1])**2)
                    distZ = np.abs(a0[2]-b1[2])
                    return a0,b1,distXY,distZ                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                    #     return a1,b0,np.linalg.norm(a1-b0)
                        distXY = np.sqrt((a1[0]-b0[0])**2 + (a1[1]-b0[1])**2)
                        distZ = np.abs(a1[2]-b0[2])
                        return a0,b0,distXY,distZ                
                    # return a1,b1,np.linalg.norm(a1-b1)
                    distXY = np.sqrt((a1[0]-b1[0])**2 + (a1[1]-b1[1])**2)
                    distZ = np.abs(a1[2]-b1[2])
                    return a1,b1,distXY,distZ                
                
                
        # Segments overlap, return distance between parallel segments
        # return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        distXY = np.linalg.norm(((d0*_A)+a0)-b0)
        distZ = 0.0
        return None,None,distXY,distZ
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    distXY = np.sqrt((pA[0]-pB[0])**2 + (pA[1]-pB[1])**2)
    distZ = np.abs(pA[2]-pB[2])
    # return pA,pB,np.linalg.norm(pA-pB)
    return pA,pB,distXY,distZ


def findDuplicates(dataset, lc_id):
    dup = dataset[dataset['LCID']==lc_id]
    if(len(dup)==1):
        return dup['TrkId'].values, list()
    else:
        trk_ids = dup['TrkId'].values
        trks = dataset[dataset['TrkId'].isin(trk_ids)]
        return trk_ids, trks


def compatAndFit(dataset, trk_id1, trk_id2, dist, ang, pThrs, energyThrs, energyThrsCumulative, zTolerance):
    trk1 = dataset[dataset['TrkId']==trk_id1]
    trk2 = dataset[dataset['TrkId']==trk_id2]

    distXY = dist[0]
    distZ = dist[1]

    lcs1_XYZ = trk1.loc[:,['x', 'y', 'z']].to_numpy() 
    lcs1_en = trk1.loc[:,'energy'].to_numpy()
    lcs2_XYZ = trk2.loc[:,['x', 'y', 'z']].to_numpy()
    lcs2_en = trk2.loc[:,'energy'].to_numpy()

    linepts1, eigenvector1 = line_fit(lcs1_XYZ, lcs1_en)
    linepts2, eigenvector2 = line_fit(lcs2_XYZ, lcs2_en)

    alignment = np.arccos(np.dot(eigenvector1, eigenvector2))
    if(alignment < 1.e-7):
        _, _, distanceXY, distanceZ = closestDistanceBetweenLines(linepts1, linepts2, clampAll=True)
    else:
        _, _, distanceXY, distanceZ = closestDistanceBetweenLines(linepts1, linepts2, clampAll=False)

    if (alignment > ang or distanceXY > distXY or distanceZ > distZ):
        return False
    else:
        lcsTot = np.concatenate((lcs1_XYZ, lcs2_XYZ))
        _, idx = np.unique(lcsTot, axis=0, return_index=True)
        lcsTot = lcsTot[np.sort(idx)]
        lcsTotEn = np.concatenate((lcs1_en, lcs2_en))
        _, idxEn = np.unique(lcsTotEn, axis=0, return_index=True)
        lcsTotEn = lcsTotEn[np.sort(idxEn)]

        # calculate pval
        pval = pval_fit(lcsTot, lcsTotEn)
        if(pval > pThrs):
            return False
        else:
            ###### Sort energies in Z, sum energies of LCs on the same layer/zvalue and apply
            ###### energy difference criteria
            enDiff = 0
            allZ = lcsTot[:,2]
            indices = np.argsort(allZ,)
            sorted_AllZ = np.flip(allZ[indices[::-1]]).tolist()
            # print(sorted_AllZ)
            sorted_En = np.flip(lcsTotEn[indices[::-1]]).tolist()
            i = 0
            while(i<len(sorted_AllZ) - 1):
                if(np.abs(sorted_AllZ[i]-sorted_AllZ[i+1])<zTolerance):
                    sorted_AllZ.pop(i+1)
                    sorted_En[i] += sorted_En[i+1]
                    sorted_En.pop(i+1)
                else:
                    i += 1

            for i in range(len(sorted_En)):
                if(i>0):
                    enContr = np.abs((sorted_En[i] - curr)/(sorted_En[i] + curr))
                    if(enContr < energyThrs):
                        enDiff += enContr
                    else:
                        return False
                curr = sorted_En[i]
            if(enDiff > energyThrsCumulative*len(sorted_En)): # we multiply the energyThrsCumulative by the number of LCs
                return False            
    return True


def mergedTrkIsValid(dataset, trkId, energyThrs, energyThrsCumulative, pThrs, zTolerance):
    trk = dataset[dataset['TrkId']==trkId].sort_values(by=['layer'])

    lcs_XYZ = trk.loc[:,['x', 'y', 'z']].to_numpy() 
    lcs_en = trk.loc[:,'energy'].to_numpy()

    # calculate pval
    pval = pval_fit(lcs_XYZ, lcs_en)
    if(pval > pThrs):
        return False
    else:
        ###### Sort energies in Z, sum energies of LCs on the same layer/zvalue and apply
        ###### energy difference criteria
        enDiff = 0
        allZ = lcs_XYZ[:,2]
        indices = np.argsort(allZ,)
        sorted_AllZ = np.flip(allZ[indices[::-1]]).tolist()
        sorted_En = np.flip(lcs_en[indices[::-1]]).tolist()
        i = 0
        while(i<len(sorted_AllZ) - 1):
            if(np.abs(sorted_AllZ[i]-sorted_AllZ[i+1])<zTolerance):
                sorted_AllZ.pop(i+1)
                sorted_En[i] += sorted_En[i+1]
                sorted_En.pop(i+1)
            else:
                i += 1

        for i in range(len(sorted_En)):
            if(i>0):
                enContr = np.abs((sorted_En[i] - curr)/(sorted_En[i] + curr))
                if(enContr < energyThrs):
                    enDiff += enContr
                else:
                    return False
            curr = sorted_En[i]
        if(enDiff > energyThrsCumulative*len(sorted_En)): # we multiply the energyThrsCumulative by the number of LCs
            return False            
    return True


def mergeTrkDup(dataset, dist, ang, pThrs, energyThrs, energyThrsCumulative, zTolerance):
    lc_ids = np.unique(dataset['LCID'].values)
    for i in lc_ids:
        dupsId, dupsTrk = findDuplicates(dataset, i) 
        old_dupsId = dupsId
        if(len(dupsId)>1):
            energiesTrk = [np.sum(dupsTrk[dupsTrk['TrkId']==j]['energy'].values) for j in dupsId]
            energiesTrk_idx = np.argsort(energiesTrk)[::-1] #order of the indices of the energies from highest to lowest
            keep_merging = True
            en_index = 0
            while keep_merging:
                indices = np.delete(energiesTrk_idx, en_index)
                for j in indices:
                    trk_merge = compatAndFit(dupsTrk, dupsId[energiesTrk_idx[en_index]], dupsId[j], dist, ang, pThrs, energyThrs, energyThrsCumulative, zTolerance)
                    if(trk_merge):
                        ##### merge tracksters (modify dupsId and dupsTrk) and redefine energiesTrk_idx and energiesTrk
                        dupsTrk.loc[dupsTrk.TrkId==dupsId[j], 'TrkId'] = dupsId[energiesTrk_idx[en_index]] # change the id of the other trackster with the id of the most energetic 
                        dupsTrk.drop_duplicates() # remove duplicate (LC) rows  
                        dupsId = np.delete(dupsId, j) #--> remove the index from the list of the indices 

                        # redefine these
                        energiesTrk = [np.sum(dupsTrk[dupsTrk['TrkId']==j]['energy'].values) for j in dupsId]
                        energiesTrk_idx = np.argsort(energiesTrk)[::-1] #order of the indices of the energies from highest to lowest
                        en_index = 0
                        break

                        ###### CHANGE TRKID IN DATASET NOT DUPSTRK WHEN MERGING!!! (Tested and working)
                        # dataset.loc[dataset.TrkId==dupsId[j], 'TrkId'] = dupsId[energiesTrk_idx[en_index]] #this is necessary, but we can avoid to have dupsTrk in return from findDuplicates

                    if(not trk_merge):
                        en_index += 1
                if(en_index==len(energiesTrk_idx)):
                    keep_merging = False

            # If some Tracksters were not merged, assign energy fraction to duplicate LC with index i
            dupLC = dupsTrk[dupsTrk['LCID']==i]
            numDup = len(dupLC)
            enDup = numDup['energy'].values
            dupsTrk.loc[dupsTrk.LCID==i, 'energy'] = enDup/numDup
            
            existsFalse = True
            while existsFalse:
                invalidTrks = list()
                for trkId in dupsId:
                    if(not mergedTrkIsValid(dupsTrk, trkId, energyThrs, energyThrsCumulative, pThrs, zTolerance)):
                        invalidTrks.append(trkId)

                ###### if exist invalid tracksters, remove common LC from least energetic and recompute energy of the other trackster
                ###### if LC is removed check if its trackster is valid; if not, kill it --- remove from dataset
                ####### -> NB: this part is correct but we need to distribute the energy of common LC to the tracksters weighing it with the local energy density

                if(len(invalidTrks)==0):
                    existsFalse = False

            #### Update dataset with dupsTrk (removing the older tracksters using old_dupsId)



    # find duplicates
    # check compatibility
    # if compatibily == True, fit the final trackster and apply energy criteria
    return True












def mergeTrkAll(dataset, dist, ang, pThrs, energyThrs, energyThrsCumulative, zTolerance):
    # check compatibility
    # if compatibily == True, fit the final trackster and apply energy criteria


    ### MERGE IDEA
    ### Find cover of Grover boxes and consider only tracksters within (we don't want to try to merge tracksters too distant)
    ### If trackster is merged, expand the cover as required
    return True



if __name__=='__main__':
    df = pd.read_csv('trackstersGrover_gTh2.0_pTh0.99.csv')
    trk_id1 = 0
    trk_id2 = 1
    dist = [10,10]
    ang = np.pi
    pThrs = 1
    energyThrs = 1
    energyThrsCumulative = 0.8
    zTolerance = 0.5

    # res = compatAndFit(df, trk_id1, trk_id2, dist, ang, pThrs, energyThrs, energyThrsCumulative, zTolerance)
    # print(res)

    lcid = 379.0
    trkids, trks = findDuplicates(df, lcid)
    enDup = trks['energy'].values
    enDup = enDup[:3]
    # print(enDup/enDup)
    trks.loc[trks.LCID==379,'energy'] = enDup/3
    print(trks)
    trkToCheck = trks[trks['TrkId']==1][['x', 'y', 'z', 'layer', 'energy', 'LCID']].to_numpy()
    trk = [k for k in trkToCheck]
    print(trk[0])
    # trkToCheck = trkToCheck.to_numpy()
    # print(trkToCheck)

'''
    old_dupsId = trkids
    # print(trkids)
    print(trks)
    energiesTrk = [np.sum(df[df['TrkId']==j]['energy'].values) for j in trkids]
    energiesTrk_idx = np.argsort(energiesTrk)[::-1] #order of the indices of the energies from highest to lowest
    # print(energiesTrk)
    # print(energiesTrk_idx)
    en_index = 0
    # print(energiesTrk_idx[en_index])
    # tmp_Trk = trks[trks['TrkId'] == trkids[energiesTrk_idx[en_index]]]
    # print(tmp_Trk)
    print(energiesTrk_idx)
    indices = np.delete(energiesTrk_idx, en_index)
    print(indices)
    print(energiesTrk_idx)
    # print('\n')
    print(df[df.TrkId==trkids[energiesTrk_idx[en_index]]])
    for j in indices:
        print('\n')
        print(j)
        trk_merge = compatAndFit(df, trkids[energiesTrk_idx[en_index]], trkids[j], dist, ang, pThrs, energyThrs, energyThrsCumulative, zTolerance)
        if(trk_merge):
            df.loc[df.TrkId==trkids[j], 'TrkId'] = trkids[energiesTrk_idx[en_index]]   
            trkids = np.delete(trkids, j)
    print(trkids)
    print(old_dupsId)
    # print(len(trkids))
'''