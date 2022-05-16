import random
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

def plots3DwithProjection(fig, x, y, z, ranges, save = ''):
    '''
    Function to plot 3D representation with all the projections on the three planes
    - data: np.ndarray(4, num_rows). .iloc[:, :3] = coordinates, .iloc[:,3] = energy
    '''
    # fig  = plt.figure()
    ax = fig.add_subplot(2,2,1, projection = '3d')    
    # cm = plt.cm.get_cmap('cividis')
    # noise_idx_l = -1
    # C = [random.random(), ]
    for i in range(len(x)):
        ax.scatter(x[i],z[i],y[i], marker='o') #, c = id, cmap = cm)#, alpha=.3, c = data[:, 3], cmap = cm)
    ax.set_xlim(ranges[0][0]-5,ranges[0][1]+5)
    ax.set_ylim(ranges[2][0]-5, ranges[2][1]+5)
    ax.set_zlim(ranges[1][0]-5,ranges[1][1]+5)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Z$')
    ax.set_zlabel('$Y$')
        # col = ['r','g','b']
    # colors = col

    ax = fig.add_subplot(2,2,2)
    for i in range(len(x)):
        ax.scatter(x[i], y[i], marker='o')#, c = id, cmap = cm)# c = data[:,3])#, alpha = .5, cmap=cm)
        ax.plot(x[i], y[i])#, c = id, cmap = cm)# c = data[:,3])#, alpha = .5, cmap=cm)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_xlim(ranges[0][0]-5,ranges[0][1]+5)
    # ax.set_ylim(340,360)
    ax.set_ylim(ranges[1][0]-5,ranges[1][1]+5)

    # ax.set_title("X-Y")


    ax = fig.add_subplot(2,2,3)
    for i in range(len(x)):
        ax.scatter(x[i], z[i], marker='o')#, c = id, cmap = cm)#c = data[:,3])#, alpha = .5,cmap=cm)
        ax.plot(x[i], z[i])   
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Z$')
    ax.set_title("X-Z")
    ax.set_xlim(ranges[0][0]-5,ranges[0][1]+5)
    ax.set_ylim(ranges[2][0]-5,ranges[2][1]+5)
    # ax.set_ylim(-100,100)

    
    ax = fig.add_subplot(2,2,4)
    for i in range(len(x)):
        ax.scatter(y[i],z[i], marker='o')#, c = id, cmap = cm)#, c = data[:,3], alpha = .5,cmap=cm)
        ax.plot(y[i],z[i])
    ax.set_xlabel('$Y$')
    ax.set_ylabel('$Z$')
    ax.set_title("Y-Z")
    ax.set_xlim(ranges[1][0]-5,ranges[1][1]+5)
    ax.set_ylim(ranges[2][0]-5,ranges[2][1]+5)
    # ax.set_ylim(-100,100)
    # plt.plot()

    # cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(im, cax=cbar_ax, label = 'Rhits energy')
    # cbar.set_label("Rechits energy", fontsize = 50)
    # plt.savefig("./trk.png")

if __name__ == "__main__":
    
    gridThreshold = 2.5 # to define the tile size
    enThreshold = 0.6
    enThresholdCumulative = 0.5
    pThreshold = 0.99
    overlap = [2,2,2]
    
    df = pd.read_csv("/Users/tony/Desktop/qTICL/results/new/Tracksters_gTh" + str(gridThreshold) + "_pTh"  + str(pThreshold) + "_en" + str(enThreshold) + "_encm" + str(enThresholdCumulative) + "_overlap" + str(overlap[0]) + str(overlap[1]) + str(overlap[2]) +".csv")

    fig = plt.figure(figsize = (30,25))
    trk_id =  np.unique(df['TrkId'].values)
    # print('TRK_IDs: ', trk_id)
    xs = list()
    ys = list()
    zs = list()
    ranges = list()

    for id in trk_id:
        x_lcs = df[df['TrkId'] == id]['x'].values
        y_lcs = df[df['TrkId'] == id]['y'].values
        z_lcs = df[df['TrkId'] == id]['z'].values
        
        xs.append(x_lcs)
        ys.append(y_lcs)
        zs.append(z_lcs)
        
        ids = [id for i in range(len(x_lcs))]
        if(id == 0):            
            ranges = [[np.min(x_lcs), np.max(x_lcs)], [np.min(y_lcs), np.max(y_lcs)], [np.min(z_lcs), np.max(z_lcs)]]
        else:
            if(np.min(x_lcs)<ranges[0][0]):
                ranges[0][0] = np.min(x_lcs)
            if(np.max(x_lcs)>ranges[0][1]):
                ranges[0][1] = np.max(x_lcs)
            if(np.min(y_lcs)<ranges[1][0]):
                ranges[1][0] = np.min(y_lcs)
            if(np.max(y_lcs)>ranges[1][1]):
                ranges[1][1] = np.max(y_lcs)
            if(np.min(z_lcs)<ranges[2][0]):
                ranges[2][0] = np.min(z_lcs)
            if(np.max(z_lcs)>ranges[2][1]):
                ranges[2][1] = np.max(z_lcs)

    plots3DwithProjection(fig, xs, ys, zs, ranges)
    plt.savefig("/Users/tony/Desktop/qTICL/results/new/Tracksters_gTh" + str(gridThreshold) + "_pTh"  + str(pThreshold) + "_en" + str(enThreshold) + "_encm" + str(enThresholdCumulative) + "_overlap" + str(overlap[0]) + str(overlap[1]) + str(overlap[2]) +".png")
