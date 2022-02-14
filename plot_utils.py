import random
import matplotlib.pyplot as plt 

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