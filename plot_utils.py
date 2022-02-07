import matplotlib.pyplot as plt 
def plots3DwithProjection(x,y,z, save = ''):
    '''
    Function to plot 3D representation with all the projections on the three planes
    - data: np.ndarray(4, num_rows). .iloc[:, :3] = coordinates, .iloc[:,3] = energy
    '''
    fig = plt.figure(figsize = (30,25))
    # fig  = plt.figure()
    ax = fig.add_subplot(2,2,1, projection = '3d')    
    cm = plt.cm.get_cmap('hot')
    noise_idx_l = -1
    ax.scatter(x,z,y, marker='o')#, alpha=.3, c = data[:, 3], cmap = cm)
    ax.set_xlim(-100,100)
    ax.set_ylim(340,360)
    ax.set_zlim(-100,100)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    # col = ['r','g','b']
    # colors = col
    ax = fig.add_subplot(2,2,2)
    ax.scatter(x, y)# c = data[:,3])#, alpha = .5, cmap=cm)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_xlim(-100,100)
    # ax.set_ylim(340,360)
    ax.set_ylim(-100,100)

    # ax.set_title("X-Y")


    ax = fig.add_subplot(2,2,3)
    ax.scatter(x, z)#c = data[:,3])#, alpha = .5,cmap=cm)

    ax.set_xlabel('$X$')
    ax.set_ylabel('$Z$')
    ax.set_title("X-Z")
    ax.set_xlim(-100,100)
    ax.set_ylim(340,360)
    # ax.set_ylim(-100,100)

    
    ax = fig.add_subplot(2,2,4)
    im = ax.scatter(y,z)#, c = data[:,3], alpha = .5,cmap=cm)

    ax.set_xlabel('$Y$')
    ax.set_ylabel('$Z$')
    ax.set_title("Y-Z")
    ax.set_xlim(-100,100)
    ax.set_ylim(340,360)
    # ax.set_ylim(-100,100)
    # plt.plot()

    # cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(im, cax=cbar_ax, label = 'Rhits energy')
    # cbar.set_label("Rechits energy", fontsize = 50)
    plt.savefig("./trk.png")