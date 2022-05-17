# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:07:31 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt

save_path = "../../lid_driven_data/morris_mesh_grid.png"
data_folder = "../../lid_driven_data/"
weights_file = "weights_hr.mat"
velocity_file = "re17000_hr.mat"

num_dim  = 2
num_xi   = 258
num_eta  = 258
num_zeta = 1

centroid_file=np.load(data_folder + "cell_center_high_res.npz")
cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
num_cell = num_xi*num_eta*num_zeta

#plt.scatter(cell_centroid[:,:,0,0], cell_centroid[:,:,0,1],s=.001)
#plt.show()
plt.figure(figsize = (7,7))
for i_xi in np.arange(0,258,4):
    x1 = [cell_centroid[i_xi,0,0,0], cell_centroid[i_xi,-1,0,0]]
    x2 = [cell_centroid[i_xi,0,0,1], cell_centroid[i_xi,-1,0,1]]
    plt.plot(x1,x2, 'b-', linewidth=.3, alpha =1)
for i_eta in np.arange(0,258,4):
    x1 = [cell_centroid[0,i_eta,0,0], cell_centroid[-1,i_eta,0,0]]
    x2 = [cell_centroid[0,i_eta,0,1], cell_centroid[-1,i_eta,0,1]]
    plt.plot(x1,x2, 'b-', linewidth=.3, alpha =1)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(save_path)