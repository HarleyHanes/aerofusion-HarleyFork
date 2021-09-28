import numpy as np
import matplotlib.pyplot as plt
from aerofusion.data import array_conversion as arr_conv
from aerofusion.numerics import curl_calc
from celluloid import Camera
import math

directory = '/scratch/dramezan/runs/Ferrante_etal_2020/low_Re/freq_25_timesteps/full_domain/'
grid_filename = 'grid_data_full_domain.npz'
aT_filename = 'modal_coeff_freq25_r50.npz'
pod_filename  = 'pod_data_low_Re_freq25_r50.npz'
u_rom_compact_filename = 'u_rom_compact_r50.npz'
velocity_rom_filename = 'velocity_rom_r50.npz'
velocity_DNS_filename = 'snapshot_data_Ferrante_low_Re_1000snap_freq25.npz'
velocity_DNS_3d_filename = 'velocity_DNS_3d.npz'
visual_directory = '/scratch/dramezan/runs/Ferrante_etal_2020/low_Re/freq_25_timesteps/full_domain/visualization/'
u_anime_name = 'u_rom_r50_1000dt_2.gif'
v_anime_name = 'v_rom_r50_1000dt_2.gif'
u_DNS_anime_name = 'u_DNS.gif'
v_DNS_anime_name = 'v_DNS.gif'

data = np.load(directory + grid_filename)

cell_volume = data['cell_volume']
num_cell = (cell_volume.shape)[0]
cell_centroid = data['cell_centroid']
cell_centroid_shape = cell_centroid.shape
##### this part is temperary -- remove for other cases
cell_center = cell_centroid[ :, 0:cell_centroid_shape[1]-1 , \
                                0:cell_centroid_shape[2]-1, :]
                             
#######################################
num_snap_visual = 999
visual_freq = 9
dim = cell_center.shape
num_xi = dim[0]
num_eta = dim[1]
num_zeta = dim[2]
xi = data['xi_index']
eta = data['eta_index']
zeta = data['zeta_index']
print('xi, eta, zeta', xi.shape, eta.shape, zeta.shape)
print('cell_volume', cell_volume.shape)
data = np.load(directory + velocity_DNS_filename)

velocity_DNS_3d = data['velocity_3D']
print('shape of 3d velocity DNS', velocity_DNS_3d.shape)
####### this part is temporary----
u_DNS = np.zeros([num_xi, num_eta, num_zeta, num_snap_visual])
v_DNS = np.zeros([num_xi, num_eta, num_zeta, num_snap_visual])
u_DNS[:,:,:,:] = velocity_DNS_3d[:, 0:num_eta, 0:num_zeta, 0, 0:num_snap_visual]
v_DNS[:,:,:,:] = velocity_DNS_3d[:, 0:num_eta, 0:num_zeta, 1, 0:num_snap_visual]
np.savez(visual_directory + velocity_DNS_3d_filename, \
u_DNS = u_DNS, v_DNS = v_DNS)

########################### 
## visualizing DNS data
num_snap_visual = num_snap_visual/visual_freq
num_snap_visual = math.floor(num_snap_visual)
print('num_snap_visual', num_snap_visual)
fig, ax = plt.subplots()
camera = Camera(fig)
for i_snap in range(num_snap_visual):
    ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
    u_DNS[:, :, 0, i_snap*visual_freq] , levels = 256,  cmap='bwr', extend = 'both')
    camera.snap()
anim = camera.animate()
anim.save(visual_directory + u_DNS_anime_name)

for i_snap in range(num_snap_visual):
    print('snap to be visualized', i_snap*visual_freq)
    ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
    v_DNS[:, :, 0, i_snap*visual_freq] , levels = 256, cmap='bwr', extend = 'both')
    camera.snap()
anim2 = camera.animate()
anim2.save(visual_directory + v_DNS_anime_name)

import ipdb
ipdb.set_trace()

data = np.load(directory + pod_filename)
v0 = data['velocity_mean']
phi = data['phi']
print('v0', v0.shape, 'phi', phi.shape)
data = np.load(directory + aT_filename)
aT = data['aT_rom']
num_snap = (aT.shape)[1]
print('aT', aT.shape)

### visualizing part of simulation
#u_mean_reduced_velocity_rom = np.matmul(phi[0:num_cell, :], \
#                                aT[:,0:num_snap_visual])
#v_mean_reduced_velocity_rom = np.matmul(phi[num_cell:2*num_cell, :],\
#                               aT[:,0:num_snap_visual])
#
#print('mean_reduced_velocity', \
#        u_mean_reduced_velocity_rom.shape, v_mean_reduced_velocity_rom.shape)
#u_rom_compact = np.zeros([num_cell, num_snap_visual])
#v_rom_compact = np.zeros([num_cell, num_snap_visual])
#for i_snap in range(num_snap_visual):
#    u_rom_compact[:,i_snap] = \
#      u_mean_reduced_velocity_rom[:,i_snap] + v0[0:num_cell]
#    v_rom_compact[:,i_snap] = \
#      v_mean_reduced_velocity_rom[:,i_snap] + v0[num_cell:2*num_cell]
#
#print('shape of velocity_rom', u_rom_compact.shape)
#np.savez( directory + u_rom_compact_filename ,\
#            u_rom_compact = u_rom_compact, v_rom_compact = v_rom_compact)
#
### velocity reconstruction 1d to 3d
#### visualizing only 2d field of xi-eta
#u_rom_3d = np.zeros([num_xi, num_eta, num_zeta, num_snap_visual])
#v_rom_3d = np.zeros([num_xi, num_eta, num_zeta, num_snap_visual])
#
#print('shape of u_rom, v_rom', u_rom_3d.shape, v_rom_3d.shape)
#
#for i_snap in range(num_snap_visual):
#    print('i_snap', i_snap)
#    u_rom_3d[:,:,:,i_snap] = arr_conv.array_1D_to_3D( \
#     xi, eta, zeta, num_cell, u_rom_compact[:, i_snap])
#    v_rom_3d[:,:,:,i_snap] = arr_conv.array_1D_to_3D( \
#     xi, eta, zeta, num_cell, v_rom_compact[:, i_snap])
#
#np.savez(visual_directory + velocity_rom_filename, u_rom_3d = u_rom_3d, v_rom_3d = v_rom_3d)

#print('loading u-rom data')
#data = np.load(visual_directory + velocity_rom_filename)
#u_rom_3d = data['u_rom_3d']
#v_rom_3d = data['v_rom_3d']
#print('shape of velocity', v_rom_3d.shape)


fig, ax = plt.subplots()
camera = Camera(fig)
for i_snap in range(num_snap_visual):
    ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
    u_rom_3d[:, :, 0, i_snap*visual_freq] , levels = 256,  cmap='bwr', extend = 'both')
    camera.snap()
anim = camera.animate()
anim.save(visual_directory + u_anime_name)

for i_snap in range(num_snap_visual):
    print('snap to be visualized', i_snap*visual_freq)
    ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
    v_rom_3d[:, :, 0, i_snap*visual_freq] , levels = 256, cmap='bwr', extend = 'both')
    camera.snap()
anim2 = camera.animate()
anim2.save(visual_directory + v_anime_name)
