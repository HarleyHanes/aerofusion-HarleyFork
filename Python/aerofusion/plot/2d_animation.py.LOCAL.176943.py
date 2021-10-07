import numpy as np
import matplotlib.pyplot as plt
from aerofusion.data import array_conversion as arr_conv
from aerofusion.numerics import curl_calc
from celluloid import Camera
import math

directory = '/scratch/dramezan/runs/Ferrante_etal_2020/low_Re/freq_25_timesteps/full_domain/r_50/'
visual_directory = '/scratch/dramezan/runs/Ferrante_etal_2020/low_Re/freq_25_timesteps/full_domain/visualization/'
grid_filename = 'grid_data_full_domain.npz'
aT_filename = 'modal_coeff_freq25_r50.npz'
pod_filename  = 'pod_data_low_Re_freq25_r50.npz'
velocity_rom_compact_filename = 'u_rom_compact_r50.npz'
velocity_rom_filename = 'velocity_rom_r50.npz'
velocity_LES_filename = 'snapshot_data_Ferrante_low_Re_1000snap_freq25.npz'
velocity_LES_3d_filename = 'velocity_LES_3d.npz'
velocity_rom_anime_name = 'u_v_rom.gif'
velocity_LES_anime_name = 'u_LES.gif'
output_png_filename = 'freq_25_energy_90_t_'

num_snap_visual = 1000
visual_freq = 10
num_snap_visual = num_snap_visual/visual_freq
num_snap_visual = math.floor(num_snap_visual)

###-------------------------------------------
print('loading grid data')
data = np.load(directory + grid_filename)
cell_volume = data['cell_volume']
num_cell = (cell_volume.shape)[0]
cell_centroid = data['cell_centroid']
cell_centroid_shape = cell_centroid.shape
##### this part is temperary -- remove for other cases
cell_center = cell_centroid[ :, 0:cell_centroid_shape[1]-1 , \
                                0:cell_centroid_shape[2]-1, :]
                             
#######################################
dim = cell_center.shape
num_xi = dim[0]
num_eta = dim[1]
num_zeta = dim[2]
xi = data['xi_index']
eta = data['eta_index']
zeta = data['zeta_index']
print('xi, eta, zeta', xi.shape, eta.shape, zeta.shape)
print('cell_volume', cell_volume.shape)

###-------------------------------------------------
print('loading pod data')
data = np.load(directory + pod_filename)
for key in data.keys():
    print('keys in pod', key)

aT_pod = data['modal_coeff']
lambda_pod = data['pod_lambda']
###-----visualizing lambda
#print('shape of modal coeff and lambda', aT_pod.shape, lambda_pod.shape)
#ratio =  (np.sum(lambda_pod[0:49]))/np.sum(lambda_pod)
#print('ratio', ratio)
#plt.plot(np.log(lambda_pod[0:999]/lambda_pod[0]), '*')
#plt.axvline(x = 50, color = 'r')
#plt.xlabel('number of modes')
#plt.ylabel('ln(lambda/lambda_0)')
#plt.savefig(visual_directory + 'lambda.png')
#plt.clf()
###-------------------------------
v0 = data['velocity_mean']
phi = data['phi']
print('v0', v0.shape, 'phi', phi.shape)
data = np.load(directory + aT_filename)
aT_rom = data['aT_rom']
num_snap = (aT_rom.shape)[1]
print('aT_rom, aT_pod', aT_rom.shape, aT_pod.shape)
#
####----------------visualizing modal coefficient 
#plt.plot(aT_rom[0, :], 'o', label='pod-rom')
#plt.plot(aT_pod[0, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_0.png')
#plt.clf()
#
#plt.plot(aT_rom[1, :], 'o', label='pod-rom')
#plt.plot(aT_pod[1, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_1.png')
#plt.clf()
#
#plt.plot(aT_rom[2, :], 'o', label='pod-rom')
#plt.plot(aT_pod[2, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_2.png')
#plt.clf()
#
#plt.plot(aT_rom[3, :], 'o', label='pod-rom')
#plt.plot(aT_pod[3, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_3.png')
#plt.clf()
#
#plt.plot(aT_rom[10, :], 'o', label='pod-rom')
#plt.plot(aT_pod[10, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_10.png')
#plt.clf()
#
#plt.plot(aT_rom[15, :], 'o', label='pod-rom')
#plt.plot(aT_pod[15, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_15.png')
#plt.clf()
#
#plt.plot(aT_rom[20, :], 'o', label='pod-rom')
#plt.plot(aT_pod[20, :], '*', label='pod')
#plt.legend()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficient')
#plt.savefig(visual_directory + 'mode_20.png')
#plt.clf()
#
#import ipdb
#ipdb.set_trace()
#

###-----------visualizing modes

mode_1_3d = np.zeros([num_xi, num_eta, num_zeta])
mode_2_3d = np.zeros([num_xi, num_eta, num_zeta])
mode_3_3d = np.zeros([num_xi, num_eta, num_zeta])
mode_10_3d = np.zeros([num_xi, num_eta, num_zeta])
mode_20_3d = np.zeros([num_xi, num_eta, num_zeta])

mode_1_3d[:,:,:] = arr_conv.array_1D_to_3D( \
     xi, eta, zeta, num_cell, phi[:, 0])
mode_2_3d[:,:,:] = arr_conv.array_1D_to_3D( \
     xi, eta, zeta, num_cell, phi[:, 1])
mode_3_3d[:,:,:] = arr_conv.array_1D_to_3D( \
     xi, eta, zeta, num_cell, phi[:, 2])
mode_10_3d[:,:,:] = arr_conv.array_1D_to_3D( \
     xi, eta, zeta, num_cell, phi[:, 9])
mode_20_3d[:,:,:] = arr_conv.array_1D_to_3D( \
     xi, eta, zeta, num_cell, phi[:, 19])

print('visualizing rom data')
num_levels = 256
vmin = - 0.012
vmax = 0.012
levels = np.linspace(vmin, vmax, num_levels)

fig, ax = plt.subplots()     
pl = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
     mode_1_3d[:, :, 0] ,  levels = num_levels, cmap='jet', extend = 'both')
fig.colorbar(pl)
plt.savefig(visual_directory + 'phi_1.png')
plt.clf()

fig, ax = plt.subplots()     
pl = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
     mode_2_3d[:, :, 0] ,  levels = num_levels, cmap='jet', extend = 'both')
fig.colorbar(pl)
plt.savefig(visual_directory + 'phi_2.png')
plt.clf()

fig, ax = plt.subplots()     
pl = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
     mode_3_3d[:, :, 0] ,  levels = num_levels, cmap='jet', extend = 'both')
fig.colorbar(pl)
plt.savefig(visual_directory + 'phi_3.png')
plt.clf()

fig, ax = plt.subplots()     
pl = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
     mode_10_3d[:, :, 0] ,  levels = num_levels, cmap='jet', extend = 'both')
fig.colorbar(pl)
plt.savefig(visual_directory + 'phi_10.png')
plt.clf()

fig, ax = plt.subplots()     
pl = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
     mode_20_3d[:, :, 0] ,  levels = num_levels, cmap='jet', extend = 'both')
fig.colorbar(pl)
plt.savefig(visual_directory + 'phi_20.png')
plt.clf()




import ipdb
ipdb.set_trace()

####----------calculating v-rom field from calculated aT and phi
#u_mean_reduced_velocity_rom = np.matmul(phi[0:num_cell, :], \
#                                aT[:,::visual_freq])
#v_mean_reduced_velocity_rom = np.matmul(phi[num_cell:2*num_cell, :],\
#                               aT[:,::visual_freq])
#print('shape of mean reduced velocity', u_mean_reduced_velocity_rom.shape)
#u_rom_compact = np.zeros([num_cell, num_snap_visual])
#v_rom_compact = np.zeros([num_cell, num_snap_visual])
#for i_snap in range(num_snap_visual):
#    u_rom_compact[:,i_snap] = \
#      u_mean_reduced_velocity_rom[:,i_snap] + v0[0:num_cell]
#    v_rom_compact[:,i_snap] = \
#      v_mean_reduced_velocity_rom[:,i_snap] + v0[num_cell:2*num_cell]
#
#print('shape of velocity_rom', u_rom_compact.shape)
#np.savez( visual_directory + velocity_rom_compact_filename ,\
#            u_rom_compact = u_rom_compact, v_rom_compact = v_rom_compact)
#
#####-----------------------------------------------------------
#### if you have calculated rom field and only need the data
#data = np.load(visual_directory + velocity_rom_compact_filename)
#u_rom_compact = data['u_rom_compact']
#v_rom_compact = data['v_rom_compact']
#print('reconstructing compact data to 3d')
#u_rom_3d = np.zeros([num_xi, num_eta, num_zeta, num_snap_visual])
#v_rom_3d = np.zeros([num_xi, num_eta, num_zeta, num_snap_visual])
#for i_snap in range(num_snap_visual):
#    u_rom_3d[:,:,:,i_snap] = arr_conv.array_1D_to_3D( \
#     xi, eta, zeta, num_cell, u_rom_compact[:, i_snap])
#    v_rom_3d[:,:,:,i_snap] = arr_conv.array_1D_to_3D( \
#     xi, eta, zeta, num_cell, v_rom_compact[:, i_snap])
#
#np.savez(visual_directory + velocity_rom_filename, \
#          u_rom_3d = u_rom_3d, v_rom_3d = v_rom_3d)
#####-----------------------------------------------------------------------
#### if you have calculated 3d-rom fields
#data = np.load(visual_directory + velocity_rom_filename)
#u_rom_3d = data['u_rom_3d']
#v_rom_3d = data['v_rom_3d']
#print('shape of u_rom', u_rom_3d.shape)
#print('shape of cell center', cell_center.shape)
####------------------------------------------------------------------
### reconstructing field with only one mode at time = 0

#u_t0_mode0 = np.matmul(phi[0:num_cell, 0], \
#                                aT_rom[0,0])
#v_t0_mode0 = np.matmul(phi[num_cell:2*num_cell, 0],\
#                               aT_rom[0,0])
u_t0_mode0 = phi[0:num_cell, 2] * aT_rom[2,0]
v_t0_mode0 = phi[num_cell:2*num_cell, 2] *  aT_rom[2,0]
print('reconstructing compact data to 3d')
u_t0_mode0_3d = np.zeros([num_xi, num_eta, num_zeta])
v_t0_mode0_3d = np.zeros([num_xi, num_eta, num_zeta])
u_t0_mode0_3d[:,:,:] = arr_conv.array_1D_to_3D( \
                              xi, eta, zeta, num_cell, u_t0_mode0)
v_t0_mode0_3d[:,:,:] = arr_conv.array_1D_to_3D( \
                              xi, eta, zeta, num_cell, v_t0_mode0)


####visualizing mode recustructed by one mode


print('visualizing rom data')
num_levels = 256
u_vmin = - 0.012
u_vmax = 0.012
v_vmin = -0.006
v_vmax = 0.006 
u_levels = np.linspace(u_vmin, u_vmax, num_levels)
v_levels = np.linspace(v_vmin, v_vmax, num_levels)

fig, ax = plt.subplots()     
uplot = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
 u_t0_mode0_3d[:, :, 0] ,  levels = u_levels, cmap='jet', extend = 'both')
#fig.colorbar(uplot)#, ax = ax1)
plt.savefig(visual_directory + 'u_mode_num3_field.png')
plt.clf()

fig, ax = plt.subplots()     
uplot = ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
 v_t0_mode0_3d[:, :, 0] , levels = v_levels, cmap='jet', extend = 'both')
#fig.colorbar(uplot)#, ax = ax1)
plt.savefig(visual_directory + 'v_mode_num3_field.png')

import ipdb
ipdb.set_trace()
###--------------------------------------------------------
#print('loading LES data for the preparation')
#data = np.load(directory + velocity_LES_filename)
#velocity_LES_3d = data['velocity_3D']
#print('shape of velocity_LES', velocity_LES_3d.shape)
#
#u_LES = np.zeros([num_xi, num_eta, num_snap_visual])
#v_LES = np.zeros([num_xi, num_eta, num_snap_visual])
#u_LES[:,:,:] = velocity_LES_3d[:, 0:num_eta, 0, 0, 1::visual_freq]
#v_LES[:,:,:] = velocity_LES_3d[:, 0:num_eta, 0, 1, 1::visual_freq]
#np.savez(visual_directory + velocity_LES_3d_filename, \
#u_LES = u_LES, v_LES = v_LES)
#
#print('shape of v_LES, u_LES', u_LES.shape, v_LES.shape)
#import ipdb
#ipdb.set_trace()
###-----------------------------------------------
## loading calculated LES velocity field
data = np.load(visual_directory + velocity_LES_3d_filename)
u_LES = data['u_LES']
v_LES = data['v_LES']

###--------------------------------------------------
print('creating field of difference between rom and LES')
u_diff = np.zeros([num_xi, num_eta, num_snap_visual])
v_diff = np.zeros([num_xi, num_eta, num_snap_visual])
u_diff[:,:,:] = u_LES[:,:,:] - u_rom_3d[:,:,0,:]
v_diff[:,:,:] = v_LES[:,:,:] - v_rom_3d[:,:,0,:]

####-------------------------------------------
print('visualizing LES-rom-diff data')
num_levels = 256
u_vmin = - 0.2
u_vmax = 1.7
v_vmin = -0.5
v_vmax = 0.5 
u_levels = np.linspace(u_vmin, u_vmax, num_levels)
v_levels = np.linspace(v_vmin, v_vmax, num_levels)
u_diff_min = -0.05 #np.min(u_diff)
u_diff_max = 0.05# np.max(u_diff)
v_diff_min = -0.05 #np.min(v_diff)
v_diff_max = 0.05 #np.max(v_diff)
udiff_levels = np.linspace(u_diff_min, u_diff_max, num_levels)
vdiff_levels = np.linspace(v_diff_min, v_diff_max, num_levels)
print('u_min u_max', u_diff_min, u_diff_max)
print('v_min v_max', v_diff_min, v_diff_max)


fig, axes = plt.subplots(2,3)     
camera = Camera(fig)                  
for i_snap in range(num_snap_visual):
    print('snap to be visualized', i_snap*visual_freq)
    axes[0,0].contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
      u_LES[:, :, i_snap] , levels = u_levels,  cmap='jet', extend = 'both')
    axes[0,0].set_title('LES')

    axes[0,1].contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
      u_rom_3d[:, :, 0, i_snap] , levels = u_levels,  cmap='jet', extend = 'both')
    axes[0,1].set_title('ROM')
   
    axes[0,2].contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
      u_diff[:, :, i_snap] , levels = udiff_levels,  cmap='jet', extend = 'both')
    axes[0,2].set_title('difference')
    
    axes[1,0].contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
      v_LES[:, :, i_snap] , levels = v_levels,  cmap='jet', extend = 'both')
    # axes[1,0].set_title('v-LES')

    axes[1,1].contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
      v_rom_3d[:, :, 0, i_snap] , levels = v_levels,  cmap='jet', extend = 'both')
    #axes[1,1].set_title('v-rom')
   
    axes[1,2].contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
      v_diff[:, :, i_snap] , levels = vdiff_levels,  cmap='jet', extend = 'both')
    #axes[1,2].set_title('v-diff')
    camera.snap()
anim = camera.animate()
anim.save(visual_directory + 'LES-ROM-freq25-r50-final.gif')

import ipdb
ipdb.set_trace()

                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
                                            
