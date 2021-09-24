import numpy as np
import matplotlib.pyplot as plt
from aerofusion.data import array_conversion as arr_conv
from celluloid import Camera

directory = '/scratch/dramezan/runs/Ferrante_etal_2020/low_Re/freq_25_timesteps/full_domain/'
grid_filename = 'grid_data_full_domain.npz'
aT_filename = 'modal_coeff_freq25_r50.npz'
pod_filename  = 'pod_data_low_Re_freq25_r50.npz'
u_rom_filename = 'u_rom_r50.npz'
anime_name = 'u_rom_r50.gif'

data = np.load(directory + grid_filename)
#for key in data.keys() :
#    print('keys in grid-data', key)

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

data = np.load(directory + pod_filename)
#for key in data.keys() :
#    print('keys in pod-data', key)

v0 = data['velocity_mean']
phi = data['phi']
print('v0', v0.shape, 'phi', phi.shape)
data = np.load(directory + aT_filename)
#for key in data.keys() :
#    print('keys in aT-data', key)

aT = data['aT_rom']
num_snap = (aT.shape)[1]
print('aT', aT.shape)

### visualizing only u component of velociry
mean_reduced_velocity_rom = np.matmul(phi[0:num_cell, :], aT)

print('mean_reduced_velocity', mean_reduced_velocity_rom.shape)


u_rom_compact = np.zeros([num_cell, num_snap])
for i_snap in range(num_snap):
    u_rom_compact[:,i_snap] = \
      mean_reduced_velocity_rom[:,i_snap] + v0[0:num_cell]

print('shape of velocity_rom', u_rom_compact.shape)
## velocity reconstruction 1d to 3d
u_rom_3d = np.zeros([num_xi, num_eta, num_zeta, num_snap])
print('shape of u_rom', u_rom_3d.shape)
for i_snap in range(num_snap):
    print('i_snap', i_snap)
    u_rom_3d[:,:,:,i_snap] = arr_conv.array_1D_to_3D( \
     xi, eta, zeta, num_cell, u_rom_compact[:, i_snap])

np.savez(directory + u_rom_filename, u_rom_r50 = u_rom_3d)

print('loading u-rom data')
u_rom_3d = np.load(directory + u_rom_filename)

fig, ax = plt.subplots()
camera = Camera(fig)
n_levels = 256
vmin = -0.2
vmax = 0.2
levels = np.linspace(vmin, vmax, n_levels)
for i_snap in range(num_snap):
    ax.contourf(cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], \
    u_rom_3d[:, :, 0, i_snap] , levels=levels, cmap='bwr', extend = 'both')
    camera.snap()
anim = camera.animate()
anim.save(directory + anime_name)

