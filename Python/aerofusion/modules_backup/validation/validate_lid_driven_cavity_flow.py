import numpy as np
import Derivative_Calc
import structure_format
import pod_modes
import curl_calc
import matplotlib.pyplot as plt
from numpy.linalg import norm

data_ref = \
  np.load('/scratch/dramezan/runs/Lid_Driven_Cavity_Data/pod_cavity_data_re25000_r50.npz')
data_matrices_ref = \
  np.load('/scratch/dramezan/runs/Lid_Driven_Cavity_Data/pod_cavity_data_re25000.npz')
python_pod_data_dir = \
  '/scratch/dramezan/runs/Lid_Driven_Cavity_Data/python_pod_data/'
Nxi = 258
Neta = 258
num_cell = Nxi * Neta
L0 = data_matrices_ref['L0']
LRe = data_matrices_ref['LRe']
C0 = data_matrices_ref['C0']
CRe = data_matrices_ref['CRe']
Q = data_matrices_ref['Q']
dvel_dx = data_ref['dvel_dx']
dvel_dy = data_ref['dvel_dy']
dphi_dx = data_ref['dphi_dx']
dphi_dy = data_ref['dphi_dy']
vel_mean = data_ref['vel_mean']
arom_ref = data_ref['aROM']
asvd_ref = data_ref['asvd']
phi_ref = data_ref['phi']
lambda_ref = np.diag(data_ref['Lambda'])
mean_reduced_velocity_rom_ref = np.matmul(phi_ref, arom_ref)

data = np.load(python_pod_data_dir + 'matrices_Re25000_r50.npz')
L0_calc = data ['L0']
LRe_calc = data['LRe']
C0_calc = data['C0']
CRe_calc = data['CRe']
Q_calc = data['Q']
data_deriv = np.load(python_pod_data_dir + 'derivatives_r50.npz')
dvel_dx_calc = data_deriv['dvel_dx']
dvel_dy_calc = data_deriv['dvel_dy']
dphi_dx_calc = data_deriv['dphi_dx']
dphi_dy_calc = data_deriv['dphi_dy']
data_mean = np.load(python_pod_data_dir + 'vel_mean.npz')
vel_mean_calc = data_mean['vel_mean']
data_svd = np.load(python_pod_data_dir + 'svd_output_r50.npz')
lambda_calc = data_svd['Lambda']
phi_calc = data_svd['phi']
asvd_calc = data_svd['a']
data_rom = np.load(python_pod_data_dir + 'cavity_rom_data_r50_rk45.npz')
mean_reduced_velocity_rom = data_rom['mean_reduced_velocity_rom']
arom_calc = data_rom['aT']


print('L0 shape', L0.shape, L0_calc.shape)
print('LRe shape', LRe.shape, LRe_calc.shape)
print('C0 shape', C0.shape, C0_calc.shape)
print('Cre shape', CRe.shape, CRe_calc.shape)
print('Q shape', Q.shape, Q_calc.shape)
#print('dvel ref', dvel_dx.shape, dvel_dy.shape)
#print('dphi ref', dphi_dx.shape, dphi_dy.shape)
#print('dvel calc', dvel_dx_calc.shape, dvel_dy_calc.shape)
#print('dphi calc', dphi_dx_calc.shape, dphi_dy_calc.shape)
#print('vel mean ref', vel_mean.shape)
#print('vel mean calc', vel_mean_calc.shape)
print('asvd calc', asvd_calc.shape)
print('asvd ref', asvd_ref.shape)
print('phi calc', phi_calc.shape)
print('phi ref', phi_ref.shape)
print('lambda calc', lambda_calc.shape)
print('lambda ref', lambda_ref.shape)
print('arom ref', arom_ref.shape)
print('arom calc', arom_calc.shape)


#norm_L0 = norm(L0 - L0_calc)
#norm_LRe = norm(LRe - LRe_calc)
#norm_C0 = norm(C0 - C0_calc)
#norm_CRe = norm(CRe - CRe_calc)
#norm_Q = norm(Q[:,0,:,:] - Q_calc[:,:,:])
#norm_dvel_x = norm(dvel_dx_calc[0:2*num_cell] - dvel_dx[:,0])
#norm_dvel_y = norm(dvel_dy_calc[0:2*num_cell] - dvel_dy[:,0])
#norm_dphi_x = norm(dphi_dx_calc[0:2*num_cell,:] - dphi_dx[:,:])
#norm_dphi_y = norm(dphi_dy_calc[0:2*num_cell, :] - dphi_dy[:,:])
#norm_u_mean = norm(vel_mean_calc[:,0] - vel_mean[0:num_cell, 0])
#norm_v_mean = norm(vel_mean_calc[:,1] - vel_mean[num_cell: 2*num_cell, 0])
#norm_asvd = norm(asvd_ref - asvd_calc)
norm_arom = norm(arom_ref - arom_calc)
#norm_phi = norm(phi_ref[:,:] - phi_calc[0:2*num_cell, :])
#
#print('norm of L0', norm_L0)
#print('norm of LRe', norm_LRe)
#print('norm of C0', norm_C0)
#print('norm of CRe', norm_CRe)
#print('norm of Q', norm_Q)
#print('norm of dphidx', norm_dphi_x)
#print('norm of dphidy', norm_dphi_y)
#print('norm of dveldx', norm_dvel_x)
#print('norm of dveldy', norm_dvel_y)
#print('norm of mean u', norm_u_mean)
#print('norm of mean v', norm_v_mean)
#print('norm of phi ', norm_phi)
#print('norm of asvd', norm_asvd)
print('norm of arom', norm_arom)

##### compare with results of ode with reerence matrices
#data= np.load(python_pod_data_dir + 'cavity_rom_data_r50_ref_matrices.npz')

#mean_reduced_velocity_rom = data['mean_reduced_velocity_rom']
#mean_reduced_velocity_rom[0:2*num_cell, :] = \
#  mean_reduced_velocity_rom_ref[0:2*num_cell, :]
#arom_ref_matrices = data_rom['aT']
#print('norm of a_ref_matrices',  norm(arom_ref - arom_ref_matrices))


r = 50

velocity_snapshots = data_ref['velocity_snap']
cell_volume = data_ref['cell_volume']
xi = data_ref['xi_index']
eta = data_ref['xi_index']
zeta = [0]
Ndim = 3
Nzeta = 1
t_plot = 2000
num_cell = Nxi*Neta
num_dof = int(num_cell*Ndim)
num_snapshots = (velocity_snapshots.shape)[1]
xi_index = np.zeros([num_cell])
eta_index = np.zeros([num_cell])
zeta_index = np.zeros([num_cell])
cell_center = np.zeros([Nxi, Neta, Nzeta, Ndim])


X = np.linspace(0,r, r)
Y = X
#plt.contourf(X, Y, L0)
#plt.colorbar()
#plt.title('L0')
#plt.savefig('matrix_L0.png')
#plt.clf()
#
#plt.contourf(X, Y, L0-L0_calc)
#plt.colorbar()
#plt.title('L0')
#plt.savefig('matrix_L0_diff.png')
#plt.clf()
#
#plt.contourf(X, Y, LRe)
#plt.colorbar()
#plt.title('LRe')
#plt.savefig('matrix_LRe.png')
#plt.clf()
#
#plt.contourf(X, Y, LRe- LRe_calc)
#plt.colorbar()
#plt.title('LRe')
#plt.savefig('matrix_LRe_diff.png')
#plt.clf()
#
#plt.plot(C0 - C0_calc)
#plt.title('C0 difference')
#plt.savefig('matrix_C0_diff.png')
#plt.clf()
#
#plt.plot(C0[:])
#plt.title('C0')
#plt.savefig('matrix_C0.png')
#plt.clf()
#
#plt.plot(CRe[:] - CRe_calc[:])
#plt.title('CRe difference')
#plt.savefig('matrix_CRe_diff.png')
#plt.clf()
#
#plt.plot(CRe[:])
#plt.title('CRe')
#plt.savefig('matrix_CRe.png')
#plt.clf()
#
#plt.plot(dvel_dx[0:num_cell, 0])
#plt.title('dvel/dx')
#plt.savefig('dvel_dx.png')
#plt.clf()
#
#plt.plot(dvel_dx_calc[0:num_cell] - dvel_dx[0:num_cell,0])
#plt.title('dvel/dx diff')
#plt.savefig('dvel_dx_diff.png')
#plt.clf()
#
#
#plt.plot(dphi_dx[0:num_cell,:])
#plt.title('dphi /dx')
#plt.savefig('dphi_dx.png')
#plt.clf()
#
#plt.plot(dphi_dx[0:num_cell,:] - dphi_dx_calc[0:num_cell, :])
#plt.title('dphi /dx diff')
#plt.savefig('dphi_dx_diff.png')
#plt.clf()
#
#plt.plot(np.log(lambda_calc/lambda_calc[0]), '*', label = 'calc')
#plt.plot(np.log(lambda_ref/lambda_ref[0]), label = 'ref')
#plt.xlabel('number od modes')
#plt.ylabel('ln(lambda/lambda_0)')
#plt.legend()
#plt.savefig('lambda_ratio.png')
#plt.clf()

#plt.plot(np.log(lambda_calc), '*', label = 'calc')
#plt.plot(np.log(lambda_ref), label= 'ref')
#plt.xlabel('number of modes')
#plt.ylabel('ln(lambda)')
#plt.legend()
#plt.savefig('lambda.png')
#plt.clf()
#


#plt.plot(((arom_ref_matrices[1,0:30000])).transpose(), label = 'ref_matrices')
#plt.plot(((arom_ref[1,0:3000])).transpose(), label = 'ref')
#plt.xlabel('time steps')
#plt.ylabel('2nd modal coefficients')
#plt.legend()
#plt.savefig('last_modal_coeff_over_t_ref_mat_rk.png')
#plt.clf()


plt.plot((arom_calc[0,0:3000]).transpose(), label = 'calc')
plt.plot((arom_ref[0,0:3000]).transpose(), label = 'ref')
plt.xlabel('time steps')
plt.ylabel('modal coefficients')
plt.legend()
plt.savefig('first_modal_coeff_r50_odeint.png')
plt.clf()
#plt.xlabel('time steps')
#plt.ylabel('modal coefficients')
#plt.savefig('ref_modal_coeff_over_t.png')
#plt.clf()

#mean_reduced_velocity_rom = data['velocity_pca']
#### 3D topology
weights_ND = np.zeros([num_dof])
weights_ND[0:num_cell*2] = cell_volume[:,0]
velocity_data = np.zeros([num_dof, num_snapshots])
velocity_data[0:num_cell*2 ,:] = velocity_snapshots[:,:]
cell_center_uniform = np.zeros(cell_center.shape)
for i_xi in range(Nxi):
  for i_eta in range(Neta):
    for i_zeta in range(Nzeta):
        ldx = i_xi + Nxi * (i_eta + Neta * i_zeta)
        xi_index[ldx] = int(i_xi)
        eta_index[ldx] = int(i_eta)
        zeta_index[ldx] = int(i_zeta)
        cell_center[i_xi, :, :, 0] = xi[i_xi]
        cell_center[:, i_eta, :, 1] = eta[i_eta]
        cell_center[:, :, i_zeta, 2] = zeta[i_zeta]


xi_index = xi_index.astype(int)
eta_index = eta_index.astype(int)
zeta_index = zeta_index.astype(int)

velocity_mean = pod_modes.Find_Mean(velocity_data)
temp = np.zeros([num_dof, num_snapshots])
velocity_mean = np.reshape(velocity_mean.transpose(), (num_dof))
temp[:,t_plot] = mean_reduced_velocity_rom[:,t_plot] + velocity_mean[:]

velocity_rom = np.zeros([num_cell, Ndim])
velocity_data_2d = np.zeros([num_cell, Ndim])
velocity_rom[:,:] = (np.reshape(temp[:,t_plot], (Ndim, num_cell))).transpose()
velocity_data_2d[:, :] = \
  (np.reshape(velocity_data[:, t_plot], (Ndim, num_cell))).transpose()

velocity_rom_3d = np.zeros([Nxi, Neta, Nzeta, Ndim])
velocity_data_3d = np.zeros([Nxi, Neta, Nzeta, Ndim])
for i_dim in range(Ndim):
  velocity_rom_3d[:,:,:,i_dim] = structure_format.array_1D_to_3D(\
    xi_index, eta_index, zeta_index, num_cell, velocity_rom[:,i_dim])
  velocity_data_3d[:, :, :, i_dim] = structure_format.array_1D_to_3D(\
    xi_index, eta_index, zeta_index, num_cell, velocity_data_2d[:, i_dim])


vorticity_fom = \
  curl_calc.curl_2d(-cell_center[0,:,0,1], cell_center[:,0,0,0],
    velocity_data_3d[:, :, 0, 0], velocity_data_3d[:, :, 0, 1])
vorticity_rom = \
  curl_calc.curl_2d(-cell_center[0,:,0,1], cell_center[:,0,0,0],
    velocity_rom_3d[:, :, 0, 0], velocity_rom_3d[:, :, 0, 1])
print('vorticity', vorticity_fom.shape)
v_min = -3
v_max = 3
n_levels = 100 
Derivative_Calc.plot_contour(\
  -cell_center[0,:,0,1],
  cell_center[:,0,0,0],
  vorticity_fom[:,:],
  'vorticity_DNS_2000dt.png',
  n_levels, v_min, v_max)
Derivative_Calc.plot_contour(\
  -cell_center[:,:,0,1],
  cell_center[:,:,0,0],
  vorticity_rom[:,:], \
  'vorticity_rom_r50_2000dt_rk45.png',
  n_levels, v_min, v_max)
