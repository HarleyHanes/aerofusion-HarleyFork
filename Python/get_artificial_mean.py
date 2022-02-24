# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:59:08 2022

@author: USER
"""
import scipy.io as mio
import numpy as np
import matplotlib.pyplot as plt
import aerofusion.data.array_conversion as arr_conv
from aerofusion.numerics import curl_calc as curl_calc
from aerofusion.plot.plot_2D import plot_pcolormesh
from aerofusion.numerics.derivatives import FD_derivative_2nd_order
from aerofusion.numerics.derivatives import Find_Derivative_FD
from aerofusion.numerics.derivatives_curvilinear_grid import jacobian_of_grid_2d2

def main(basis_vort_vec, basis_orient_vec, basis_x_loc_vec, basis_y_loc_vec, \
         basis_extent_vec, Xi, Eta, Xi_mesh, Eta_mesh, cell_centroid, base_extent = 20, plot = False):
    
    num_dim  = 2
    num_xi   = Xi_mesh.shape[0]
    num_eta  = Xi_mesh.shape[1]
    num_zeta = 1
    
    zeta = np.zeros(Xi.shape, dtype = int)
        
    

    num_cell = num_xi*num_eta*num_zeta

    #Define Functions used to formulate each basis function and initialize velocity
    radius = lambda xCent, yCent, x,y, COV: np.sqrt(\
        COV[0,0]*(x-xCent)**2+(COV[0,1]+COV[1,0])*(y-yCent)*(x-xCent)+COV[1,1]*(y-yCent)**2)
    expRBF = lambda xCent, yCent, x, y, COV: np.exp(-radius(xCent,yCent,x,y, COV)**2)
    #expRBF = lambda xCent, yCent, x, y, COV: np.exp(-(x**2+y**2))
    #expRBF_dx = lambda x,y: -2*x*np.exp(-(x**2+y**2))
    #expRBF_dy = lambda x,y: -2*y*np.exp(-(x**2+y**2))
    
    velocity_2D = np.zeros((num_xi, num_eta, num_dim))
    

    for i_basis in range(len(basis_orient_vec)):
            #load in parameters for previty
            orient = basis_orient_vec[i_basis]
            max_vort = basis_vort_vec[i_basis]
            x0 = basis_x_loc_vec[i_basis]
            y0 = basis_y_loc_vec[i_basis]
            rel_extent = basis_extent_vec[i_basis]
            
            #Formulate covariance matrix
            axis_length = np.array([[rel_extent*base_extent, 0], [0, base_extent]])
            rotation = np.array([[np.sin(orient), -np.cos(orient)],\
                                 [np.cos(orient), np.sin(orient)]])
            cov = np.matmul(np.matmul(rotation.transpose(), axis_length), rotation)
            
            #Formulate basis and convert to 3d for deriv calc
            basis_2D = expRBF(x0, y0, Xi_mesh, Eta_mesh, cov)
            del cov
            basis_2D_upsampled = np.empty(basis_2D.shape + (1,1))
            basis_2D_upsampled[:,:,0,0] = basis_2D
            
            #Convert stream function to vorticity
            basis_vel_2D = np.empty(basis_2D.shape + (2,))
            #Compute first just difference matrices, then scale derivatives by centroids
            # plt.pcolormesh(Xi_mesh,
            #              Eta_mesh,
            #              basis_2D_upsampled[:,:,0,0])
            # plt.title("Upsampled basis")
            # plt.show()
            #(dstream_dx, dstream_dy) = Find_Derivative_FD(basis_2D_upsampled, \
            #                                            cell_centroid, 2)
            # dstream_dxi = np.gradient(basis_2D, axis = 0)
            # dstream_deta = np.gradient(basis_2D, axis = 1)
            # (dcell_dx, dcell_dy) = FD_derivative_2nd_order(cell_centroid)
            # dxi_dx = dcell_dx[:,:,0,1]
            # deta_dy = dcell_dy[:,:,0,0]
            # print(deta_dy[:,:,:,1]-deta_dy[:,:,:,0])
            # plt.imshow(deta_dy[:,:,:,0])
            # plt.show()
            # plt.imshow(deta_dy[:,:,:,1])
            # plt.show()
            # plt.imshow(dxi_dx[:,:,:,0])
            # plt.show()
            # plt.imshow(dxi_dx[:,:,:,1])
            # plt.show()
            (dstream_dxi, dstream_deta) = FD_derivative_2nd_order(basis_2D_upsampled)
            # jac = jacobian_of_grid_2d2(Xi, Eta, zeta, cell_centroid, 2)
            
            # dx_dxi = arr_conv.array_1D_to_2D(Xi, Eta, 258, 258, jac[0,1,:])
            # dy_deta = arr_conv.array_1D_to_2D(Xi, Eta, 258, 258, jac[1,0,:])
            
            # dxi_dx = np.linalg.inv(dx_dxi)
            # deta_dy = np.linalg.inv(dy_deta)
            
            basis_vel_2D[:,:,0] = dstream_deta[:,:,0,0]#*deta_dy[:,:,0]
            basis_vel_2D[:,:,1] = -dstream_dxi[:,:,0,0]#*dxi_dx[:,:,1]
            # plt.pcolormesh(Xi_mesh,
            #              Eta_mesh,
            #              basis_vel_2D[:,:,0])
            # plt.title("u")
            # plt.show()
            # plt.pcolormesh(Xi_mesh,
            #              Eta_mesh,
            #              expRBF_dy(Xi_mesh, Eta_mesh)-basis_vel_2D[:,:,0])
            # plt.title("u error")
            # plt.show()
            # plt.pcolormesh(Xi_mesh,
            #              Eta_mesh,
            #              basis_vel_2D[:,:,0])
            # plt.title("v")
            # plt.show()
            # plt.pcolormesh(Xi_mesh,
            #              Eta_mesh,
            #              -expRBF_dx(Xi_mesh, Eta_mesh)-basis_vel_2D[:,:,0])
            # plt.title("v error")
            # plt.show()
            
            #Convert to vorticity 
            vorticity = \
              curl_calc.curl_2d(-cell_centroid[:,0,0,1], -cell_centroid[0,:,0,0],
                basis_vel_2D[:,:,0], basis_vel_2D[:,:,1])
            #Determine how much the voriticity would need to be scaled by so we
            # can rescale the velocity by the same amount since their magnitudes
            # follow scalar multiple relationship
            prop_vort_change = np.max(np.abs(vorticity))/max_vort
            basis_vel_2D = basis_vel_2D/ prop_vort_change
            
            velocity_2D += basis_vel_2D
            
    
    #------------------------Get velocity_1D_compact

    #Convert to 1D
    velocity_1D=np.empty((num_cell,2))
    for i in range(2):
        velocity_1D[:,i] = arr_conv.array_2D_to_1D(\
                       Xi, Eta, num_cell, velocity_2D[:,:,i])
    velocity_1D_compact = arr_conv.array_1D_to_compact(velocity_1D)
    
    #Convert back to mesh for sanity check
    # check_velocity_1D = np.reshape(velocity_1D_compact, (num_cell, num_dim), order = 'F')
    # check_velocity_2D = np.empty(velocity_2D.shape)
    # for i in range(2):
    #     check_velocity_2D[:,:,i] = arr_conv.array_1D_to_2D(\
    #                             Xi, Eta, num_xi, num_eta, check_velocity_1D[:,i])
    # if np.max(check_velocity_2D-velocity_2D)>1e-10:
    #     raise(Exception("Array Conversion flawed"))
    if plot:
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          velocity_2D[:,:,0],
          "artificial_u0_vel",
          vmin = "auto", #-np.max(np.abs(data_2D)),
          vmax = "auto", #np.max(np.abs(data_2D)),
          colorbar_label= "u",
          font_size = 30)
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          velocity_2D[:,:,1],
          "artificial_v0_vel",
          vmin = "auto", #-np.max(np.abs(data_2D)),
          vmax = "auto", #np.max(np.abs(data_2D)),
          colorbar_label= "v",
          font_size = 30)
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vorticity*prop_vort_change,
          "artificial_vort",
          vmin = "auto", #-np.max(np.abs(data_2D)),
          vmax = "auto", #np.max(np.abs(data_2D)),
          colorbar_label= "vorticity",
          font_size = 30)
        # plot_pcolormesh(\
        #   Xi_mesh,
        #   Eta_mesh,
        #   dstream_dxi,
        #   "artificial_dstream_dxi",
        #   vmin = "auto", #-np.max(np.abs(data_2D)),
        #   vmax = "auto", #np.max(np.abs(data_2D)),
        #   colorbar_label= "dstream_dxi",
        #   font_size = 30)
        # plot_pcolormesh(\
        #   Xi_mesh,
        #   Eta_mesh,
        #   dstream_deta,
        #   "artificial_dstream_deta",
        #   vmin = "auto", #-np.max(np.abs(data_2D)),
        #   vmax = "auto", #np.max(np.abs(data_2D)),
        #   colorbar_label= "dstream_deta",
        #   font_size = 30)
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          basis_2D,
          "artificial_stream",
          vmin = "auto", #-np.max(np.abs(data_2D)),
          vmax = "auto", #np.max(np.abs(data_2D)),
          colorbar_label= "stream",
          font_size = 30)

    return velocity_1D_compact

if __name__ == '__main__':
    main()

