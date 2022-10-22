# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:10:36 2022

@author: USER
"""
import numpy as np
from get_artificial_mean import main as get_artificial_mean
from get_pod_lid_input_variant import main as get_pod
from aerofusion.numerics import derivatives_curvilinear_grid as curvder
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.data import array_conversion as arr_conv
from aerofusion.numerics import curl_calc as curl_calc
import gc
from aerofusion.plot.plot_2D import plot_pcolormesh

def main(poi_normalized, poi_selector, qoi_selector, poi_bounds, num_modes,\
         discretization, velocity_unreduced_1D_compact, integration_times,
         center_mat = np.empty((0)), local_radius = .2, plot = False, 
         mean_type = "artificial", use_energy = False):
 
    
    #=============================Load Discretization==========================
    Xi = discretization["Xi"]
    Eta = discretization["Eta"]
    zeta = discretization["Zeta"]
    cell_centroid = discretization["cell_centroid"]
    num_cell = discretization["num_cell"]
    weights_ND = discretization["weights_ND"]
    del discretization
    num_xi = 258
    num_eta = 258
    num_zeta = 1
    n_dim = 2
    n_cell = num_xi * num_eta * num_zeta
    
    #=============================Inititalize Matrices=========================
    if poi_selector.ndim !=1:
        raise Exception("poi_selector more than 1 dimension")
    if poi_normalized.ndim ==1:
        n_poi = poi_normalized.size
        n_samp =1
    elif poi_normalized.ndim == 2:
        n_poi = poi_normalized.shape[1]
        n_samp = poi_normalized.shape[0]
    if n_poi!= poi_selector.size:
        raise Exception("poi and poi_selector different lengths")
    n_qoi = qoi_selector.size
    #Intialize matrices for local POIs to improve processing
    basis_speed_mat = np.empty((n_samp,0))
    basis_orient_mat = np.empty((n_samp,0))
    basis_x_loc_mat = np.empty((n_samp,0))
    basis_y_loc_mat = np.empty((n_samp,0))
    basis_extent_mat = np.empty((n_samp,0))
    #=============================Unapply normalization========================
    poi = un_normalize_poi(poi_normalized, poi_bounds)
    #print("poi: " + str(poi))
    del poi_normalized
    #================================Load POI Values===========================
    #Load pois and unnormalize
    #print("n_poi: " + str(n_poi))
    #print("n_samp: " + str(n_samp))
    if n_samp ==1:
        for i_poi in range(n_poi):
            if poi_selector[i_poi].lower() == 're':
                reynolds_number_vec = poi[[i_poi]]
            elif poi_selector[i_poi].lower() == 'boundary exponent mult':
                boundary_exp_vec = poi[[i_poi]]
            elif poi_selector[i_poi].lower() == 'penalty strength exp':
                pen_strength_exp_vec = poi[[i_poi]]
            elif poi_selector[i_poi].lower()[0:11] == 'basis speed':
                basis_speed_mat=np.append(basis_speed_mat, \
                                         poi[i_poi].reshape(1,1),\
                                         axis=1)
            elif poi_selector[i_poi].lower()[0:12] == 'basis orient':
                basis_orient_mat=np.append(basis_orient_mat, \
                                           poi[i_poi].reshape(1,1), \
                                           axis=1)
            elif poi_selector[i_poi].lower()[0:16] == 'basis x-location':
                basis_x_loc_mat=np.append(basis_x_loc_mat,\
                                             poi[i_poi].reshape(1,1), \
                                             axis=1)
            elif poi_selector[i_poi].lower()[0:16] == 'basis y-location':
                basis_y_loc_mat=np.append(basis_y_loc_mat,\
                                             poi[i_poi].reshape(1,1), \
                                             axis=1)
            elif poi_selector[i_poi].lower()[0:12] == 'basis extent':
                basis_extent_mat=np.append(basis_extent_mat, \
                                           poi[i_poi].reshape(1,1), \
                                           axis=1)
            else :
                raise Exception("Unrecognized poi name: " + str(poi_selector[i_poi]))
    else:
        for i_poi in range(n_poi):
            if poi_selector[i_poi].lower() == 're':
                reynolds_number_vec = poi[:,i_poi]
            elif poi_selector[i_poi].lower() == 'boundary exponent mult':
                boundary_exp_vec = poi[:, i_poi]
            elif poi_selector[i_poi].lower() == 'penalty strength exp':
                pen_strength_exp_vec = poi[:,i_poi]
            elif poi_selector[i_poi].lower()[0:11] == 'basis speed':
                basis_speed_mat=np.append(basis_speed_mat, \
                                         poi[:,i_poi].reshape(n_samp,1),\
                                         axis=1)
            elif poi_selector[i_poi].lower()[0:12] == 'basis orient':
                basis_orient_mat=np.append(basis_orient_mat, \
                                           poi[:,i_poi].reshape(n_samp,1), \
                                           axis=1)
            elif poi_selector[i_poi].lower()[0:16] == 'basis x-location':
                basis_x_loc_mat=np.append(basis_x_loc_mat,\
                                             poi[:,i_poi].reshape(n_samp,1), \
                                             axis=1)
            elif poi_selector[i_poi].lower()[0:16] == 'basis y-location':
                basis_y_loc_mat=np.append(basis_y_loc_mat,\
                                             poi[:,i_poi].reshape(n_samp,1), \
                                             axis=1)
            elif poi_selector[i_poi].lower()[0:12] == 'basis extent':
                basis_extent_mat=np.append(basis_extent_mat, \
                                           poi[:,i_poi].reshape(n_samp,1), \
                                           axis=1)
            else :
                print("Poi_name: " + str(poi_selector[i_poi]))
                raise Exception("Unrecognized poi name")
    gc.collect()
    #===============================Start model runs===========================
    for i_samp in range(n_samp):
        #Unpack POIs
        reynolds_number = reynolds_number_vec[i_samp]
        boundary_exp = 10**(boundary_exp_vec[i_samp])
        penalty_strength = 10**pen_strength_exp_vec[i_samp]
        basis_speed_vec = basis_speed_mat[i_samp]
        basis_orient_vec = basis_orient_mat[i_samp]
        basis_x_loc_vec = basis_x_loc_mat[i_samp]
        basis_y_loc_vec = basis_y_loc_mat[i_samp]
        basis_extent_vec = basis_extent_mat[i_samp]
        #Formulate boundary vector
        x_boundary = cell_centroid[0,:,0,0]
        boundary_vec = (np.abs(1-x_boundary)*np.abs(1+x_boundary))**boundary_exp
        
        #--------------------Construct snapshot and pod------------------------
        if i_samp >= 1:
            basis_speed_vec_old = basis_speed_mat[i_samp-1]
            basis_orient_vec_old = basis_orient_mat[i_samp-1]
            basis_x_loc_vec_old = basis_x_loc_mat[i_samp-1]
            basis_y_loc_vec_old = basis_y_loc_mat[i_samp-1]
            basis_extent_vec_old = basis_extent_mat[i_samp-1]
            #If mean reduction is same as previous, do not run basis reduction
            if np.all(basis_speed_vec == basis_speed_vec_old) and \
               np.all(basis_orient_vec == basis_orient_vec_old) and \
               np.all(basis_x_loc_vec == basis_x_loc_vec_old) and \
               np.all(basis_y_loc_vec == basis_y_loc_vec_old) and \
               np.all(basis_extent_vec ==basis_extent_vec_old):
                   # Keep old poi
                   #print("Skipping Mean reduction")
                   pass
            else:
                vel_0_1D_compact = get_artificial_mean(\
                                  basis_speed_vec, basis_orient_vec, basis_x_loc_vec,\
                                  basis_y_loc_vec, basis_extent_vec, Xi, Eta, \
                                  cell_centroid, plot = plot)
            
                (phi, modal_coeff, pod_lambda) = get_pod(velocity_unreduced_1D_compact, \
                                                         mean_type, \
                                                         vel_0_1D_compact,\
                                                         weights_ND, \
                                                         num_modes,
                                                         use_energy = use_energy)
                
                
        else:
            vel_0_1D_compact = get_artificial_mean(\
                              basis_speed_vec, basis_orient_vec, basis_x_loc_vec,\
                              basis_y_loc_vec, basis_extent_vec, Xi, Eta, \
                              cell_centroid, plot= plot)
            (phi, modal_coeff, pod_lambda) = get_pod(velocity_unreduced_1D_compact, \
                                                     "artificial", \
                                                     vel_0_1D_compact,\
                                                     weights_ND, \
                                                     num_modes,
                                                     use_energy = use_energy)
        
        #--------------------------ROM Matrix calculation----------------------
        #Convert 1D mean reduction to 3D for ROM matrix calculations
        vel_0_1D = arr_conv.array_compact_to_1D(vel_0_1D_compact, n_dim, n_cell)
        vel_0_3D = np.empty(((num_xi, num_eta, num_zeta, n_dim)))
        for i_dim in range(n_dim):
            vel_0_3D[:,:,0,i_dim] = arr_conv.array_1D_to_2D(\
              Xi, Eta, num_xi, num_eta, vel_0_1D[:,i_dim])
        vel_0_2D = vel_0_3D[:,:,0,:]
        #Convert from compact to 3D variations
        #Caclulate ROM matrices
        #print("Calculating Jacobian")
        jacobian = curvder.jacobian_of_grid_2d2(\
            Xi,
            Eta,
            zeta,
            cell_centroid,
            2)
            #options.rom.jacobian.order_derivatives_x,
            #options.rom.jacobian.order_derivatives_y,
            #options.rom.jacobian.order_derivatives_z)
        (L0_calc, LRe_calc, C0_calc, CRe_calc, Q_calc) = \
            incrom.pod_rom_matrices_2d(\
              Xi,
              Eta,
              zeta,
              cell_centroid,
              num_cell,
              phi,
              weights_ND,
              vel_0_3D,
              jacobian,
              6)
                
        #print(np.max(LRe_calc))
        #print(np.max(L0_calc))
        #print(np.max(C0_calc))
        #       #options.rom.jacobian.order_derivatives_x,
        #       #options.rom.jacobian.order_derivatives_y, 
        #       #options.rom.jacobian.order_derivatives_z)
        (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
          Xi,
          Eta,
          zeta,
          cell_centroid,
          num_cell,
          phi,
          weights_ND,
          vel_0_2D, 
          boundary_vec) 
        del jacobian, vel_0_2D
        
        gc.collect()
        #-----------------------------Integrate ROM----------------------------
        char_L = 1
        
        #print("Solving System")
        aT = incrom.rom_calc_rk45_boundary(\
                reynolds_number,
                char_L,
                L0_calc,
                LRe_calc,
                C0_calc,
                CRe_calc,
                Q_calc,
                B_calc,
                B0_calc,
                modal_coeff,
                integration_times,
                penalty_strength)

        modal_coeff = aT[:,[-1]]
        mean_reduced_velocity = np.matmul(phi, modal_coeff).flatten()
        velocity_rom_1D_compact = mean_reduced_velocity + vel_0_1D_compact
        #Transfer to 2D
        velocity_rom_1D = arr_conv.array_compact_to_1D(velocity_rom_1D_compact, n_dim, n_cell)
        weights_1D = arr_conv.array_compact_to_1D(weights_ND, n_dim, n_cell)

        velocity_rom_2D = np.zeros((num_xi, num_eta, n_dim))
        weights_2D = np.zeros((num_xi, num_eta, n_dim))
        for i_dim in range(2):
              velocity_rom_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
                Xi, Eta, num_xi, num_eta, velocity_rom_1D[:,i_dim])
              weights_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
                Xi, Eta, num_xi, num_eta, weights_1D[:,i_dim])
        
        vorticity_2D = curl_calc.curl_2d(-cell_centroid[:,0,0,1], -cell_centroid[0,:,0,0],
                                         velocity_rom_2D[:, :, 0], velocity_rom_2D[:, :,1])
        del mean_reduced_velocity, \
            velocity_rom_1D, L0_calc, LRe_calc, C0_calc, CRe_calc, Q_calc, \
            B_calc, B0_calc
        gc.collect()
        #----------------------------Get QOI values----------------------------
        #print("Getting QOIs")
        qois_samp = np.empty((0,))
        for i_qoi in range(n_qoi):
            if qoi_selector[i_qoi].lower() == "energy":
                energy = np.sum(np.abs(modal_coeff))
                qois_samp = np.append(qois_samp, energy)
            elif qoi_selector[i_qoi].lower () == "vorticity": 
                vort = np.sum(np.abs(vorticity_2D*weights_2D[:,:,0]))
                qois_samp = np.append(qois_samp, vort)
            elif qoi_selector[i_qoi].lower()[:15] == "local vorticity":
                center_number = int(qoi_selector[i_qoi][16:])
                x_cent = center_mat[center_number, 0]
                y_cent = center_mat[center_number, 1]
                vorticity_local_2D = local_response(vorticity_2D, cell_centroid, \
                                                      x_cent, y_cent, local_radius)
                vort = np.max(np.abs(vorticity_local_2D*weights_2D[:,:,0]))
                qois_samp = np.append(qois_samp, vort)
            elif qoi_selector[i_qoi].lower() == "full velocity":
                qois_samp = np.append(qois_samp, velocity_rom_1D_compact)
            else :
                raise Exception("Unknown qoi: " + str(qoi_selector[i_qoi]))
        if i_samp == 0:
            qois = np.empty((n_samp, qois_samp.size))
        #print(qois_samp)
        qois[i_samp] = qois_samp
        del qois_samp
        #print(qois)
    del phi, vorticity_2D, modal_coeff
    gc.collect
    return qois.squeeze()

def local_response(data_2D, cell_centroid, x_cent, y_cent, radius):
    #Identify where in cell_centroid is within radius
    region = np.sqrt((cell_centroid[:,:,0,0] - x_cent)**2+\
                     (cell_centroid[:,:,0,1] - y_cent)**2) <= radius
    #Extract those voriticity values
    data_reduced_2D = data_2D * region
    #Take max and min
    return data_reduced_2D
    
def normalize_pois(poi_base, bounds):
    poi_normalized = np.empty(poi_base.shape)
    if poi_base.ndim == 1:
        n_poi = poi_base.shape[0]
        if bounds.shape!=(n_poi,2):
            raise Exception("Invalid bounds shape of " + str(bounds.shape))
        for i_poi in range(n_poi):
            poi_normalized[i_poi] = (poi_base[i_poi]-bounds[i_poi][0])/(bounds[i_poi][1]-bounds[i_poi][0])
    elif poi_base.ndim == 2:
        n_poi = poi_base.shape[1]
        if bounds.shape!=(n_poi,2):
            raise Exception("Invalid bounds shape of " + str(bounds.shape))
        for i_poi in range(n_poi):
            poi_normalized[:,i_poi] = (poi_base[:, i_poi]-bounds[i_poi][0])/(bounds[i_poi][1]-bounds[i_poi][0])
    return poi_normalized
            
def un_normalize_poi(poi_normalized, bounds):
    poi_base = np.empty(poi_normalized.shape)
    if poi_base.ndim == 1:
        n_poi = poi_normalized.shape[0]
        if bounds.shape!=(n_poi,2):
            raise Exception("Invalid bounds shape of " + str(bounds.shape))
        for i_poi in range(n_poi):
            poi_base[i_poi] = (poi_normalized[i_poi])*(bounds[i_poi][1]-bounds[i_poi][0])+bounds[i_poi][0]
    elif poi_base.ndim == 2:
        n_poi = poi_base.shape[1]
        if bounds.shape!=(n_poi,2):
            raise Exception("Invalid bounds shape of " + str(bounds.shape))
        for i_poi in range(n_poi):
            poi_base[:,i_poi] = (poi_normalized[:,i_poi])*(bounds[i_poi][1]-bounds[i_poi][0])+bounds[i_poi][0]
    return poi_base
