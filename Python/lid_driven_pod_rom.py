# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:10:36 2022

@author: USER
"""
import numpy as np
import get_artificial_mean

def main(poi_normalized, poi_selector, qoi_selector,un_normalizer):
    #=============================Unapply normalization========================
    poi = un_normalizer(poi_normalized)
    del poi_normalized
    #================================Load POI Values===========================
    if poi_selector.ndim !=1:
        raise Exception("poi_selector more than 1 dimension")
    if poi.ndim ==1:
        n_poi = poi.size
        n_samp =1
    elif poi.ndim == 2:
        n_poi = poi.shape[1]
        n_samp = poi.shape=[0]
    if n_poi!= poi_selector.size:
        raise Exception("poi and poi_selector different lengths")
    #Intialize matrices for local POIs to improve processing
    basis_vort_mat = np.empty((n_samp,0))
    basis_orient_mat = np.empty((n_samp,0))
    basis_x_loc_mat = np.empty((n_samp,0))
    basis_y_loc_mat = np.empty((n_samp,0))
    basis_extent_mat = np.empty((n_samp,0))
    #Load pois and unnormalize
    if n_samp ==1:
        for i_poi in range(n_poi):
            if poi_selector[n_poi].lower() == 're':
                reynolds_number_vec = poi[[i_poi]]
            elif poi_selector[n_poi].lower() == 'boundary exponent':
                boundary_exp_vec = poi[[i_poi]]
            elif poi_selector[n_poi].lower() == 'penalty strength':
                pen_strength_vec = poi[[i_poi]]
            elif poi_selector[n_poi].lower()[0:10] == 'basis vort':
                basis_vort_mat=np.append(basis_vort_mat, \
                                         poi[i_poi].reshape(1,1),\
                                         axis=1)
            elif poi_selector[n_poi].lower()[0:12] == 'basis orient':
                basis_orient_mat=np.append(basis_orient_mat, \
                                           poi[i_poi].reshape(1,1), \
                                           axis=1)
            elif poi_selector[n_poi].lower()[0:16] == 'basis x-location':
                basis_x_loc_mat=np.append(basis_x_loc_mat,\
                                             poi[i_poi].reshape(1,1), \
                                             axis=1)
            elif poi_selector[n_poi].lower()[0:16] == 'basis y-location':
                basis_y_loc_mat=np.append(basis_y_loc_mat,\
                                             poi[i_poi].reshape(1,1), \
                                             axis=1)
            elif poi_selector[n_poi].lower()[0:12] == 'basis extent':
                basis_extent_mat=np.append(basis_extent_mat, \
                                           poi[i_poi].reshape(1,1), \
                                           axis=1)
            else :
                raise Exception("Unrecognized poi name: " + str(poi_selector[i_poi]))
    else:
        for i_poi in range(n_poi):
            if poi_selector[n_poi].lower() == 're':
                reynolds_number_vec = poi[:,i_poi]
            elif poi_selector[n_poi].lower() == 'boundary exponent':
                boundary_exp_vec = poi[:, i_poi]
            elif poi_selector[n_poi].lower() == 'penalty strength':
                pen_strength_vec = poi[:,i_poi]
            elif poi_selector[n_poi].lower()[0:10] == 'basis vort':
                basis_vort_mat=np.append(basis_vort_mat, \
                                         poi[:,i_poi].reshape(n_samp,1),\
                                         axis=1)
            elif poi_selector[n_poi].lower()[0:12] == 'basis orient':
                basis_orient_mat=np.append(basis_orient_mat, \
                                           poi[:,i_poi].reshape(n_samp,1), \
                                           axis=1)
            elif poi_selector[n_poi].lower()[0:16] == 'basis x-location':
                basis_x_loc_mat=np.append(basis_x_loc_mat,\
                                             poi[:,i_poi].reshape(n_samp,1), \
                                             axis=1)
            elif poi_selector[n_poi].lower()[0:16] == 'basis y-location':
                basis_y_loc_mat=np.append(basis_y_loc_mat,\
                                             poi[:,i_poi].reshape(n_samp,1), \
                                             axis=1)
            elif poi_selector[n_poi].lower()[0:12] == 'basis extent':
                basis_extent_mat=np.append(basis_extent_mat, \
                                           poi[:,i_poi].reshape(n_samp,1), \
                                           axis=1)
            else :
                raise Exception("Unrecognized poi name: " + str(poi_selector[i_poi]))
    for i_samp in range(n_samp):
        #Unpack POIs
        reynolds_number = reynolds_number_vec[i_samp]
        boundary_exp = boundary_exp_vec[i_samp]
        pen_strength = pen_strength_vec[i_samp]
        basis_vort_vec = basis_vort_mat[i_samp]
        basis_orient_vec = basis_orient_mat[i_samp]
        basis_x_loc_vec = basis_x_loc_mat[i_samp]
        basis_y_loc_vec = basis_y_loc_mat[i_samp]
        basis_extent_vec = basis_extent_mat[i_samp]
        #Construct snapshot and pod
        if i_samp >= 1:
            basis_vort_vec_old = basis_vort_mat[i_samp-1]
            basis_orient_vec_old = basis_orient_mat[i_samp-1]
            basis_x_loc_vec_old = basis_x_loc_mat[i_samp-1]
            basis_y_loc_vec_old = basis_y_loc_mat[i_samp-1]
            basis_extent_vec_old = basis_extent_mat[i_samp-1]
            #If mean reduction is same as previous, do not run basis reduction
            if np.all(basis_vort_vec == basis_vort_vec_old) and \
               np.all(basis_orient_vec == basis_orient_vec_old) and \
               np.all(basis_x_loc_vec == basis_x_loc_vec_old) and \
               np.all(basis_y_loc_vec == basis_y_loc_vec_old) and \
               np.all(basis_extent_vec ==basis_extent_vec_old):
                   # Keep old poi
                   pass
            else:
                mean_reduction = get_artificial_mean(\
                                  basis_vort_vec, basis_orient_vec, basis_x_loc_vec,\
                                  basis_y_loc_vec, basis_extent_vec, Xi, Eta, \
                                  Xi_mesh, Eta_mesh, cell_centroid)
                
                
        else:
            mean_reduction = get_artificial_mean(\
                              basis_vort_vec, basis_orient_vec, basis_x_loc_vec,\
                              basis_y_loc_vec, basis_extent_vec, Xi, Eta, \
                              Xi_mesh, Eta_mesh, cell_centroid)
    
    #Caclulate ROM matrices
    
    #Integrate ROM
    
    #Get QOI values
    
    return QOIs
    
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
            
    