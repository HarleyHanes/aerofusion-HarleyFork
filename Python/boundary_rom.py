# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:40:51 2022

@author: USER
"""
import numpy as np
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.numerics import curl_calc as curl_calc
from aerofusion.data import array_conversion as arr_conv

def BoundaryPenaltyROM(POIs, QOIselector,rom_matrices_filename, integration_times, \
           discretization, pod,penalty_strength):
    print("Running ROM")
    
    #Unpack discretization
    Xi = discretization["Xi"]
    Eta = discretization["Eta"]
    Zeta = discretization["Zeta"]
    cell_centroid = discretization["cell_centroid"]
    num_cell = discretization["num_cell"]
    weights_ND = discretization["weights_ND"]
    
    #Unpack pod
    phi = pod["phi"]
    modal_coeff = pod["modal_coeff"]
    vel_0_2D = pod["vel_0_2D"]
    reynolds_number = pod["reynolds_number"]
    
    num_xi = vel_0_2D.shape[0]
    num_eta = vel_0_2D.shape[1]
    num_cell = num_xi*num_eta
    
    #Unpack POIs
    boundary_vec = POIs
    
    #Load ROM matrices
    matrices = np.load(rom_matrices_filename)
    L0_calc  = matrices['L0_calc']
    LRe_calc = matrices['LRe_calc']
    C0_calc  = matrices['C0_calc']
    CRe_calc = matrices['CRe_calc']
    Q_calc   = matrices['Q_calc']


    char_L = 1
    #print(reynolds_number)
    #Calculate ROM solution
    #Initialize matrices
    if boundary_vec.ndim == 2: 
        for iSamp in range(boundary_vec.shape[0]):
        
            #Compute Boundary ROM matrices
            (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
              Xi,
              Eta,
              Zeta,
              cell_centroid,
              num_cell,
              phi,
              weights_ND,
              vel_0_2D, 
              boundary_vec[iSamp])
            #print('Reynolds Number before rom_calc_rk45:' + str(reynolds_number[iSamp]))
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
            
           # print('Modal Coefficients at t=0'+str(aT[:,0]))
           # print('Modal Coeffecients at t=tmax'+str(aT[:,-1]))
           # print('Max mean_reduced_velocity:'+str(np.max(mean_reduced_velocity_rom)))
           # print('Mean mean_reduced_velocity:'+ str(np.mean(mean_reduced_velocity_rom)))
           # print()
    
            #Extract QOIs
            if QOIselector.lower()=="full data":
                #print(str(np.max(mean_reduced_velocity_rom)) + ', ' + str(np.mean(mean_reduced_velocity_rom)))       
               # QOIs[iSamp,:]=mean_reduced_velocity_rom[:,-1]
                mean_reduced_velocity_rom = np.matmul(phi, aT)
                QOIs.append(mean_reduced_velocity_rom[:,-1])
                del mean_reduced_velocity_rom
            if QOIselector.lower()=="modal coeff":
                QOIs.append(aT.flatten())
            if QOIselector.lower() =="kinetic energy":
                energy = np.sum(aT, axis = 0)
                energy = energy[-1]
                QOIs.append(energy)
    elif boundary_vec.ndim == 1:
        #Compute Boundary ROM matrices
        (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
          Xi,
          Eta,
          Zeta,
          cell_centroid,
          num_cell,
          phi,
          weights_ND,
          vel_0_2D, 
          boundary_vec)
        #print('Reynolds Number before rom_calc_rk45:' + str(reynolds_number[iSamp]))
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
                10^4)
        
       # print('Modal Coefficients at t=0'+str(aT[:,0]))
       # print('Modal Coeffecients at t=tmax'+str(aT[:,-1]))
       # print('Max mean_reduced_velocity:'+str(np.max(mean_reduced_velocity_rom)))
       # print('Mean mean_reduced_velocity:'+ str(np.mean(mean_reduced_velocity_rom)))
       # print()

        #Extract QOIs
        if QOIselector.lower()=="full data":
            #print(str(np.max(mean_reduced_velocity_rom)) + ', ' + str(np.mean(mean_reduced_velocity_rom)))       
           # QOIs[iSamp,:]=mean_reduced_velocity_rom[:,-1]
            QOIs= np.matmul(phi, aT[:,[-1]]).flatten()
            del mean_reduced_velocity_rom
        if QOIselector.lower()=="modal coeff":
            QOIs = aT.flatten()
            #QOIs=aT[:,-1]
        if QOIselector.lower() =="kinetic energy":
            energy = np.sum(aT, axis = 0)
            energy = energy[[-1]]
            QOIs = energy
        if QOIselector.lower() == "vorticity":
            #Compute velocity
            mean_reduced_velocity_rom = np.matmul(phi, aT[:,[-1]]).flatten()
            #Transfer to 2D
            velocity_rom_1D = np.reshape(mean_reduced_velocity_rom, (num_cell, 2), order = "F")
            velocity_rom_2D = np.zeros(vel_0_2D.shape)
            for i_dim in range(2):
                  velocity_rom_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
                    Xi, Eta, num_xi, num_eta, velocity_rom_1D[:,i_dim])
            #Add mean velocity
            velocity = velocity_rom_2D + vel_0_2D
            #Compute vorticity
            vorticity_2D = curl_calc.curl_2d(-cell_centroid[:,0,0,1], -cell_centroid[0,:,0,0],
              velocity[:, :, 0], velocity[:, :,1])
            #Map back to 1D_compact
            QOIs = arr_conv.array_2D_to_1D( Xi, Eta, num_cell,\
                                                  vorticity_2D)    
    else:
        raise Exception("More than 2 dimensions detected for boundary vector")
    return QOIs.flatten()