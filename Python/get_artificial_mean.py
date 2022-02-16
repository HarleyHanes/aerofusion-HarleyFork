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

def main(basis_vort_vec, basis_orient_vec, basis_location_vec, basis_extent_vec, \
         Xi, Eta, Xi_mesh, Eta_mesh):
    
    num_dim  = 2
    num_xi   = Xi_mesh.shape[0]
    num_eta  = Xi_mesh.shape[1]
    num_zeta = 1
        
    

    num_cell = num_xi*num_eta*num_zeta
    

    
    #Make functions functions
    fMesh = np.zeros(Xi_mesh.shape)
    for i_basis in range(len(basis_vort_vec)):
            
        #Make functions functions
        fMesh = np.zeros(Xi_mesh.shape)
        locality = 200
        eddyStrength = 3
        COV0 = np.array([[1,0],[0,1]])
        COVBL1 = np.array([[1,0],[1.6,1]])
        COVBL2 = np.array([[1,.9],[.9,1]])
        
        COVTL1 = np.array([[1,0],[.5,2]])
        COVTL2 = np.array([[5,0],[0,1]])
        
        COVBR1 = np.array([[1,-.8],[-.8,1]])
        COVBR2 = np.array([[1,0],[-2,3]])
        #COVBR2 = np.array([[1,-.99],[-.99,1]])
        #Helper Function
        radius = lambda xCent, yCent, x,y, COV: np.sqrt(\
            COV[0,0]*(xCent-x)**2+(COV[0,1]+COV[1,0])*(yCent-y)*(xCent-x)+COV[1,1]*(yCent-y)**2)
        expRBF = lambda xCent, yCent, x, y, COV, locality : np.exp(-locality*radius(xCent,yCent,x,y, COV)**2)
        #invRBF = lambda xCent, yCent, x, y, locality : 1/(1+locality * radius(xCent,yCent,x,y))
        #polyRBF = lambda xCent, yCent, x, y, locality : radius
        #Center Flow
        #fCenterFlow = lambda x,y: 2-(x**2+y**2) #np.exp(-(np.abs(x)**2+ np.abs(y)**2))
        #Top Left Flow
        fBCcurve = lambda xCent, yCent, x,y: np.log((1/np.abs(xCent-x)*(1/np.abs(yCent-y)))**63)/200
        #fCenterFlow = lambda x,y: np.exp(-np.abs((x)*(y)))
            
        #Add Circular Flows
        #fMesh+=fCenterFlow(Xi_mesh, Eta_mesh)
        # #fMesh+=fTLflow(Xi_mesh,Eta_mesh)
        # fMesh += eddyStrength*invRBF(.85, -.85, Xi_mesh, Eta_mesh, locality)
        # fMesh += eddyStrength*invRBF(-.85, -.85, Xi_mesh, Eta_mesh, locality)
        # fMesh += eddyStrength*invRBF(-.85, .85, Xi_mesh, Eta_mesh, locality)
        
        #fMesh += .3*eddyStrength*expRBF(.91, -.91, Xi_mesh, Eta_mesh, COVBR2, locality/1.5)
        #fMesh += .3*eddyStrength*expRBF(.4, -.93, Xi_mesh, Eta_mesh, COVBR1, locality*3)
        
        fMesh += .8*eddyStrength*expRBF(.4, -.93, Xi_mesh, Eta_mesh, COVBR2, locality*.6)
        fMesh += 1.5*eddyStrength*expRBF(.8, -.8, Xi_mesh, Eta_mesh, COVBR1, locality*.2)
        
        # fMesh += .3*eddyStrength*expRBF(.78, -.78, Xi_mesh, Eta_mesh, locality)
        # fMesh += .3*eddyStrength*expRBF(.82, -.73, Xi_mesh, Eta_mesh, locality)
        # fMesh += .3*eddyStrength*expRBF(.73, -.82, Xi_mesh, Eta_mesh, locality)
        
        fMesh += .8*eddyStrength*expRBF(-.88, -.65, Xi_mesh, Eta_mesh, COVBL2, locality*2)
        fMesh += .8*eddyStrength*expRBF(-.85, -.85, Xi_mesh, Eta_mesh, COVBL1, locality*.4)
        
        fMesh += 4.6*eddyStrength*expRBF(-.8, .8, Xi_mesh, Eta_mesh, COVTL1, locality*.2)
        fMesh +=  1.2*eddyStrength*expRBF(-.95, .59, Xi_mesh, Eta_mesh, COVTL2, locality)
        #Add Boundary Flows
        #fMesh+=fBCcurve(1,1,Xi_mesh, Eta_mesh)
        #fMesh+=fBCcurve(1,-1,Xi_mesh, Eta_mesh)
        #fMesh+=fBCcurve(-1,1,Xi_mesh, Eta_mesh)
        #fMesh+=fBCcurve(-1,-1,Xi_mesh, Eta_mesh)
    #-----------------------------------Convert to Velocity-----------------------
    v = - np.gradient(fMesh, axis = 1)
    u = np.gradient(fMesh, axis = 0)
    
    #------------------------Get velocity_1D_compact
    #Convert to 2D
    velocity_2D = np.empty(u.shape + (2,))
    velocity_2D[:,:,0] = u
    velocity_2D[:,:,1] = v
    #Convert to 1D
    velocity_1D=np.empty((num_cell,2))
    for i in range(2):
        velocity_1D[:,i] = arr_conv.array_2D_to_1D(\
                       Xi, Eta, num_cell, velocity_2D[:,:,i])
    velocity_1D_compact = np.reshape(velocity_1D, (num_cell*num_dim), order = 'F')
    
    #Convert back to mesh for sanity check
    check_velocity_1D = np.reshape(velocity_1D_compact, (num_cell, num_dim), order = 'F')
    check_velocity_2D = np.empty(velocity_2D.shape)
    for i in range(2):
        check_velocity_2D[:,:,i] = arr_conv.array_1D_to_2D(\
                                Xi, Eta, num_xi, num_eta, check_velocity_1D[:,i])
    if np.max(check_velocity_2D-velocity_2D)>1e-10:
        raise(Exception("Array Conversion flawed"))

    return(velocity_1D_compact, velocity_1D, velocity_2D)

if __name__ == '__main__':
    main()

