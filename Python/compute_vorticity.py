# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:19:50 2022

@author: USER
"""
import numpy as np

def Vorticity_2D(velocity_2D):
    u=velocity_2D[:,:,0]
    v=velocity_2D[:,:,1]
    #Compute dv/dx
    dvdx = np.gradient(v, axis = 0)
    #Comput du/dy
    dudy = np.gradient(u, axis = 1)
    
    vorticity = dvdx-dudy
    
    return vorticity