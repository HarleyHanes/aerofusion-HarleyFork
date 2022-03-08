# -----------------------------------------------------------------------------
# \file pod_modes.py
# \brief Calculate POD modes
# -----------------------------------------------------------------------------
import numpy as np
from scipy.linalg import svd
from numpy.linalg import norm

# -----------------------------------------------------------------------------
# POD modes of vectorial field is found here.
# w is cell size with the dimensions of [Ncell*dim]
# The input is a 2D array of [Ncell*dim, Nt]
# a0 is the initial modal coefficients
def Find_Modes(A, w, r):

  dim = A.shape
  num_snap = dim[1]
  C = 0
  C = A.transpose()
  C = C*w/num_snap
  C = np.matmul(C,A)
  U, sigma, YT = svd(C)
  
  a0 = np.matmul(np.sqrt(np.diag(sigma[0:r])), (YT[0:r,:]*np.sqrt(num_snap)))
  phi = 0
  phi = U[:,0:r]* np.power(np.sqrt(sigma[0:r]*num_snap), -1)
  phi = np.matmul(A,phi)

  Lambda = np.sqrt(np.diag(sigma[0:r]))
  #print('A - A_ROM =', norm( A - np.matmul(phi, a0)))
  return (phi,a0, np.sqrt(sigma))

# -----------------------------------------------------------------------------
# Var is our variable of interest that we want to calculate mean over time
# Var is a 5d array [Nxi, Neta, Nzeta, VarDim, Nt]
def Find_Mean(Var):
  dim = Var.shape
  Ncell = dim[0]
  Nt = dim[1]
  Mean = np.zeros(Ncell)
  Mean = Var.mean(axis=1)
  return Mean

def find_number_of_modes (A, w, energy_criteria):
    dim = A.shape
    num_snap = dim[1]
    C = 0
    C = A.transpose()
    C = C * w / num_snap
    C = np.matmul(C, A)
    U, sigma, YT = svd(C)
    lambda_pod = np.sqrt(sigma)
    num_modes = 0
    for i_mode in range(num_snap+1):
        ratio =  (np.sum(lambda_pod[0:i_mode]))/np.sum(lambda_pod)
        if ratio >=  energy_criteria:
            num_modes = i_mode
            break

    return(num_modes)
