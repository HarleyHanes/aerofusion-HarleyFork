import numpy as np
import findiff

def curl_2d (x,y,u,v):
  d_dy = findiff.FinDiff(0, y, acc=10)
  d_dx = findiff.FinDiff(1, x, acc=10)
  dv_dx = d_dx(v)
  du_dy = d_dy(u)
  curl_2d = dv_dx - du_dy
  # (nul, nul, D) = Derivative_Calc.cheb_derivative(velocity)
  # curl_2d = np.matmul(-v , D.transpose()) - np.matmul(D, u)
  return (curl_2d)
