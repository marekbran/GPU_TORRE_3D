import pyopencl as cl
import numpy as np
import time
import sys
import scipy.io
import math
import torch

def pcg_iteration_gpu(A,b,tol_val,max_it,M,x,gpu_extended_memory):

    r = b - A @ x
    z = M *r
    p = z
    j = 1



    conv_val = torch.sqrt(torch.max(torch.sum(r**2, dim=0) / torch.sum(b**2, dim=0)))
    
    while (conv_val > tol_val) & (j < max_it):
        Ap = A @ p
        alpha = torch.sum(r * z) / torch.sum(p * Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M * r_new
        beta = torch.sum(r_new * z_new) / torch.sum(r * z)
        p = z_new + beta * p
        r = r_new
        z = z_new

        conv_val = torch.sqrt(torch.max(torch.sum(r**2, dim=0) / torch.sum(b**2, dim=0)))
        j += 1

    n_iter = j 

    return x



if __name__ == "__main__":

    mat_data = scipy.io.loadmat('GPU_TORRE_3D/pcg_iteration_gpu.mat')
    device = torch.device("cuda")

    A = mat_data['A']
    A = torch.from_numpy(A)
    A = A.to(device)
    
    b = mat_data['b']
    b = torch.from_numpy(b)
    b = b.to(device)
    
    M = mat_data['M']
    M = torch.from_numpy(M)
    M = M.to(device)
    
    x_mat = mat_data['x_cpu']
    
  # Initialize x as a zero vector
    x = torch.zeros(b.shape, dtype=torch.double, device=device) # Initialize x as a zero vector
    tol_val =1e-15
    max_it = 100000
    conv_val = mat_data['conv_val_cpu']
    n_iter = mat_data['n_iter']

    start_time = time.time()
    x = pcg_iteration_gpu(A, b, tol_val, max_it, M, x, 3)
    end_time = time.time()
    
    
    print(f"Execution time: {end_time - start_time} seconds")
 
    