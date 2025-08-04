import pyopencl as cl
import numpy as np
import time
import sys
import scipy.io
import math

def pcg_iteration_gpu(A,b,tol_val,max_it,M,x,gpu_extended_memory):

    r = b - A @ x
    z = M *r
    p = z
    j = 1


    conv_val = np.sqrt(np.max(np.sum(r**2, axis=0) / np.sum(b**2, axis=0)))
    
    while (conv_val > tol_val) & (j < max_it):
        Ap = A @ p
        alpha = np.sum(r * z) / np.sum(p * Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        z_new = M * r_new
        beta = np.sum(r_new * z_new) / np.sum(r * z)
        p = z_new + beta * p
        r = r_new
        z = z_new

        conv_val = np.sqrt(np.max(np.sum(r**2, axis=0) / np.sum(b**2, axis=0)))
        j += 1

    n_iter = j 

    return x



if __name__ == "__main__":

    mat_data = scipy.io.loadmat('pcg_iteration_gpu.mat')

    A = mat_data['A']
    b = mat_data['b']
    M = mat_data['M']
    x_mat = mat_data['x_cpu']
    x = np.zeros(b.shape)  # Initialize x as a zero vector
    tol_val =1e-5
    max_it = 1000
    conv_val = mat_data['conv_val_cpu']
    n_iter = mat_data['n_iter']

    x = pcg_iteration_gpu(A, b, tol_val, max_it, M, x, 3)

    