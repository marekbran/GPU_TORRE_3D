import pyopencl as cl
import numpy as np
import time
import sys
import scipy.io
import torch


"""
The funciton B_prod computes the product of a vector u with a tensor defined by the entries of a 3D structure.
"""


def B_prod(u,entry_ind,n,t,gpu_extended_memory):
    try:
        start_time = time.time()
        #u = u.astype(np.float32)
        v_ind = np.array([[0, 1, 2, 3], [1, 2, 0, 3], [2, 3, 0, 1], [3, 0, 2, 1]])
        v_ind = torch.from_numpy(v_ind).to(u.device)


        if entry_ind == 1:
            entry_ind_vec = np.array([1, 2])
        elif entry_ind == 2:
            entry_ind_vec = np.array([2, 0])
        elif entry_ind == 3:
            entry_ind_vec = np.array([0, 1])
        
        u = u / 6
        entry_ind_vec = torch.tensor(entry_ind_vec, device=device)


        for i in range(4):
            row_indices = t[:, v_ind[i, 1]]  #parallelise -1
            aux_vec = n[row_indices[:, None], entry_ind_vec]

            row_indices = t[:, v_ind[i, 3]] 
            v1= n[row_indices[:, None], entry_ind_vec]
            v1 = v1 - aux_vec

            row_indices = t[:, v_ind[i, 2]] 
            v2= n[row_indices[:, None], entry_ind_vec]
            v2 = v2 - aux_vec

            aux_vec = v1[:, 0] * v2[:, 1]
            aux_vec = aux_vec - v1[:, 1] * v2[:, 0]



            for k in range(3):
                row_indices = t[:, i] 
                u_perm = u[row_indices, k]




                if i == 0: 
                    if k == 0:
                        p_1 = u_perm * aux_vec
                    elif k == 1:
                        p_2 = u_perm * aux_vec
                    elif k == 2:
                        p_3 = u_perm * aux_vec
                

                else:
                    if k == 0:
                        p_1 = p_1 + u_perm * aux_vec
                    elif k == 1:
                        p_2 = p_2 + u_perm * aux_vec
                    elif k == 2:
                        p_3 = p_3 + u_perm * aux_vec
        if gpu_extended_memory == 0 or gpu_extended_memory == 1:
            p = np.zeros((n.shape[0], 3))
            p[:, 0] = p_1
            p[:, 1] = p_2            
            p[:, 2] = p_3
        else:
            p = torch.stack([p_1, p_2, p_3])

        
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return p  
        


    except cl.LogicError as e:
        status_message = f"OpenCL LogicError: {e}. Ensure OpenCL drivers and device are configured."
        error_occurred = True
        print(f"Python Error (OpenCL): {status_message}", file=sys.stderr) # Print to stderr for MATLAB
    except Exception as e:
        status_message = f"An unexpected error occurred: {e}"
        error_occurred = True
        print(f"Python Error: {status_message}", file=sys.stderr) # Print to stderr for MATLAB





if __name__ == "__main__":
    mat_data = scipy.io.loadmat('GPU_TORRE_3D\B_prod.mat')
    device = torch.device("cuda")
    n = mat_data['n_array_cpu']
    n = torch.from_numpy(n)
    n = n.to(device)
    
    
    u = mat_data['u']
    u = u.astype(np.float32)
    u = torch.from_numpy(u)
    u = u.to(device)
    
    
    t = mat_data['t_array_cpu']
    t = torch.from_numpy(t)
    t = t.to(device)
    t = t-1
    
    
    p_mat = mat_data['p_cpu']




    
    p = B_prod(u, 3, n, t, 3)
    p = p.cpu().numpy()  # Move to CPU and convert to numpy array
    
    

    p = p.T
    p_norm = np.linalg.norm(p - p_mat)
    p_mat_norm = np.linalg.norm(p_mat)
    p_only_norm = np.linalg.norm(p)
    print(f"p_norm: {p_norm}")
    print(f"p_mat_norm: {p_mat_norm}")
    print(f"p_only_norm: {p_only_norm}")
    print(p_norm / p_mat_norm)



   
