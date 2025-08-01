import pyopencl as cl
import numpy as np
import time
from boundary_point_source import boundary_point_source
import scipy.io
from pulse_window_function import pulse_window_function
from B_T_prod import B_T_prod
from B_prod import B_prod
from pcg_iteration_gpu import pcg_iteration_gpu
#from five_largest import five_largest



def one_loop(d_t, s_orbit, k, boundary_vec_1, boundary_vec_2, div_vec, p_1, p_2, p_3, n_array, t_array, gpu_extended_memory, ast_ind, R, u, I_u, orbit_ind, a_o, C,  M, aux_vec_init, d_t_pml_val, I_p_x, I_p_y, I_p_z, orbit_tetra_ind, A, r_f, r_aux, r_u, r_aux_init, t_0, i, r_1, r_2, r_3, r_dp_1_dt_vec, r_du_dt_vec):

    t = d_t 
    t_shift = 0.01
    pulse_length = 1.0
    carrier_cycles_per_pulse_cycle = 5.0
    carrier_mode = 'Hann'
    time_arg = t - s_orbit[:, k] + t_shift

    diagonal_elements = A.diagonal() 
    reciprocal_diagonal = 1.0 / diagonal_elements
    new_A_value = reciprocal_diagonal.astype(np.float32)
    A = new_A_value

    pulse_window, d_pulse_window = pulse_window_function(time_arg, pulse_length, carrier_cycles_per_pulse_cycle, carrier_mode)
    f_aux = pulse_window * boundary_vec_1[:, k] + d_pulse_window * boundary_vec_2[:, k]




    div_vec[ast_ind, 0] = p_1[ast_ind,0]+p_2[ast_ind,1]+p_3[ast_ind,2]

    test_mat_vec = R * u

    aux_vec = -B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory)
    aux_vec -=  test_mat_vec

    

    aux_vec[orbit_ind[:,0],0] += a_o[k,0] * f_aux[:]
    aux_vec[orbit_ind[:,0],1] += a_o[k,1] * f_aux[:]
    aux_vec[orbit_ind[:,0],2] += a_o[k,2] * f_aux[:]







    


    aux_vec_init = pcg_iteration_gpu(C,aux_vec,1e-5,1000,M,aux_vec_init,gpu_extended_memory)

    test_aux_vec_init = aux_vec_init

    du_dt_vec = aux_vec_init[orbit_ind[:,0],:]
    u_aux = d_t_pml_val * u[I_u[:,0],:]



    u +=  d_t * aux_vec_init
    
    u[I_u[:,0],:] = u[I_u[:,0],:] - u_aux




    p_aux = B_prod(u, 1, n_array, t_array, gpu_extended_memory).astype(np.float32)


    aux_vec = (A * p_aux).T
    p_aux = d_t_pml_val * p_1[I_p_x[:,0],:]



    p_1 = p_1 + d_t * aux_vec
    dp_1_dt_vec = aux_vec[orbit_tetra_ind[:,0],:]
    print(dp_1_dt_vec.shape)
    
    p_1[I_p_x[:,0],:] -= p_aux





    p_aux = B_prod(u, 2, n_array, t_array, gpu_extended_memory).astype(np.float32)
    aux_vec = (A * p_aux).T
    p_aux = d_t_pml_val * p_2[I_p_y,:]
    p_2 = p_2 + d_t * aux_vec
    dp_2_dt_vec = aux_vec[orbit_tetra_ind[:,0],:]
    p_2[I_p_y,:] -= p_aux
    p_2 = p_2.astype(np.float32)


    p_aux = B_prod(u, 3, n_array, t_array, gpu_extended_memory).astype(np.float32)  
    aux_vec = (A * p_aux).T
    p_aux = d_t_pml_val * p_3[I_p_z,:]
    p_3 = p_3 + d_t * aux_vec
    dp_3_dt_vec = aux_vec[orbit_tetra_ind[:,0],:]
    p_3[I_p_z,:] -= p_aux
    p_3 = p_3.astype(np.float32)


    """

    dp_1_dt_vec_norm = np.linalg.norm(dp_1_dt_vec - r_dp_1_dt_vec)
    r_dp_1_dt_vec_norm = np.linalg.norm(r_dp_1_dt_vec)
    dp_1_dt_vec_only_norm = np.linalg.norm(dp_1_dt_vec)

    print(f"dp_1_dt_vec_norm: {dp_1_dt_vec_norm:.40f}")
    print(f"r_dp_1_dt_vec_norm: {r_dp_1_dt_vec_norm:.40f}")
    print(f"dp_1_dt_vec_only_norm: {dp_1_dt_vec_only_norm:.40f}")
    print(dp_1_dt_vec_norm / r_dp_1_dt_vec_norm)
    print("\n")

    r_du_dt_vec_norm = np.linalg.norm(r_du_dt_vec)
    du_dt_vec_norm = np.linalg.norm(du_dt_vec - r_du_dt_vec)
    du_dt_vec_only_norm = np.linalg.norm(du_dt_vec)

    print(f"du_dt_vec_norm: {du_dt_vec_norm:.40f}")
    print(f"r_du_dt_vec_norm: {r_du_dt_vec_norm:.40f}")
    print(f"du_dt_vec_only_norm: {du_dt_vec_only_norm:.40f}")
    print(du_dt_vec_norm / r_du_dt_vec_norm)
    print("\n")



    



    print(p_1.dtype)

    print(r_1.dtype) 

    print(f"p_1 : {p_1[2,0]:.40f}")

    print(f"r_1 : {r_1[2,0]:.40f}")

    mask = (p_1 != r_1)
    print(f"Number of differing elements: {np.sum(mask)}")
    print(mask)

    p_norm = np.linalg.norm(p_1 - r_1) 
    p_mat_norm = np.linalg.norm(r_1)
    p_1_norm = np.linalg.norm(p_1)


    print(f"Norm of p_1 - r_1: {p_norm:.40f}")
    print(f"Norm of r_1: {p_mat_norm:.40f}")
    print(f"Norm of p_1: {p_1_norm:.40f}")
    print(p_norm / p_mat_norm)
    print("\n")

    p_2_norm = np.linalg.norm(p_2 - r_2)
    p_2_mat_norm = np.linalg.norm(r_2)
    print(f"Norm of p_2 - r_2: {p_2_norm:.40f}")    
    print(f"Norm of r_2: {p_2_mat_norm:.40f}")    
    print(f"Norm of p_2: {np.linalg.norm(p_2):.40f}")    
    print(p_2_norm / p_2_mat_norm)
    print("\n")

    p_3_norm = np.linalg.norm(p_3 - r_3)
    p_3_mat_norm = np.linalg.norm(r_3)
    print(f"Norm of p_3 - r_3: {p_3_norm:.40f}")    
    print(f"Norm of r_3: {p_3_mat_norm:.40f}")    
    print(f"Norm of p_3: {np.linalg.norm(p_3):.40f}")    
    print(p_3_norm / p_3_mat_norm)
    print("\n")

    u_norm = np.linalg.norm(u - r_u)
    u_mat_norm = np.linalg.norm(r_u)    
    print(f"Norm of u - r_u: {u_norm:.40f}")    
    print(f"Norm of r_u: {u_mat_norm:.40f}")    
    print(f"Norm of u: {np.linalg.norm(u):.40f}")    
    print(u_norm / u_mat_norm)
    print("\n")


    

    
    
    line_1_norm = np.linalg.norm(line_1 - l_1)
    line_2_norm = np.linalg.norm(line_2 - l_2)
    line_3_norm = np.linalg.norm(line_3 - l_3)
    l_1_norm = np.linalg.norm(l_1)
    l_2_norm = np.linalg.norm(l_2)
    l_3_norm = np.linalg.norm(l_3)
    line_1_only_norm = np.linalg.norm(line_1)
    line_2_only_norm = np.linalg.norm(line_2)
    line_3_only_norm = np.linalg.norm(line_3)
    print(f"Norm of line_1 - l_1: {line_1_norm:.40f}")    
    print(f"Norm of line_2 - l_2: {line_2_norm:.40f}")    
    print(f"Norm of line_3 - l_3: {line_3_norm:.40f}")  
    print(f"Norm of l_1: {l_1_norm:.40f}")
    print(f"Norm of l_2: {l_2_norm:.40f}")
    print(f"Norm of l_3: {l_3_norm:.40f}") 
    print(f"Norm of line_1: {line_1_only_norm:.40f}")
    print(f"Norm of line_2: {line_2_only_norm:.40f}")
    print(f"Norm of line_3: {line_3_only_norm:.40f}") 
    print(line_1_norm / np.linalg.norm(l_1))
    print(line_2_norm / np.linalg.norm(l_2))
    print(line_3_norm / np.linalg.norm(l_3))
    """

    return p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec


