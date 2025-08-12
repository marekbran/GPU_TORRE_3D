import pyopencl as cl
import numpy as np
import time
from boundary_point_source import boundary_point_source
import scipy.io
from pulse_window_function import pulse_window_function
from GPU_TORRE_3D.B_T_prod_GPU import B_T_prod
from B_prod import B_prod
from GPU_TORRE_3D.pcg_iteration import pcg_iteration_gpu
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


    return p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec


