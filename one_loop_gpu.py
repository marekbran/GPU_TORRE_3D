import numpy as np
import time
import torch
import scipy.io
import gc


from boundary_point_source_GPU import boundary_point_source
from pulse_window_GPU import pulse_window_function_torch
from B_T_prod_GPU import B_T_prod_GPU
from B_prod_GPU import B_prod
from pcg_iteration_GPU import pcg_iteration_gpu

#from five_largest import five_largest



def one_loop(d_t, s_orbit, k, boundary_vec_1, boundary_vec_2, div_vec, p_1, p_2, p_3, n_array, t_array, gpu_extended_memory, ast_ind, R, u, I_u, orbit_ind, a_o, C, M, d_t_pml_val, I_p_x, I_p_y, I_p_z, orbit_tetra_ind, A, t_0, device, aux_vec_init):
    
    
    aux_vec = scripted_B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory)
    scripted_pcg_iteration_GPU(C,aux_vec,torch.tensor(1e-5),torch.tensor(1000),M,aux_vec_init,gpu_extended_memory)
    scripted_B_prod(u, torch.tensor(3), n_array, t_array, gpu_extended_memory)
    
    del aux_vec
    torch.cuda.empty_cache()
    gc.collect()

    
    print(f"GPU memory used: {torch.cuda.memory_allocated() / (1024 ** 3)} GB")
    
    start_time = time.time()
    i = 1
    t = d_t 
    t_shift = 0.01
    pulse_length = 1.0
    carrier_cycles_per_pulse_cycle = 5.0
    carrier_mode = 'Hann'
    time_arg = t - s_orbit[:, k] + t_shift

    pulse_window, d_pulse_window = pulse_window_function_torch(time_arg, pulse_length, carrier_cycles_per_pulse_cycle, carrier_mode,device)

    f_aux = pulse_window * boundary_vec_1[:, k] + d_pulse_window * boundary_vec_2[:, k]



    
    div_vec[ast_ind, 0] = p_1[ast_ind,0]+p_2[ast_ind,1]+p_3[ast_ind,2]

    

    

    
    aux_vec = -scripted_B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory) - (R @ u)

    
    


      

    aux_vec[orbit_ind[:,0],0] += a_o[k,0] * f_aux[:]
    aux_vec[orbit_ind[:,0],1] += a_o[k,1] * f_aux[:]
    aux_vec[orbit_ind[:,0],2] += a_o[k,2] * f_aux[:]

    

    end_time = time.time()
    aux_vec_init = scripted_pcg_iteration_GPU(C,aux_vec,torch.tensor(1e-5),torch.tensor(1000),M,aux_vec_init,gpu_extended_memory)
    end_time_1 = time.time()
    
    #del C, M, R

    

    du_dt_vec = aux_vec_init[orbit_ind[:,0],:]
    u_aux = d_t_pml_val * u[I_u[:,0],:]



    u +=  d_t * aux_vec_init
    
    u[I_u[:,0],:] = u[I_u[:,0],:] - u_aux



    
    p_aux = scripted_B_prod(u, torch.tensor(1), n_array, t_array, gpu_extended_memory)
    
    
    
    
    aux_vec = (A * p_aux).T
    p_aux = d_t_pml_val * p_1[I_p_x[:,0],:]


    
    p_1 = p_1 + d_t * aux_vec
    
    
    
    
    dp_1_dt_vec = aux_vec[orbit_tetra_ind[:,0],:]

    
    p_1[I_p_x[:,0],:] -= p_aux
    
    
    p_aux = scripted_B_prod(u, torch.tensor(2), n_array, t_array, gpu_extended_memory)
    

    
    aux_vec = (A * p_aux).T
    p_aux = d_t_pml_val * p_2[I_p_y,:]
    p_2 = p_2 + d_t * aux_vec
    dp_2_dt_vec = aux_vec[orbit_tetra_ind[:,0],:]
    p_2[I_p_y,:] -= p_aux
    


    p_aux = scripted_B_prod(u, torch.tensor(3), n_array, t_array, gpu_extended_memory)

    
    aux_vec = (A * p_aux).T
    p_aux = d_t_pml_val * p_3[I_p_z,:]
    p_3 = p_3 + d_t * aux_vec
    dp_3_dt_vec = aux_vec[orbit_tetra_ind[:,0],:]
    p_3[I_p_z,:] -= p_aux
    

    
    end_time_2 = time.time()
  
    
    print(f"Execution time up to  B_T_prod: {end_time - start_time} seconds")
    print(f"Execution time of second pcg_iteration: {end_time_1 - end_time} seconds")
    print(f"Execution time: {end_time_2 - start_time} seconds")
    
    return p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec



if __name__ == "__main__":
    mat_data = scipy.io.loadmat(r'GPU_TORRE_3D\system_data_1.mat')
    mat_data_2 = scipy.io.loadmat(r'GPU_TORRE_3D\bounndary_point_source.mat')
    mat_data_test = scipy.io.loadmat(r'GPU_TORRE_3D\system_data_1_test.mat')
    mat_data_3 = scipy.io.loadmat('GPU_TORRE_3D/pcg_iteration_gpu.mat')
    device = torch.device("cuda")
    

    source_points = mat_data_2['source_points']
    source_points = torch.from_numpy(source_points)
    source_points = source_points.to(device)
    
    orbit_triangles = mat_data_2['orbit_triangles']
    orbit_triangles = torch.from_numpy(orbit_triangles)
    orbit_triangles = orbit_triangles.to(device)
    
    orbit_nodes = mat_data_2['orbit_nodes']
    orbit_nodes = torch.from_numpy(orbit_nodes)
    orbit_nodes = orbit_nodes.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    A = mat_data['A']
    
    diagonal_elements = A.diagonal() 
    reciprocal_diagonal = 1.0 / diagonal_elements
    new_A_value = reciprocal_diagonal.astype(np.float32)
    A = new_A_value
    
    A = torch.from_numpy(A).to(device)
    
    
    coo_matrix = mat_data['C']
    a = coo_matrix.row
    b = coo_matrix.col
    indices = np.array([a, b])


    C = torch.sparse_coo_tensor(indices, coo_matrix.data, coo_matrix.shape, device=device)
    
    
    d_t = 1e-5
    s_orbit = orbit_nodes
    n_array = torch.from_numpy(mat_data['n_array_cpu']).to(device)
    t_array = torch.from_numpy(mat_data['t_array_cpu']).to(device)
    u = torch.from_numpy(mat_data['u']).to(device)
    coo_matrix = mat_data['R']
    a = coo_matrix.row
    b = coo_matrix.col
    indices = np.array([a, b])


    R = torch.sparse_coo_tensor(indices, coo_matrix.data, coo_matrix.shape, device=device)
    R = R.to(torch.double)
    
    I_u = torch.from_numpy(mat_data['I_u']).to(device)
    I_p_x = torch.from_numpy(mat_data['I_p_x']).to(device)
    I_p_y = torch.from_numpy(mat_data['I_p_y']).to(device)
    I_p_z = torch.from_numpy(mat_data['I_p_z']).to(device)
    orbit_ind = torch.from_numpy(mat_data['orbit_ind']).to(device)
    orbit_tetra_ind = torch.from_numpy(mat_data['orbit_tetra_ind']).to(device)
    a_o = torch.from_numpy(mat_data['a_o']).to(device)
    M = torch.from_numpy(mat_data['M_cpu']).to(device)
    ast_ind = torch.from_numpy(mat_data['ast_ind']).to(device)
    #j_ind_ast = torch.from_numpy(mat_data['j_ind_ast']).to(device)

    r_1 = mat_data['r_1_cpu']
    r_2 = mat_data['r_2_cpu']
    r_3 = mat_data['r_3_cpu']
    #r_u = torch.from_numpy(mat_data['r_u_cpu']).to(device)
    r_mat_vec = mat_data_test['test_mat_vec']
    
    #r_mat_vec_t = torch.from_numpy(r_mat_vec).to(device)
    
    orbit_ind = orbit_ind.to(torch.long)
    
    du_dt_vec = torch.zeros((orbit_ind.shape[0], 3))
    t_vec = torch.zeros(orbit_ind.shape[0])
    data_param = 4
    k = 1
    gpu_extended_memory = torch.tensor(3)
    d_t_pml_val = np.float32(7)
    I_u = I_u - 1
    I_p_x = I_p_x - 1
    I_p_y = I_p_y - 1
    I_p_z = I_p_z - 1
    orbit_ind = orbit_ind - 1
    orbit_tetra_ind = orbit_tetra_ind - 1
    ast_ind = ast_ind - 1
    #j_ind_ast = j_ind_ast.to(torch.int32)
    #j_ind_ast = j_ind_ast - 1
    t_array = t_array - 1
    
    (boundary_vec_1, boundary_vec_2) = boundary_point_source(source_points, orbit_triangles, orbit_nodes)
    
    
    del source_points, orbit_triangles, orbit_nodes
    torch.cuda.empty_cache()
    gc.collect()
    
    
    boundary_vec_1 = boundary_vec_1.to(device)
    boundary_vec_2 = boundary_vec_2.to(device)
    
    
    
    
    
    
    
    data_ind = -1
    time_series_ind = 0
    t_0 = d_t



    
    div_vec = torch.zeros((A.shape[0], 1), dtype=torch.float32, device=device)
    aux_init = torch.zeros(u.shape, dtype=torch.double, device=device) # u.dtype is used to match the original tensor's type

    p_1 = torch.zeros((A.shape[0], 3), dtype=torch.float32, device=device)
    p_2 = torch.zeros((A.shape[0], 3), dtype=torch.float32, device=device)
    p_3 = torch.zeros((A.shape[0], 3), dtype=torch.float32, device=device)
    

    scripted_B_T_prod = torch.jit.script(B_T_prod_GPU)
    scripted_B_prod = torch.jit.script(B_prod)
    
    scripted_pcg_iteration_GPU = torch.jit.script(pcg_iteration_gpu)
    
    
    
    
    p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec = one_loop(d_t, s_orbit, k, boundary_vec_1, boundary_vec_2, div_vec, p_1, p_2, p_3, n_array, t_array, gpu_extended_memory, ast_ind, R, u, I_u, orbit_ind, a_o, C,  M,  d_t_pml_val, I_p_x, I_p_y, I_p_z, orbit_tetra_ind, A,  t_0, device, aux_init)
    
    
    
    
    
    p_1 = p_1.cpu().numpy()
    p_2 = p_2.cpu().numpy()
    p_3 = p_3.cpu().numpy()
    
    p_1_norm = np.linalg.norm(p_1 - r_1)
    r_1_norm = np.linalg.norm(r_1)
    p_1_only_norm = np.linalg.norm(p_1)
    
    print(f"p_1_norm: {p_1_norm}")
    print(f"r_1_norm: {r_1_norm}")
    print(f"p_1_only_norm: {p_1_only_norm}")
    print(p_1_norm / r_1_norm)
    print("\n")
    
    
    
    
    
    