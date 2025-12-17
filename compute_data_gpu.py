import numpy as np
import time
import torch
import scipy.io
import gc
import math
import os

from boundary_point_source_GPU import boundary_point_source
from pulse_window_GPU import pulse_window_function_torch
from B_T_prod_GPU import B_T_prod_GPU
from B_prod_GPU import B_prod
from pcg_iteration_GPU import pcg_iteration_gpu
from surface_integral_gpu import surface_integral






if __name__ == "__main__":
    mat_data = scipy.io.loadmat(r'/media/datadisk/adam/forward_scripts/system_data_1.mat')
    mat_data_2 = scipy.io.loadmat(r'/media/datadisk/adam/forward_scripts/bounndary_point_source.mat')
    mat_data_test = scipy.io.loadmat(r'/media/datadisk/adam/forward_scripts/system_data_1_test.mat')
    mat_data_3 = scipy.io.loadmat('/media/datadisk/adam/forward_scripts/pcg_iteration_gpu.mat')
    device = torch.device("cuda")

    source_points = mat_data_2['source_points']
    source_points = torch.from_numpy(source_points)
    source_points = source_points.to(device)

    orbit_triangles = mat_data_2['orbit_triangles']
    orbit_triangles = orbit_triangles.astype(np.float32)
    orbit_triangles = torch.from_numpy(orbit_triangles)
    orbit_triangles = orbit_triangles.to(device)

    orbit_nodes = mat_data_2['orbit_nodes']
    orbit_nodes = torch.from_numpy(orbit_nodes)
    orbit_nodes = orbit_nodes.to(device)


    A = mat_data['A']


    diagonal_elements = A.diagonal()
    reciprocal_diagonal = 1.0 / diagonal_elements
    new_A_value = reciprocal_diagonal.astype(np.float32)
    A = new_A_value

    A = torch.from_numpy(A).to(device)

    sparse_matrix = mat_data['C']

    coo_matrix = sparse_matrix.tocoo()
    a = coo_matrix.row
    b = coo_matrix.col
    indices = np.array([a, b])


    C = torch.sparse_coo_tensor(indices, coo_matrix.data, coo_matrix.shape, device=device)
    # start time taken from here
    d_t = 1e-5
    s_orbit = orbit_nodes

    # Load arrays
    n_array = torch.from_numpy(mat_data['n_array_cpu']).to(device)
    t_array = torch.from_numpy(mat_data['t_array_cpu']).to(device)

    # 1. CHANGE: Cast initial 'u' to complex128 (since aux_vec_init was double)
    u = torch.from_numpy(mat_data['u']).to(device)
    u = u.to(dtype=torch.complex128)

    # R Matrix Setup (Keep as double/real, PyTorch handles Real @ Complex multiplication)
    sparse_matrix = mat_data['R']
    coo_matrix = sparse_matrix.tocoo()
    a = coo_matrix.row
    b = coo_matrix.col
    indices = np.array([a, b])
    R = torch.sparse_coo_tensor(indices, coo_matrix.data, coo_matrix.shape, device=device)
    R = R.to(torch.double) # R stays real to save memory

    # Indices loading (Keep as integer/long)
    I_u = torch.from_numpy(mat_data['I_u']).to(device).to(torch.long)
    I_p_x = torch.from_numpy(mat_data['I_p_x']).to(device).to(torch.long)
    I_p_y = torch.from_numpy(mat_data['I_p_y']).to(device).to(torch.long)
    I_p_z = torch.from_numpy(mat_data['I_p_z']).to(device).to(torch.long)
    orbit_ind = torch.from_numpy(mat_data['orbit_ind']).to(device)
    orbit_tetra_ind = torch.from_numpy(mat_data['orbit_tetra_ind']).to(device).to(torch.long)

    # 2. CHANGE: Source orientation 'a_o' might need to be complex if sources have phase
    a_o = torch.from_numpy(mat_data['a_o']).to(device)
    a_o = a_o.to(dtype=torch.complex64)

    M = torch.from_numpy(mat_data['M_cpu']).to(device)
    ast_ind = torch.from_numpy(mat_data['ast_ind']).to(device).to(torch.long)
    j_ind_ast = mat_data['j_ind_ast'].astype(np.float32)
    j_ind_ast = torch.from_numpy(j_ind_ast).to(device)

    # 0-based indexing adjustments
    orbit_ind = orbit_ind.to(torch.long)
    orbit_ind = orbit_ind - 1
    I_u = I_u - 1
    I_p_x = I_p_x - 1
    I_p_y = I_p_y - 1
    I_p_z = I_p_z - 1
    orbit_tetra_ind = orbit_tetra_ind - 1
    ast_ind = ast_ind - 1
    j_ind_ast = j_ind_ast.to(torch.long)
    j_ind_ast = j_ind_ast - 1
    t_array = t_array - 1

    # 3. CHANGE: Initialization of time-stepping vectors to COMPLEX types
    # p vectors were float32 -> now complex64
    # u/aux vectors were double -> now complex128

    du_dt_vec = torch.zeros((orbit_ind.shape[0], 3), dtype=torch.complex128, device=device)
    t_vec = torch.zeros(60001)
    data_param = 80
    k = 1
    gpu_extended_memory = torch.tensor(3)
    d_t_pml_val = np.float32(7)

    # 4. CHANGE: Data storage matrices must be complex to store the results
    num_steps = math.ceil(len(t_vec) / data_param)

    t_data = torch.zeros(num_steps) # Time is still real

    # Field data storage -> Complex64 (to save space) or Complex128
    u_data = torch.zeros((orbit_ind.shape[0], num_steps), dtype=torch.complex64)
    du_dt_data = torch.zeros((orbit_ind.shape[0], num_steps), dtype=torch.complex64)

    p_1_data = torch.zeros((orbit_tetra_ind.shape[0], num_steps), dtype=torch.complex64)
    p_2_data = torch.zeros((orbit_tetra_ind.shape[0], num_steps), dtype=torch.complex64)
    p_3_data = torch.zeros((orbit_tetra_ind.shape[0], num_steps), dtype=torch.complex64)

    u_data_mat = torch.zeros((j_ind_ast.shape[0], num_steps), dtype=torch.complex64)
    f_data_mat = torch.zeros((j_ind_ast.shape[0], num_steps), dtype=torch.complex64)

    # Boundary vectors
    (boundary_vec_1, boundary_vec_2) = boundary_point_source(source_points, orbit_triangles, orbit_nodes)

    # If boundary vectors contain phase info, cast them too
    boundary_vec_1 = boundary_vec_1.to(device).to(dtype=torch.complex64)
    boundary_vec_2 = boundary_vec_2.to(device).to(dtype=torch.complex64)

    data_ind = -1
    time_series_ind = 0
    t_0 = 0
    start_time = time.time()

    if time_series_ind == 0:
        # 5. CHANGE: Solver variables initialized as COMPLEX
        # div_vec matches p vectors (complex64)
        div_vec = torch.zeros((A.shape[0], 1), dtype=torch.complex64, device=device)

        # aux_vec_init matches u (complex128)
        aux_vec_init = torch.zeros(u.shape, dtype=torch.complex128, device=device)

        # p vectors (complex64)
        p_1 = torch.zeros((A.shape[0], 3), dtype=torch.complex64, device=device)
        p_2 = torch.zeros((A.shape[0], 3), dtype=torch.complex64, device=device)
        p_3 = torch.zeros((A.shape[0], 3), dtype=torch.complex64, device=device)

        scripted_B_T_prod = torch.jit.script(B_T_prod_GPU)
        scripted_B_prod = torch.jit.script(B_prod)
        scripted_pcg_iteration_GPU = torch.jit.script(pcg_iteration_gpu)

        # First run checks
        aux_vec = scripted_B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory)

        # Ensure the zero vector passed to PCG matches the type of aux_vec (complex128)
        # Note: aux_vec might be complex64 coming from p vectors, but PCG expects higher precision
        aux_vec = aux_vec.to(dtype=torch.complex128)

        scripted_pcg_iteration_GPU(C, aux_vec, torch.tensor(1e-5), torch.tensor(1000), M, aux_vec_init, gpu_extended_memory)
        scripted_B_prod(u, torch.tensor(3), n_array, t_array, gpu_extended_memory)

        del aux_vec
        torch.cuda.empty_cache()
        gc.collect()

    for i in range(1, 6): #changed form len(t_vec) to 6

        t = t_0 + (i-1)*d_t
        t_shift = 0.01
        pulse_length = 1.0
        carrier_cycles_per_pulse_cycle = 5.0
        carrier_mode = 'Hann'
        time_arg = t - s_orbit[:, k] + t_shift

        pulse_window, d_pulse_window = pulse_window_function_torch(time_arg, pulse_length, carrier_cycles_per_pulse_cycle, carrier_mode,device)

        f_aux = pulse_window * boundary_vec_1[:, k] + d_pulse_window * boundary_vec_2[:, k]




        div_vec = div_vec.to(p_1.dtype)
        div_vec[ast_ind, 0] = p_1[ast_ind,0]+p_2[ast_ind,1]+p_3[ast_ind,2]






        aux_vec = -scripted_B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory)



        aux_vec -= (R @ u)





        aux_vec[orbit_ind[:,0],0] += a_o[k,0] * f_aux[:]
        aux_vec[orbit_ind[:,0],1] += a_o[k,1] * f_aux[:]
        aux_vec[orbit_ind[:,0],2] += a_o[k,2] * f_aux[:]



        aux_vec_init = scripted_pcg_iteration_GPU(C,aux_vec,torch.tensor(1e-5),torch.tensor(1000),M,aux_vec_init,gpu_extended_memory)





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






        if (i-1) % data_param == 0: # changed form  (i-1) % data_param == 0 to i == 3
            data_ind += 1
            t_data[data_ind] = t


            u_data[:, data_ind] = u[orbit_ind[:,0],0].cpu()*a_o[k,0].cpu() + u[orbit_ind[:,0],1].cpu()*a_o[k,1].cpu() + u[orbit_ind[:,0],2].cpu()*a_o[k,2].cpu()
            du_dt_data[:, data_ind] = du_dt_vec[:,0].cpu()*a_o[k,0].cpu() + du_dt_vec[:,1].cpu()*a_o[k,1].cpu() + du_dt_vec[:,2].cpu()*a_o[k,2].cpu()
            p_1_data[:, data_ind] = dp_1_dt_vec[:,0].cpu()*a_o[k,0].cpu() + dp_1_dt_vec[:,1].cpu()*a_o[k,1].cpu() + dp_1_dt_vec[:,2].cpu()*a_o[k,2].cpu()
            p_2_data[:, data_ind] = dp_2_dt_vec[:,0].cpu()*a_o[k,0].cpu() + dp_2_dt_vec[:,1].cpu()*a_o[k,1].cpu() + dp_2_dt_vec[:,2].cpu()*a_o[k,2].cpu()
            p_3_data[:, data_ind] = dp_3_dt_vec[:,0].cpu()*a_o[k,0].cpu() + dp_3_dt_vec[:,1].cpu()*a_o[k,1].cpu() + dp_3_dt_vec[:,2].cpu()*a_o[k,2].cpu()
            u_data_mat[:, data_ind] = u[j_ind_ast[:,0],0].cpu()*a_o[k,0].cpu() + u[j_ind_ast[:,0],1].cpu()*a_o[k,1].cpu() + u[j_ind_ast[:,0],2].cpu()*a_o[k,2].cpu()



            div_vec = div_vec.to(p_1.dtype)
            div_vec[ast_ind[:,0], 0] = p_1[ast_ind[:,0],0]+p_2[ast_ind[:,0],1]+p_3[ast_ind[:,0],2]



            aux_vec = B_T_prod_GPU(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory) + R @ u
            aux_vec = pcg_iteration_gpu(C,aux_vec,1e-5,1000,M,torch.zeros(u.shape, dtype=torch.double, device=device),gpu_extended_memory)
            du_dt_vec_ast = aux_vec[j_ind_ast[:,0],:]
            f_data_mat[:,data_ind] = du_dt_vec_ast[:,0].cpu()*a_o[k,0].cpu() + du_dt_vec_ast[:,1].cpu()*a_o[k,1].cpu() + du_dt_vec_ast[:,2].cpu()*a_o[k,2].cpu()

    os.makedirs(f'python_scripts/data_1', exist_ok=True)

    if k == 1:
        rec_data, test = surface_integral(u_data, du_dt_data, p_1_data, p_2_data, p_3_data, t_data, t_shift, source_points, orbit_nodes, orbit_triangles, device)

        rec_data = rec_data.cpu().numpy()

        scipy.io.savemat(f'python_scripts/data_1/computed_data_gpu_1.mat', {'u_data': u_data.numpy(), 'du_dt_data': du_dt_data.numpy(), 'p_1_data': p_1_data.numpy(), 'p_2_data': p_2_data.numpy(), 'p_3_data': p_3_data.numpy(), 't_data': t_data.numpy(), 'f_data_mat': f_data_mat.numpy(), 'u_data_mat': u_data_mat.numpy(), 'rec_data': rec_data})
    else:
        current_iterate = i

        if time_series_ind < k:

            scipy.io.savemat('python_scripts/data_1/point_1_field_data_{carrier_mode}.mat', {'u': u, 'aux_vec_init': aux_vec_init, 'p_1': p_1, 'p_2': p_2, 'p_3': p_3, 'current_iterate': current_iterate})
        else:
             if os.path.exists('python_scripts/data_{k}/point_{k}_field_data_{carrier_mode}.mat'):
                 os.remove('python_scripts/data_{k}/point_{k}_field_data_{carrier_mode}.mat')

        scipy.io.savemat('python_scripts/data_{k}/point_1_data_{carrier_mode}_{time_series_id}.mat', {
    'u_data': u_data,
    'du_dt_data': du_dt_data,
    'u_data_mat': u_data_mat,
    'f_data_mat': f_data_mat,
    'p_1_data': p_1_data,
    'p_2_data': p_2_data,
    'p_3_data': p_3_data,
    't_data': t_data,
    'pulse_length': pulse_length,
    'carrier_cycles_per_pulse_cycle': carrier_cycles_per_pulse_cycle
})

    end_time = time.time() #copied from below
    total_time = end_time - start_time
    scipy.io.savemat(f'python_scripts/data_1/total_time.mat',{'total_time': total_time})
    print(f"Execution time: {end_time - start_time} seconds")