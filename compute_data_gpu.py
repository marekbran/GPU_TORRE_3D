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

    start_time = time.time()
    d_t = 1e-5
    s_orbit = orbit_nodes
    n_array = torch.from_numpy(mat_data['n_array_cpu']).to(device)
    t_array = torch.from_numpy(mat_data['t_array_cpu']).to(device)
    u = torch.from_numpy(mat_data['u']).to(device)
    sparse_matrix = mat_data['R']
    coo_matrix = sparse_matrix.tocoo()
    a = coo_matrix.row
    b = coo_matrix.col
    indices = np.array([a, b])


    R = torch.sparse_coo_tensor(indices, coo_matrix.data, coo_matrix.shape, device=device)
    R = R.to(torch.double)

    I_u = torch.from_numpy(mat_data['I_u']).to(device).to(torch.long)
    I_p_x = torch.from_numpy(mat_data['I_p_x']).to(device).to(torch.long)
    I_p_y = torch.from_numpy(mat_data['I_p_y']).to(device).to(torch.long)
    I_p_z = torch.from_numpy(mat_data['I_p_z']).to(device).to(torch.long)
    orbit_ind = torch.from_numpy(mat_data['orbit_ind']).to(device)
    orbit_tetra_ind = torch.from_numpy(mat_data['orbit_tetra_ind']).to(device).to(torch.long)
    a_o = torch.from_numpy(mat_data['a_o']).to(device)
    M = torch.from_numpy(mat_data['M_cpu']).to(device)
    ast_ind = torch.from_numpy(mat_data['ast_ind']).to(device).to(torch.long)
    j_ind_ast = mat_data['j_ind_ast'].astype(np.float32)
    j_ind_ast = torch.from_numpy(j_ind_ast).to(device)

    r_1 = mat_data['r_1_cpu']
    r_2 = mat_data['r_2_cpu']
    r_3 = mat_data['r_3_cpu']
    #r_u = torch.from_numpy(mat_data['r_u_cpu']).to(device)
    r_mat_vec = mat_data_test['test_mat_vec']

    #r_mat_vec_t = torch.from_numpy(r_mat_vec).to(device)

    orbit_ind = orbit_ind.to(torch.long)

    du_dt_vec = torch.zeros((orbit_ind.shape[0], 3))
    t_vec = torch.zeros(600001)
    data_param = 750
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
    j_ind_ast = j_ind_ast.to(torch.long)
    j_ind_ast = j_ind_ast - 1
    t_array = t_array - 1


    t_data = torch.zeros( math.ceil(len(t_vec) / data_param))
    u_data = torch.zeros((orbit_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    du_dt_data = torch.zeros((orbit_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    p_1_data = torch.zeros((orbit_tetra_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    p_2_data = torch.zeros((orbit_tetra_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    p_3_data = torch.zeros((orbit_tetra_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    u_data_mat = torch.zeros((j_ind_ast.shape[0], math.ceil(len(t_vec) / data_param)))
    f_data_mat = torch.zeros((j_ind_ast.shape[0], math.ceil(len(t_vec) / data_param)))

    (boundary_vec_1, boundary_vec_2) = boundary_point_source(source_points, orbit_triangles, orbit_nodes)




    boundary_vec_1 = boundary_vec_1.to(device)
    boundary_vec_2 = boundary_vec_2.to(device)


    data_ind = -1
    time_series_ind = 0
    t_0 = d_t






    if time_series_ind == 0:
        div_vec = torch.zeros((A.shape[0], 1), dtype=torch.float32, device=device)
        aux_vec_init = torch.zeros(u.shape, dtype=torch.double, device=device)

        p_1 = torch.zeros((A.shape[0], 3), dtype=torch.float32, device=device)
        p_2 = torch.zeros((A.shape[0], 3), dtype=torch.float32, device=device)
        p_3 = torch.zeros((A.shape[0], 3), dtype=torch.float32, device=device)


        scripted_B_T_prod = torch.jit.script(B_T_prod_GPU)
        scripted_B_prod = torch.jit.script(B_prod)

        scripted_pcg_iteration_GPU = torch.jit.script(pcg_iteration_gpu)

        aux_vec = scripted_B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory)
        scripted_pcg_iteration_GPU(C,aux_vec,torch.tensor(1e-5),torch.tensor(1000),M,aux_vec_init,gpu_extended_memory)
        scripted_B_prod(u, torch.tensor(3), n_array, t_array, gpu_extended_memory)

        del aux_vec
        torch.cuda.empty_cache()
        gc.collect()


    for i in range(1, len(t_vec)): #changed form len(t_vec) to 6

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






        aux_vec = -scripted_B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory) - (R @ u)







        aux_vec[orbit_ind[:,0],0] += a_o[k,0] * f_aux[:]
        aux_vec[orbit_ind[:,0],1] += a_o[k,1] * f_aux[:]
        aux_vec[orbit_ind[:,0],2] += a_o[k,2] * f_aux[:]



        end_time = time.time()
        aux_vec_init = scripted_pcg_iteration_GPU(C,aux_vec,torch.tensor(1e-5),torch.tensor(1000),M,aux_vec_init,gpu_extended_memory)
        end_time_1 = time.time()





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

    os.makedirs(f'python_scripts/data_{k}', exist_ok=True)

    if k == 1:
        rec_data = surface_integral(u_data, du_dt_data, p_1_data, p_2_data, p_3_data, t_data, t_shift, source_points, orbit_nodes, orbit_triangles, device) 


        scipy.io.savemat(f'python_scripts/data_1/computed_data_gpu_1.mat', {'u_data': u_data.numpy(), 'du_dt_data': du_dt_data.numpy(), 'p_1_data': p_1_data.numpy(), 'p_2_data': p_2_data.numpy(), 'p_3_data': p_3_data.numpy(), 't_data': t_data.numpy(), 'f_data_mat': f_data_mat.numpy(), 'u_data_mat': u_data_mat.numpy()})
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
    end_time = time.time()
    total_time = end_time - start_time
    scipy.io.savemat(f'python_scripts/data_1/total_time.mat',{'total_time': total_time})
    print(f"Execution time: {end_time - start_time} seconds")