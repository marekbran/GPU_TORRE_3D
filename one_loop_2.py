import pyopencl as cl
import numpy as np
import time
from boundary_point_source import boundary_point_source
import scipy.io
from pulse_window_function import pulse_window_function
from B_T_prod import B_T_prod
from B_prod import B_prod
from pcg_iteration_gpu import pcg_iteration_gpu
from five_largest import five_largest
from one_loop import one_loop
import math





if __name__ == "__main__":
    mat_data = scipy.io.loadmat('system_data_1.mat')
    mat_data_test = scipy.io.loadmat('system_data_1_test.mat')
    mat_data_2 = scipy.io.loadmat('bounndary_point_source.mat')
    mat_data_full = scipy.io.loadmat('BIGger_loop.mat')


    source_points = mat_data_2['source_points']
    orbit_triangles = mat_data_2['orbit_triangles']
    orbit_nodes = mat_data_2['orbit_nodes']

    A = mat_data['A']
    C = mat_data['C']
    d_t = np.float32(1e-5)
    s_orbit = orbit_nodes 
    n_array = mat_data['n_array_cpu']
    t_array = mat_data['t_array_cpu']    
    u = mat_data['u']  
    R = mat_data['R']
    I_u = mat_data['I_u']
    I_p_x = mat_data['I_p_x']
    I_p_y = mat_data['I_p_y']    
    I_p_z = mat_data['I_p_z']
    orbit_ind = mat_data['orbit_ind']
    orbit_tetra_ind = mat_data['orbit_tetra_ind']
    a_o = mat_data['a_o']
    M = mat_data['M_cpu']
    ast_ind = mat_data['ast_ind']
    j_ind_ast = mat_data['j_ind_ast']


    r_1 = mat_data['r_1_cpu']
    r_2 = mat_data['r_2_cpu']
    r_3 = mat_data['r_3_cpu']
    r_u = mat_data['r_u_cpu']

    # test data
    r_aux = mat_data_test['test_aux_vec']
    r_mat = mat_data_test['test_mat_vec']
    r_aux_1 = mat_data_test['test_aux_vec_1']
    r_f = mat_data_test['f_aux']
    r_aux_init = mat_data_test['test_aux_init']
    l_1 = mat_data_test['l_1']
    l_2 = mat_data_test['l_2']
    l_3 = mat_data_test['l_3']
    r_dp_1_dt_vec = mat_data_test['dp_1_dt_vec_cpu']
    r_du_dt_vec = mat_data_test['du_dt_vec_cpu']


    r_1_data = mat_data_full['p_1_data_single']
    r_u_data = mat_data_full['u_data_mat']
    r_f_data = mat_data_full['f_data_mat']
 

    du_dt_vec = np.zeros((orbit_ind.shape[0], 3))
    t_vec = np.zeros(orbit_ind.shape[0])
    data_param = 4
    k = 1
    gpu_extended_memory = 3
    d_t_pml_val = np.float32(7)
    I_u = I_u - 1
    I_p_x = I_p_x - 1
    I_p_y = I_p_y - 1
    I_p_z = I_p_z - 1
    orbit_ind = orbit_ind - 1
    orbit_tetra_ind = orbit_tetra_ind - 1
    ast_ind = ast_ind - 1
    j_ind_ast = j_ind_ast - 1
    

    t_data = np.zeros( math.ceil(len(t_vec) / data_param))
    u_data = np.zeros((orbit_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    du_dt_data = np.zeros((orbit_ind.shape[0], math.ceil(len(t_vec) / data_param)))
    p_1_data = np.zeros((orbit_tetra_ind.shape[0], math.ceil(len(t_vec) / data_param)), dtype=np.float32)
    p_2_data = np.zeros((orbit_tetra_ind.shape[0], math.ceil(len(t_vec) / data_param)), dtype=np.float32)
    p_3_data = np.zeros((orbit_tetra_ind.shape[0], math.ceil(len(t_vec) / data_param)), dtype=np.float32)
    u_data_mat = np.zeros((j_ind_ast.shape[0], math.ceil(len(t_vec) / data_param)))
    f_data_mat = np.zeros((j_ind_ast.shape[0], math.ceil(len(t_vec) / data_param)))
    

    (boundary_vec_1, boundary_vec_2) = boundary_point_source(source_points, orbit_triangles, orbit_nodes)

    data_ind = -1
    time_series_ind = 0
    t_0 = d_t



    if time_series_ind == 0:
        div_vec = np.zeros((A.shape[0], 1), dtype=np.float32)
        aux_init = np.zeros(u.shape)

        p_1 = np.zeros((A.shape[0], 3), dtype=np.float32)
        p_2 = np.zeros((A.shape[0], 3), dtype=np.float32)
        p_3 = np.zeros((A.shape[0], 3), dtype=np.float32)




    start_time = time.time()
    # for i in range(1, 6):
    for i in range(1, 6):
    
        p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec = one_loop(d_t, s_orbit, k, boundary_vec_1, boundary_vec_2, div_vec, p_1, p_2, p_3, n_array, t_array, gpu_extended_memory, ast_ind, R, u, I_u, orbit_ind, a_o, C, M, aux_init, d_t_pml_val, I_p_x, I_p_y, I_p_z, orbit_tetra_ind, A, r_f, r_aux, r_u, r_aux_init, t_0, i, r_1, r_2, r_3, r_dp_1_dt_vec, r_du_dt_vec)
        
        
        
        if (i-1) % data_param == 0:
            data_ind += 1
            t_data[data_ind] = t

            u_data[:, data_ind] = u[orbit_ind[:,0],0]*a_o[k,0] + u[orbit_ind[:,0],1]*a_o[k,1] + u[orbit_ind[:,0],2]*a_o[k,2]
            du_dt_data[:, data_ind] = du_dt_vec[:,0]*a_o[k,0] + du_dt_vec[:,1]*a_o[k,1] + du_dt_vec[:,2]*a_o[k,2]
            p_1_data[:, data_ind] = dp_1_dt_vec[:,0]*a_o[k,0] + dp_1_dt_vec[:,1]*a_o[k,1] + dp_1_dt_vec[:,2]*a_o[k,2]
            p_2_data[:, data_ind] = dp_2_dt_vec[:,0]*a_o[k,0] + dp_2_dt_vec[:,1]*a_o[k,1] + dp_2_dt_vec[:,2]*a_o[k,2]
            p_3_data[:, data_ind] = dp_3_dt_vec[:,0]*a_o[k,0] + dp_3_dt_vec[:,1]*a_o[k,1] + dp_3_dt_vec[:,2]*a_o[k,2]
            u_data_mat[:, data_ind] = u[j_ind_ast[:,0],0]*a_o[k,0] + u[j_ind_ast[:,0],1]*a_o[k,1] + u[j_ind_ast[:,0],2]*a_o[k,2]

            div_vec[ast_ind[:,0], 0] = p_1[ast_ind[:,0],0]+p_2[ast_ind[:,0],1]+p_3[ast_ind[:,0],2]

            test_mat_vec = R * u

            aux_vec = B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory) + test_mat_vec
            aux_vec = pcg_iteration_gpu(C,aux_vec,1e-5,1000,M,np.zeros((aux_vec.shape[0],3)),gpu_extended_memory)
            du_dt_vec_ast = aux_vec[j_ind_ast[:,0],:]
            f_data_mat[:,data_ind] = du_dt_vec_ast[:,0]*a_o[k,0] + du_dt_vec_ast[:,1]*a_o[k,1] + du_dt_vec_ast[:,2]*a_o[k,2]
            


    

        
        

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    
    r_1_data_norm = np.linalg.norm(r_1_data)
    p_1_data_norm = np.linalg.norm(p_1_data - r_1_data)
    p_1_data_only_norm = np.linalg.norm(p_1_data)

    print(f"r_1_data_norm: {r_1_data_norm:.40f}")
    print(f"p_1_data_norm: {p_1_data_norm:.40f}")
    print(f"p_1_data_only_norm: {p_1_data_only_norm:.40f}")
    print(p_1_data_norm / r_1_data_norm)
    print("\n")

    f_data_mat_norm = np.linalg.norm(f_data_mat - r_f_data)
    f_data_mat_only_norm = np.linalg.norm(f_data_mat)
    r_f_data_norm = np.linalg.norm(r_f_data)

    print(f"f_data_mat_norm: {f_data_mat_norm:.40f}")
    print(f"f_data_mat_only_norm: {f_data_mat_only_norm:.40f}")
    print(f"r_f_data_norm: {r_f_data_norm:.40f}")
    print(f_data_mat_norm / r_f_data_norm)
    print("\n")

    







