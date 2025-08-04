import pyopencl as cl
import numpy as np
import time
import math
import scipy.io


def surface_integral(u_data, du_dt_data, p_1_data, p_2_data, p_3_data, t_data, t_shift, source_points, orbit_nodes, orbit_triangles):
    n_nodes = orbit_nodes.shape[0]
    n_source_points = source_points.shape[0]
    n_triangles = orbit_triangles.shape[0]



    t_c_points = (1/3)*(orbit_nodes[orbit_triangles[:,0], :] + orbit_nodes[orbit_triangles[:,1], :] + orbit_nodes[orbit_triangles[:,2], :])
    s_vec = np.zeros((n_triangles, n_source_points)) 
    scaling_vec_2 = np.zeros((n_triangles, n_source_points))

    aux_vec_1 = orbit_nodes[orbit_triangles[:,1], :] - orbit_nodes[orbit_triangles[:,0], :] 
    aux_vec_2 = orbit_nodes[orbit_triangles[:,2], :] - orbit_nodes[orbit_triangles[:,0], :] 
    n_vec_aux = np.cross(aux_vec_1, aux_vec_2)
    ala_vec = np.sqrt(np.sum(n_vec_aux**2, axis=1)) / 2
    n_vec_aux = n_vec_aux / (2 * ala_vec[:, np.newaxis])
    n_vec_aux = np.sign(np.sum(t_c_points * n_vec_aux, axis=1))[:, np.newaxis] * n_vec_aux

    

    for i in range(0, n_source_points): 

        source_vec  = t_c_points - source_points[np.int32(i*np.ones(n_triangles)), :]
        s_vec[:, i] = np.sqrt(np.sum(source_vec**2, axis=1)) 
        source_vec = source_vec / s_vec[:, [i]]
        scaling_vec_2[:,i] = np.sum(source_vec*n_vec_aux, axis=1)
    
    
    #d_t_2 = t_data(1) - t_data(0); 
    d_t_2 = 2
    ret_time = np.floor((s_vec-t_shift)/d_t_2)

    u_data = (1/3)*(u_data[orbit_triangles[:,0],:] + u_data[orbit_triangles[:,1],:] + u_data[orbit_triangles[:,2],:])
    du_dt_data = (1/3)*(du_dt_data[orbit_triangles[:,0],:]+du_dt_data[orbit_triangles[:,1],:]+du_dt_data[orbit_triangles[:,2],:])
    du_dn_data = p_1_data*n_vec_aux[:,np.int32(np.zeros(p_1_data.shape[1]))] + p_2_data*n_vec_aux[:,np.int32(np.ones(p_2_data.shape[1]))] + p_3_data*n_vec_aux[:,2*np.int32(np.ones(p_3_data.shape[1]))]

    
    
    integ_vec = np.zeros((n_source_points, 2 * t_data.shape[1]))
    ind_aux_1 = np.arange(0, n_source_points)
    ind_aux_1 = np.tile(ind_aux_1, (n_triangles, 1))

    #for t_ind in range(0,u_data.shape[1]):
    for t_ind in range(0,3):
        aux_vec_1 =  -ala_vec[:]*u_data[:, t_ind]
        aux_vec_1 = scaling_vec_2*aux_vec_1[:][:, np.newaxis]/(s_vec**2)
        aux_vec_2 =  -ala_vec[:]*du_dt_data[:, t_ind]
        aux_vec_2 = scaling_vec_2*aux_vec_2[:][:, np.newaxis]/(s_vec)
        aux_vec_3 =  -ala_vec[:]*du_dn_data[:, t_ind]
        aux_vec_3 = aux_vec_3[:][:, np.newaxis]/(s_vec)
        

        row_indices = (ind_aux_1).flatten()
        row_indices = row_indices.astype(int)

        col_indices = (ret_time.flatten() + t_ind).astype(int)
        col_indices = col_indices.astype(int)
        #col_indices -= 1

        values_to_accumulate = (aux_vec_1 + aux_vec_2 + aux_vec_3).flatten()
        output_shape = integ_vec.shape
        aux_vec = np.zeros(output_shape, dtype=values_to_accumulate.dtype)

        np.add.at(aux_vec, (row_indices, col_indices), values_to_accumulate)
        #aux_vec = np.bincount([ ind_aux_1[:], ret_time[:]+t_ind], aux_vec_1[:]+aux_vec_2[:]+aux_vec_3[:], integ_vec.shape)

        integ_vec = integ_vec + aux_vec




    integ_vec = integ_vec/(4*math.pi)
    test = integ_vec
    print(test.dtype)

    return integ_vec[:,0:t_data.shape[1]], test


if __name__ == "__main__":
    mat_data = scipy.io.loadmat('GPU_TORRE_3D\system_data_1.mat')
    mat_data_full = scipy.io.loadmat('GPU_TORRE_3D\surface_integral.mat')

    source_points = mat_data['source_points']
    orbit_triangles = mat_data['orbit_triangles']
    orbit_nodes = mat_data['orbit_nodes']

    u_data = mat_data_full['u_data']
    du_dt_data = mat_data_full['du_dt_data']
    p_1_data = mat_data_full['p_1_data_single']
    p_2_data = mat_data_full['p_2_data_single']
    p_3_data = mat_data_full['p_3_data_single']
    t_data = mat_data_full['t_data']
    integ_vec = mat_data_full['integ_vec']
    r_test = mat_data_full['test']

    t_shift = 0.01
    orbit_triangles -= 1

    start_time = time.time()
    surface_integral, test = surface_integral(u_data, du_dt_data, p_1_data, p_2_data, p_3_data, t_data, t_shift, source_points, orbit_nodes, orbit_triangles)
    end_time = time.time()

    surfece_integral_norm = np.linalg.norm(surface_integral - integ_vec)
    integ_vec_norm = np.linalg.norm(integ_vec)
    source_integral_only_norm = np.linalg.norm(surface_integral)

    print(f"Surface Integral Norm: {surfece_integral_norm}")
    print(f"Integ Vec Norm: {integ_vec_norm}")
    print(f"Source Integral Only Norm: {source_integral_only_norm}")
    print(surfece_integral_norm / integ_vec_norm)
    print("\n")

    test_norm = np.linalg.norm(test - r_test)
    r_test_norm = np.linalg.norm(r_test)
    test_only_norm = np.linalg.norm(test)

    print(f"Test Norm: {test_norm}")    
    print(f"R Test Norm: {r_test_norm}")
    print(f"Test Only Norm: {test_only_norm}")
    print(test_norm / r_test_norm)
    print("\n")




    print(f"Execution time: {end_time - start_time} seconds")