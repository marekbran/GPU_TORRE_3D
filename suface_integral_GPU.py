import pyopencl as cl
import numpy as np
import time
import math
import scipy.io
import torch


"""
Reguires large GPU memory, so it is not suitable for small GPUs.
"""


def surface_integral(u_data, du_dt_data, p_1_data, p_2_data, p_3_data, t_data, t_shift, source_points, orbit_nodes, orbit_triangles, device):
    n_nodes = orbit_nodes.shape[0]
    n_source_points = source_points.shape[0]
    n_triangles = orbit_triangles.shape[0]


    orbit_triangles = orbit_triangles.to(torch.long)
    t_c_points = (1/3)*(orbit_nodes[orbit_triangles[:,0], :] + orbit_nodes[orbit_triangles[:,1], :] + orbit_nodes[orbit_triangles[:,2], :])
    s_vec = torch.zeros((n_triangles, n_source_points)).to(device)
    scaling_vec_2 = torch.zeros((n_triangles, n_source_points))

    aux_vec_1 = orbit_nodes[orbit_triangles[:,1], :] - orbit_nodes[orbit_triangles[:,0], :] 
    aux_vec_2 = orbit_nodes[orbit_triangles[:,2], :] - orbit_nodes[orbit_triangles[:,0], :] 
    n_vec_aux = torch.linalg.cross(aux_vec_1, aux_vec_2, dim=-1)
    ala_vec = torch.sqrt(torch.sum(n_vec_aux**2, dim=1)) / 2
    n_vec_aux = n_vec_aux / (2 * ala_vec[:, np.newaxis])
    n_vec_aux = torch.sign(torch.sum(t_c_points * n_vec_aux, dim=1)).unsqueeze(1) * n_vec_aux

    

    for i in range(0, n_source_points): 

        row_indices = (i * torch.ones(n_triangles, dtype=torch.int32, device=source_points.device))
        source_vec  = t_c_points - source_points[row_indices, :]
        
        s_vec[:, i] = torch.sqrt(torch.sum(source_vec**2, dim=1)) 
        source_vec = source_vec / s_vec[:, [i]]
        scaling_vec_2[:,i] = torch.sum(source_vec*n_vec_aux, dim=1)
    
    
    print(t_data.shape, t_data)
    d_t_2 = t_data[1] - t_data[0]
    ret_time = torch.floor((s_vec-t_shift)/d_t_2)


    u_data = (1/3)*(u_data[orbit_triangles[:,0],:] + u_data[orbit_triangles[:,1],:] + u_data[orbit_triangles[:,2],:])
    du_dt_data = (1/3)*(du_dt_data[orbit_triangles[:,0],:]+du_dt_data[orbit_triangles[:,1],:]+du_dt_data[orbit_triangles[:,2],:])
    num_cols = p_1_data.shape[1]

    idx_0 = torch.zeros(num_cols, dtype=torch.int32, device=device)
    idx_1 = torch.ones(num_cols, dtype=torch.int32, device=device)
    idx_2 = 2 * torch.ones(num_cols, dtype=torch.int32, device=device)

    
    du_dn_data = p_1_data * n_vec_aux[:, idx_0] + \
                p_2_data * n_vec_aux[:, idx_1] + \
                p_3_data * n_vec_aux[:, idx_2]

    
    
    integ_vec = torch.zeros((n_source_points, 2 * t_data.shape[1]), device=device)
    ind_aux_1 = torch.arange(0, n_source_points, device=device)
    ind_aux_1 = ind_aux_1.unsqueeze(0).expand(n_triangles, -1)

    for t_ind in range(0,u_data.shape[1]):
        aux_vec_1 =  -ala_vec[:]*u_data[:, t_ind]
        aux_vec_1 = scaling_vec_2 * aux_vec_1.unsqueeze(-1) / (s_vec**2)
        aux_vec_2 =  -ala_vec[:]*du_dt_data[:, t_ind]
        aux_vec_2 = scaling_vec_2*aux_vec_2.unsqueeze(-1)/(s_vec)
        aux_vec_3 =  -ala_vec[:]*du_dn_data[:, t_ind]
        aux_vec_3 = aux_vec_3.unsqueeze(-1)/(s_vec)
        

        row_indices = ind_aux_1.flatten().long() # Casting to long is crucial for indexing
        col_indices = (ret_time.flatten() + t_ind).long()

        values_to_accumulate = (aux_vec_1 + aux_vec_2 + aux_vec_3).flatten()

        # 2. Convert 2D indices to a single, flattened index
        output_shape = integ_vec.shape
        flat_indices = row_indices * output_shape[1] + col_indices

        # 3. Create the output tensor and use scatter_add_
        flat_aux_vec = torch.zeros(output_shape[0] * output_shape[1], dtype=values_to_accumulate.dtype, device=device)
        flat_aux_vec.scatter_add_(0, flat_indices, values_to_accumulate)

        # 4. Reshape the result back to the original 2D shape
        aux_vec = flat_aux_vec.reshape(output_shape)
        #aux_vec = np.bincount([ ind_aux_1[:], ret_time[:]+t_ind], aux_vec_1[:]+aux_vec_2[:]+aux_vec_3[:], integ_vec.shape)

        integ_vec = integ_vec + aux_vec




    integ_vec = integ_vec/(4*math.pi)
    test = integ_vec
    

    return integ_vec[:,0:t_data.shape[0]], test


if __name__ == "__main__":
    mat_data_full = scipy.io.loadmat('GPU_TORRE_3D\surface_integral.mat')
    mat_data = scipy.io.loadmat(r"GPU_TORRE_3D\bounndary_point_source.mat")
    device = torch.device("cuda")
    
    source_points = mat_data['source_points']
    source_points = torch.from_numpy(source_points)
    source_points = source_points.to(device)
    
    orbit_triangles = mat_data['orbit_triangles']
    orbit_triangles = torch.from_numpy(orbit_triangles)
    orbit_triangles = orbit_triangles.to(device)
    
    orbit_nodes = mat_data['orbit_nodes']
    orbit_nodes = torch.from_numpy(orbit_nodes)
    orbit_nodes = orbit_nodes.to(device)

    u_data = mat_data_full['u_data']
    u_data = torch.from_numpy(u_data)
    u_data = u_data.to(device)
    
    du_dt_data = mat_data_full['du_dt_data']
    du_dt_data = torch.from_numpy(du_dt_data)
    du_dt_data = du_dt_data.to(device)
    
    p_1_data = mat_data_full['p_1_data_single']
    p_1_data = torch.from_numpy(p_1_data)
    p_1_data = p_1_data.to(device)
    
    p_2_data = mat_data_full['p_2_data_single']
    p_2_data = torch.from_numpy(p_2_data)
    p_2_data = p_2_data.to(device)
    
    p_3_data = mat_data_full['p_3_data_single']
    p_3_data = torch.from_numpy(p_3_data)
    p_3_data = p_3_data.to(device)
    
    t_data = mat_data_full['t_data']
    t_data = torch.from_numpy(t_data)
    t_data = t_data.to(device)
    
    integ_vec = mat_data_full['integ_vec']
    integ_vec = torch.from_numpy(integ_vec)
    integ_vec = integ_vec.to(device)
    
    r_test = mat_data_full['test']

    t_shift = 0.01
    orbit_triangles = orbit_triangles.to(torch.long)
    orbit_triangles -= 1

    start_time = time.time()
    surface_integral, test = surface_integral(u_data, du_dt_data, p_1_data, p_2_data, p_3_data, t_data, t_shift, source_points, orbit_nodes, orbit_triangles, device)
    end_time = time.time()
    
    test = test.cpu().numpy()

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