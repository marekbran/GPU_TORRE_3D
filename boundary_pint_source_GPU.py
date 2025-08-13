import pyopencl as cl
import numpy as np
import time
import scipy.io
import torch


def boundary_point_source(source_points, orbit_triangles, orbit_nodes):

    n_nodes = orbit_nodes.shape[0]
    n_source_points = source_points.shape[0]

    boundary_vec_1 = torch.zeros((n_nodes,n_source_points))
    boundary_vec_2 = torch.zeros((n_nodes, n_source_points))

    for i in range(n_source_points): 
        d_min = torch.min(torch.sum((source_points[i, :] - orbit_nodes) ** 2, axis=1))
        d_ind = torch.argmin(torch.sum((source_points[i, :] - orbit_nodes) ** 2, axis=1))
        boundary_vec_2[d_ind, i] = 1

    return (boundary_vec_1, boundary_vec_2)



if __name__ == "__main__":
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
    
    start_time = time.time()
    boundary_vec_1, boundary_vec_2 = boundary_point_source(source_points, orbit_triangles, orbit_nodes)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")