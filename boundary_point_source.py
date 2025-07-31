import pyopencl as cl
import numpy as np
import time
import sys
import scipy.io


def boundary_point_source(source_points, orbit_triangles, orbit_nodes):

    n_nodes = orbit_nodes.shape[0]
    n_source_points = source_points.shape[0]

    boundary_vec_1 = np.zeros((n_nodes,n_source_points))
    boundary_vec_2 = np.zeros((n_nodes, n_source_points))

    for i in range(n_source_points): 
        d_min = np.min(np.sum((source_points[i, :] - orbit_nodes) ** 2, axis=1))
        d_ind = np.argmin(np.sum((source_points[i, :] - orbit_nodes) ** 2, axis=1))
        boundary_vec_2[d_ind, i] = 1

    return (boundary_vec_1, boundary_vec_2)



if __name__ == "__main__":
    mat_data = scipy.io.loadmat('bounndary_point_source.mat')
    source_points = mat_data['source_points']
    orbit_triangles = mat_data['orbit_triangles']
    orbit_nodes = mat_data['orbit_nodes']

    start_time = time.time()
    boundary_vec_1, boundary_vec_2 = boundary_point_source(source_points, orbit_triangles, orbit_nodes)
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.4f} seconds")