# GPU TORRE 3D in Python

This repository contains a simplified version of a [GPU-TORRE-3D](https://github.com/sampsapursiainen/GPU-Torre-3D/tree/main) MATLAB package written in Python. Included is a simplified version of compute_data_gpu, which handles the main logic, and 7 helper functions.



## Overview

This repository implements a simplified, Python-based version of the GPU-TORRE-3D electromagnetic simulation package. The code is designed to run matrix and vector operations on the GPU (using PyOpenCL) for efficiency and is structured around several main computational routines. The following comments and explanations describe the purpose and main logic of each file and important code blocks.

---

## File: `compute_data_gpu.py`

**Purpose:**  
This is the main script for running the simulation. It loads input data, initializes variables, and loops over time steps to compute electromagnetic field evolution using GPU-accelerated routines.

**Key Sections:**

```python
# Imports necessary libraries and helper modules for computation.
import pyopencl as cl
import numpy as np
import time
...
from GPU_TORRE_3D.one_loop_smaller import one_loop
```

```python
# Loads input data (matrices, arrays, etc.) from .mat files for simulation setup.
mat_data = scipy.io.loadmat('GPU_TORRE_3D\system_data_1.mat')
# Many arrays such as n_array, t_array, u, R, etc. are loaded here.

```

```python
# Initializes variables and arrays for simulation output.
t_data = np.zeros(...)
u_data = np.zeros(...)
...

```

```python
# Main simulation loop: calls the `one_loop` function at each time step,
# updating field vectors and collecting output data.
p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec = one_loop(...)

```

```python
# After simulation, prints execution time and saves or processes results.
print(f"Execution time: {end_time - start_time:.4f} seconds")
```

---

## File: `one_loop_smaller.py`

**Purpose:**  
Implements a single time step of the simulation, updating field variables using the finite element method and GPU routines.

**Key Sections:**

```python
def one_loop(...):
    # Prepares pulse window, updates divergence vector and field arrays.
    pulse_window, d_pulse_window = pulse_window_function(...)
    f_aux = pulse_window * boundary_vec_1[:, k] + d_pulse_window * boundary_vec_2[:, k]

    # Updates divergence vector and computes auxiliary vectors for the finite element system.
    div_vec[ast_ind, 0] = p_1[ast_ind,0]+p_2[ast_ind,1]+p_3[ast_ind,2]
    aux_vec = -B_T_prod(p_1, p_2, p_3, div_vec, n_array, t_array, gpu_extended_memory)
    ...
    # Solves linear system with preconditioned conjugate gradient (PCG) method.
    aux_vec_init = pcg_iteration_gpu(C,aux_vec,1e-5,1000,M,aux_vec_init,gpu_extended_memory)
    ...
    # Updates fields u, p_1, p_2, p_3 using new values and time stepping.
    u += d_t * aux_vec_init
    p_1 = p_1 + d_t * aux_vec
    ...
    # Returns updated variables for the next time step.
    return p_1, p_2, p_3, u, t, dp_1_dt_vec, dp_2_dt_vec, dp_3_dt_vec
```

---

## File: `B_prod.py`

**Purpose:**  
Performs matrix-vector products related to the finite element system, offloading computation to the GPU when possible.

**Key Sections:**

```python
def B_prod(u, entry_ind, n, t, gpu_extended_memory):
    # Computes the B matrix product for different components (x, y, z)
    # Handles GPU memory management and error handling for OpenCL.
    try:
        ...
        # Assembles result into output vector p.
        return p
    except cl.LogicError as e:
        # Handles OpenCL errors and prints informative messages.
        print(f"Python Error (OpenCL): {status_message}", file=sys.stderr)
```

---

## File: `pcg_iteration_gpu.py`

**Purpose:**  
Implements the Preconditioned Conjugate Gradient (PCG) method to solve sparse linear systems, optionally using the GPU.

**Key Sections:**

```python
def pcg_iteration_gpu(A, b, tol_val, max_it, M, x, gpu_extended_memory):
    # Iterative PCG solver:
    r = b - A @ x
    z = M * r
    p = z
    ...
    while (conv_val > tol_val) & (j < max_it):
        # Updates search direction and solution until convergence.
        ...
    return x
```

---

## File: `boundary_point_source.py`

**Purpose:**  
Handles the setup and computation of boundary source vectors required for the simulation.

**Key Sections:**

```python
def boundary_point_source(source_points, orbit_triangles, orbit_nodes):
    # Computes boundary vectors given the geometry of the source and mesh.
    ...
if __name__ == "__main__":
    # Loads test data and runs the boundary vector computation as a script.
    mat_data = scipy.io.loadmat('GPU_TORRE_3D\bounndary_point_source.mat')
    ...
```

---

## General Workflow

1. **Initialization:** Input data and parameters are loaded from disk.
2. **Boundary Setup:** Boundary source vectors are calculated.
3. **Time Stepping:** For each time step:
    - The main loop (`one_loop`) is called.
    - Field vectors are updated using the finite element method.
    - Matrix-vector products and linear solvers run on the GPU where possible.
    - Output data is collected.
4. **Output:** Results are saved or printed, and execution time reported.

---

## Additional Notes

- **GPU Acceleration:** The core vector/matrix operations are designed to take advantage of GPU computing using PyOpenCL for speed.
- **Error Handling:** Functions include error catching for OpenCL/GPU failures and print relevant messages.
- **.mat File Usage:** The simulation relies on MATLAB-generated `.mat` files for mesh, geometry, and system parameters.

---

For further clarity, each function and script can be annotated with inline comments as shown above. If you want detailed, line-by-line comments added to each specific file, please specify the file(s), and I can generate those for you!