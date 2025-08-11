import pyopencl as cl
import numpy as np
import time
import sys
import scipy.io




def B_T_prod(p_1, p_2, p_3, div_vec, n, t, gpu_extended_memory):

    try: 
        div_u = np.zeros((n.shape[0], 3))
        v_ind = np.array([[0, 1, 2, 3], [1, 2, 0, 3], [2, 3, 0, 1], [3, 0, 2, 1]])
        size_n = n.shape[0]


        for i in range(4):
            row_indices = t[:, v_ind[i, 1]] - 1
            aux_vec = n[row_indices, :]
            
            


            row_indices = t[:, v_ind[i, 3]] - 1
            v1 = n[row_indices, :]
            v1 = v1 - aux_vec

            row_indices = t[:,v_ind[i, 2]] - 1
            v2 = n[row_indices, :]
            v2 = v2 - aux_vec
            
            

            for k in range(3):
 
                p_aux = p_1[: , k]
                if k == 0:
                    p_aux = p_aux - div_vec[:, 0]
  

                    
                aux_vec = p_aux * v1[:,1] *v2[:,2]

                
                aux_vec = aux_vec - p_aux * v1[:,2] * v2[:,1]
  



                p_aux = p_2[: , k]
                if k == 1:
                    p_aux = p_aux - div_vec[:, 0]
 
                aux_vec = aux_vec - p_aux * v1[:,0] * v2[:,2]

                aux_vec = aux_vec + p_aux * v1[:,2] * v2[:,0]



                p_aux = p_3[: , k]
                if k == 2:
                    p_aux = p_aux - div_vec[:, 0]

                aux_vec = aux_vec + p_aux * v1[:,0] * v2[:,1]


                aux_vec = aux_vec - p_aux * v1[:,1] * v2[:,0]




                if i == 0:
                    div_u[:, k] = np.bincount(t[:, i] - 1, weights=aux_vec, minlength=size_n)

                    
                

                else:

                    div_u[:, k] = div_u[:, k] + np.bincount(t[:, i] - 1, weights=aux_vec, minlength=size_n)
                



                    

        div_u = div_u / 6
        
        
        
       
        return div_u


    except cl.LogicError as e:
        status_message = f"OpenCL LogicError: {e}. Ensure OpenCL drivers and device are configured."
        error_occurred = True
        print(f"Python Error (OpenCL): {status_message}", file=sys.stderr) # Print to stderr for MATLAB
    except Exception as e:
        status_message = f"An unexpected error occurred: {e}"
        error_occurred = True
        print(f"Python Error: {status_message}", file=sys.stderr) # Print to stderr for MATLAB


if __name__ == "__main__":

    mat_data = scipy.io.loadmat('GPU_TORRE_3D/B_T_prod.mat')

    n = mat_data['n_array_cpu']
    t = mat_data['t_array_cpu']
    p_1 = mat_data['p_1']
    p_2 = mat_data['p_2']
    p_3 = mat_data['p_3']
    div_vec = mat_data['div_vec']




    A = mat_data['A_cpu']

    start_time = time.time()
    p = B_T_prod(p_1, p_2, p_3, div_vec, n, t, 3)

    end_time = time.time()


    


    print(f"Execution time: {end_time - start_time} seconds")
