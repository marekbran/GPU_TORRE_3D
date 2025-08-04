import pyopencl as cl
import numpy as np


def pulse_window_function(time_vector, pulse_length, carrier_cycles_per_pulse_cycle, carrier_mode):

    if carrier_mode == 'Hann':

        normalized_time = (time_vector + 0.5 * pulse_length) / pulse_length 
        window_time_normalized = (time_vector / (0.5 * pulse_length)) 
        
       
        pulse_window = np.zeros_like(time_vector, dtype=np.float64)
        

        mask = (time_vector >= -0.5 * pulse_length) & (time_vector <= 0.5 * pulse_length)
        
        pulse_window[mask] = 0.5 * (1 + np.cos(2 * np.pi * time_vector[mask] / pulse_length))



        d_pulse_window = np.zeros_like(time_vector, dtype=np.float64)
        d_pulse_window[mask] = -np.pi / pulse_length * np.sin(2 * np.pi * time_vector[mask] / pulse_length)
        


    elif carrier_mode == 'Gaussian':

        sigma = pulse_length / 4  
        pulse_window = np.exp(-0.5 * (time_vector / sigma)**2)

        d_pulse_window = pulse_window * (-time_vector / sigma**2)

    else:
        raise ValueError(f"Unsupported carrier_mode: {carrier_mode}")

    return pulse_window, d_pulse_window