import torch

def pulse_window_function_torch(time_vector, pulse_length, carrier_cycles_per_pulse_cycle, carrier_mode, device):
    """
    Calculates pulse window functions using PyTorch tensors on a specified device.
    
    Args:
        time_vector (torch.Tensor): A tensor of time values.
        pulse_length (float): The length of the pulse.
        carrier_cycles_per_pulse_cycle (float): Not used in this function, but kept for signature.
        carrier_mode (str): 'Hann' or 'Gaussian'.
        device (torch.device): The device to perform computations on (e.g., 'cuda' or 'cpu').
        
    Returns:
        tuple: A tuple containing the pulse window tensor and its derivative tensor.
    """
    
    time_vector = time_vector.to(device)

    if carrier_mode == 'Hann':
        
        pulse_window = torch.zeros_like(time_vector, dtype=torch.float64, device=device)
        d_pulse_window = torch.zeros_like(time_vector, dtype=torch.float64, device=device)
        
        mask = (time_vector >= -0.5 * pulse_length) & (time_vector <= 0.5 * pulse_length)
        
        masked_time = time_vector[mask]
        
        # Calculate the Hann window function
        hann_val = 0.5 * (1 + torch.cos(2 * torch.pi * masked_time / pulse_length))
        pulse_window[mask] = hann_val
        
        # Calculate the derivative of the Hann window function
        d_hann_val = -torch.pi / pulse_length * torch.sin(2 * torch.pi * masked_time / pulse_length)
        d_pulse_window[mask] = d_hann_val

    elif carrier_mode == 'Gaussian':
        
        sigma = pulse_length / 4.0
        
        # Calculate the Gaussian pulse window
        pulse_window = torch.exp(-0.5 * (time_vector / sigma)**2)

        # Calculate the derivative of the Gaussian pulse window
        d_pulse_window = pulse_window * (-time_vector / sigma**2)

    else:
        raise ValueError(f"Unsupported carrier_mode: {carrier_mode}")

    return pulse_window, d_pulse_window