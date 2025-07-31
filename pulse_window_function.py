import pyopencl as cl
import numpy as np


def pulse_window_function(time_vector, pulse_length, carrier_cycles_per_pulse_cycle, carrier_mode):
    """
    Python equivalent of your MATLAB pulse_window_function.

    Args:
        time_vector (np.ndarray): The time points at which to evaluate the window.
                                  This corresponds to `t-s_orbit(:,k)+t_shift`.
        pulse_length (float): The total length/duration of the pulse.
        carrier_cycles_per_pulse_cycle (float): Number of carrier cycles per pulse cycle.
        carrier_mode (str or any): Parameter to determine the type of window or carrier.

    Returns:
        tuple: A tuple containing:
            - pulse_window (np.ndarray): The values of the pulse window at time_vector.
            - d_pulse_window (np.ndarray): The derivative of the pulse window.
    """
    # --- Implement your window logic here based on your MATLAB function ---

    # Placeholder for a common window type (e.g., Hann window)
    # This is just an example; replace with your actual logic
    if carrier_mode == 'Hann':
        # Create a Hann window of appropriate length for the pulse
        # The 'time_vector' needs to be mapped to the [0, 1] range of the window
        # or the window needs to be scaled and shifted to match the pulse_length.

        # A common way to generate a window for a given duration:
        # Assuming time_vector spans the duration of the pulse
        # You might need to normalize time_vector to [0, 1] or similar
        normalized_time = (time_vector + 0.5 * pulse_length) / pulse_length # Adjust based on how your t-s_orbit is centered
        # Ensure normalized_time is within [0, 1] for window functions
        # For values outside this, the window might be zero.
        # This part requires careful translation of your specific window definition.

        # Example: Simple Hann window centered at 0, spanning +/- pulse_length/2
        # (Assuming your time_vector is centered around the pulse center)
        # Create a time base for the window itself
        window_time_normalized = (time_vector / (0.5 * pulse_length)) # normalize to [-1, 1]
        
        # Ensure values outside the pulse are zero
        pulse_window = np.zeros_like(time_vector, dtype=np.float64)
        
        # Calculate the window for the active region
        # This part needs to precisely match your MATLAB implementation
        # A common form for a Hann window over [-L/2, L/2] is 0.5 * (1 + cos(2*pi*t/L))
        mask = (time_vector >= -0.5 * pulse_length) & (time_vector <= 0.5 * pulse_length)
        
        # This is a generic way. Your actual formula might differ.
        pulse_window[mask] = 0.5 * (1 + np.cos(2 * np.pi * time_vector[mask] / pulse_length))

        # --- Calculate the derivative (d_pulse_window) ---
        # This is the trickiest part, as MATLAB's window functions often return derivatives directly.
        # In Python, you might need to:
        # 1. Use symbolic differentiation (SymPy) if your window formula is simple.
        # 2. Use numerical differentiation (np.gradient or finite differences).
        # 3. Implement the analytical derivative yourself if you know the formula.

        # Analytical derivative of a Hann window 0.5 * (1 + cos(2*pi*t/L))
        # d/dt [0.5 * (1 + cos(2*pi*t/L))] = 0.5 * (-sin(2*pi*t/L)) * (2*pi/L)
        # = -pi/L * sin(2*pi*t/L)
        d_pulse_window = np.zeros_like(time_vector, dtype=np.float64)
        d_pulse_window[mask] = -np.pi / pulse_length * np.sin(2 * np.pi * time_vector[mask] / pulse_length)
        
        # For edges, the derivative might be zero if the window goes smoothly to zero.
        # Or it might be non-zero if it's a sharp cutoff. This depends on your exact definition.

    elif carrier_mode == 'Gaussian':
        # Example for a Gaussian window
        # Gaussian window: exp(-0.5 * (t / sigma)^2)
        # You'll need to define how sigma relates to pulse_length in your MATLAB code
        # For simplicity, let's assume a fixed sigma or calculated from pulse_length
        sigma = pulse_length / 4  # Example
        pulse_window = np.exp(-0.5 * (time_vector / sigma)**2)

        # Analytical derivative of Gaussian window
        d_pulse_window = pulse_window * (-time_vector / sigma**2)

    # Add more `elif` blocks for other `carrier_mode` values
    else:
        # Handle unknown carrier_mode or provide a default window
        raise ValueError(f"Unsupported carrier_mode: {carrier_mode}")

    return pulse_window, d_pulse_window