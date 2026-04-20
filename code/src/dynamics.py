import numpy as np
from numba import njit, prange

# =====================================================================
# STANDARD PYTHON FUNCTIONS
# =====================================================================

def generate_orbit(z0, func, max_iter=100):
    """
    Generates the forward orbit of a point z0 under a rational function.

    In complex dynamics, the forward orbit is the sequence {z_0, R(z_0), 
    R(R(z_0)), ...}. This function computes the orbit up to a maximum 
    number of iterations.

    Args:
        z0 (complex): The initial seed point in the complex plane.
        func (callable): A Python function representing the rational function R(z).
        max_iter (int): The maximum number of iterations to perform.

    Returns:
        np.ndarray: A 1D complex array of length max_iter + 1 containing the orbit.
    """
    orbit = np.zeros(max_iter + 1, dtype=np.complex128)
    orbit[0] = z0
    
    for i in range(max_iter):
        try:
            orbit[i + 1] = func(orbit[i])
        except (OverflowError, ZeroDivisionError):
            # Handle mapping to the point at infinity on the Riemann sphere
            orbit[i + 1] = complex(np.inf, np.inf)
            # Once at infinity, we stop recording meaningful dynamics for basic plots
            break
            
    return orbit


def generate_escape_time_fractal(x_min, x_max, y_min, y_max, width, height, func, max_iter=100, escape_radius=2.0):
    """
    Generates a matrix representing the escape times for a region in the complex plane.
    
    This is typically used to visualize the filled Julia set of a polynomial or 
    rational function. Points that do not escape the radius within max_iter 
    are considered bounded.

    Args:
        x_min (float): The minimum real coordinate.
        x_max (float): The maximum real coordinate.
        y_min (float): The minimum imaginary coordinate.
        y_max (float): The maximum imaginary coordinate.
        width (int): The number of pixels/grid points in the real direction.
        height (int): The number of pixels/grid points in the imaginary direction.
        func (callable): A Python function representing the rational function R(z).
        max_iter (int): The maximum number of iterations.
        escape_radius (float): The radius threshold for escaping to infinity.

    Returns:
        np.ndarray: A 2D integer array of shape (height, width) containing escape times.
    """
    escape_times = np.zeros((height, width), dtype=np.int64)
    x_coords = np.linspace(x_min, x_max, width)
    y_coords = np.linspace(y_min, y_max, height)

    for i in range(height):
        for j in range(width):
            z = complex(x_coords[j], y_coords[i])
            iters = 0
            while abs(z) <= escape_radius and iters < max_iter:
                try:
                    z = func(z)
                    iters += 1
                except (OverflowError, ZeroDivisionError):
                    iters = max_iter
                    break
            escape_times[i, j] = iters

    return escape_times


def generate_fatou_basins(x_min, x_max, y_min, y_max, width, height, func, attractors, max_iter=100, tolerance=1e-6):
    """
    Computes the Fatou basins of attraction for a given rational function.
    
    This function maps each point in the grid to the index of the attracting 
    fixed point or periodic cycle it converges to.

    Args:
        x_min (float): The minimum real coordinate.
        x_max (float): The maximum real coordinate.
        y_min (float): The minimum imaginary coordinate.
        y_max (float): The maximum imaginary coordinate.
        width (int): The number of pixels/grid points in the real direction.
        height (int): The number of pixels/grid points in the imaginary direction.
        func (callable): A Python function representing the rational function R(z).
        attractors (list or np.ndarray): Complex numbers representing attracting points or cycles.
        max_iter (int): The maximum number of iterations.
        tolerance (float): The distance threshold to consider a point converged to an attractor.

    Returns:
        np.ndarray: A 2D integer array of shape (height, width) where each pixel value 
                    corresponds to the index of the converged attractor (or -1 if no convergence).
    """
    basins = np.full((height, width), -1, dtype=np.int64)
    x_coords = np.linspace(x_min, x_max, width)
    y_coords = np.linspace(y_min, y_max, height)
    attractors_array = np.array(attractors, dtype=np.complex128)

    for i in range(height):
        for j in range(width):
            z = complex(x_coords[j], y_coords[i])
            for _ in range(max_iter):
                try:
                    z = func(z)
                except (OverflowError, ZeroDivisionError):
                    break
            
            # Check distance to known attractors
            for k, attractor in enumerate(attractors_array):
                if abs(z - attractor) < tolerance:
                    basins[i, j] = k
                    break

    return basins


# =====================================================================
# NUMBA OPTIMIZED FACTORY FUNCTIONS
# =====================================================================

def make_generate_orbit_jitted(jitted_func):
    """
    Creates a Numba-optimized function to compute the forward orbit.

    Args:
        jitted_func (callable): A Numba `@njit` compiled function for R(z).

    Returns:
        callable: An optimized function `generate_orbit_jitted(z0, max_iter)`
                  that returns a 1D complex numpy array.
    """
    @njit
    def generate_orbit_jitted(z0, max_iter=100):
        orbit = np.zeros(max_iter + 1, dtype=np.complex128)
        orbit[0] = z0
        for i in range(max_iter):
            # Note: Numba handles infs cleanly for complex math if compiled correctly
            orbit[i + 1] = jitted_func(orbit[i])
        return orbit

    return generate_orbit_jitted


def make_generate_escape_time_fractal_jitted(jitted_func):
    """
    Creates a highly optimized, parallelized function for escape time fractals (Julia sets).

    Args:
        jitted_func (callable): A Numba `@njit` compiled function for R(z).

    Returns:
        callable: An optimized function `generate_escape_time_fractal_jitted(
                  x_min, x_max, y_min, y_max, width, height, max_iter, escape_radius)`
                  that returns a 2D integer array.
    """
    @njit(parallel=True)
    def generate_escape_time_fractal_jitted(x_min, x_max, y_min, y_max, width, height, max_iter=100, escape_radius=2.0):
        escape_times = np.zeros((height, width), dtype=np.int64)
        x_coords = np.linspace(x_min, x_max, width)
        y_coords = np.linspace(y_min, y_max, height)

        for i in prange(height):
            for j in range(width):
                z = complex(x_coords[j], y_coords[i])
                iters = 0
                while abs(z) <= escape_radius and iters < max_iter:
                    z = jitted_func(z)
                    iters += 1
                escape_times[i, j] = iters

        return escape_times

    return generate_escape_time_fractal_jitted


def make_generate_fatou_basins_jitted(jitted_func):
    """
    Creates a highly optimized, parallelized function to compute Fatou basins.

    Args:
        jitted_func (callable): A Numba `@njit` compiled function for R(z).

    Returns:
        callable: An optimized function `generate_fatou_basins_jitted(
                  x_min, x_max, y_min, y_max, width, height, attractors, max_iter, tolerance)`
                  that returns a 2D integer array. Note that `attractors` MUST be passed
                  as a 1D numpy array of complex numbers.
    """
    @njit(parallel=True)
    def generate_fatou_basins_jitted(x_min, x_max, y_min, y_max, width, height, attractors, max_iter=100, tolerance=1e-6):
        basins = np.full((height, width), -1, dtype=np.int64)
        x_coords = np.linspace(x_min, x_max, width)
        y_coords = np.linspace(y_min, y_max, height)

        for i in prange(height):
            for j in range(width):
                z = complex(x_coords[j], y_coords[i])
                for _ in range(max_iter):
                    z = jitted_func(z)
                
                for k in range(len(attractors)):
                    if abs(z - attractors[k]) < tolerance:
                        basins[i, j] = k
                        break

        return basins

    return generate_fatou_basins_jitted