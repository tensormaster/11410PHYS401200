# gnerated by ChatGPT
import numpy as np
from scipy.integrate import quad

def kappa_from_K(K):
    """
    Compute kappa = [2 sinh(2K)] / [cosh(2K)^2]
    """
    return 2.0 * np.sinh(2.0 * K) / (np.cosh(2.0 * K)**2)


def _integrand(theta, kappa):
    """
    Integrand:
        ln( 0.5 * (1 + sqrt(1 - kappa^2 * sin^2(theta))) )
    for a fixed kappa.
    """
    s = np.sin(theta)
    inside_sqrt = 1.0 - (kappa**2) * (s**2)

    # Numerical safety: clamp tiny negative values to 0
    inside_sqrt = np.clip(inside_sqrt, 0.0, None)

    return np.log(0.5 * (1.0 + np.sqrt(inside_sqrt)))


def free_energy_per_site(T, J=1.0, k_B=1.0, **quad_kwargs):
    """
    Onsager free energy per site f(T) for the 2D Ising model
    on a square lattice, nearest-neighbour coupling J, zero field.

    Scalar version: T is a float.

    Parameters
    ----------
    T : float
        Temperature
    J : float, optional
        Coupling constant (default 1.0)
    k_B : float, optional
        Boltzmann constant (default 1.0)
    quad_kwargs : dict, optional
        Extra keyword arguments passed to scipy.integrate.quad
        (e.g., epsabs=1e-10, epsrel=1e-10).

    Returns
    -------
    f : float
        Free energy per spin (per site).
    """
    beta = 1.0 / (k_B * T)
    K = beta * J
    kap = kappa_from_K(K)

    # Integral over theta in [0, pi]
    I, err = quad(_integrand, 0.0, np.pi, args=(kap,), **quad_kwargs)

    # Dimensionless free energy: -beta f
    minus_beta_f = np.log(2.0 * np.cosh(2.0 * K)) + (1.0 / (2.0 * np.pi)) * I

    # Return f
    f = -minus_beta_f / beta
    return f


if __name__ == "__main__":
    # Example usage: J = 1, k_B = 1
    J = 1.0
    k_B = 1.0

    # Critical temperature for 2D square Ising:
    # sinh(2K_c) = 1 -> K_c = 0.5 * asinh(1)
    Kc = 0.5 * np.arcsinh(1.0)
    Tc = J / (k_B * Kc)
    print("Estimated critical temperature Tc ≈", Tc)

    for T in [0.5 * Tc, Tc, 2.0 * Tc]:
        fT = free_energy_per_site(T, J=J, k_B=k_B, epsabs=1e-10, epsrel=1e-10)
        print(f"T = {T:.6f},  f(T) = {fT:.10f}")
