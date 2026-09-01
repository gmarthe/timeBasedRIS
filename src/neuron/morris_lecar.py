"""
Morris-Lecar neuron model (Sourikopoulos et al. 2017).
Equations 1-5 from the paper.
"""

import numpy as np
from scipy.integrate import solve_ivp


def mss(Vm, V1, V2):
    """Steady-state sodium activation (Eq. 3)."""
    return 0.5 * (1 + np.tanh((Vm - V1) / V2))


def nss(Vm, V3, V4):
    """Steady-state potassium activation (Eq. 4)."""
    return 0.5 * (1 + np.tanh((Vm - V3) / V4))


def lambda_n(Vm, V3, V4, lambda0):
    """Potassium rate function (Eq. 5)."""
    return lambda0 * np.cosh((Vm - V3) / (2 * V4))


def ml_ode(t, y, Iex, params):
    """
    Morris-Lecar ODE system (Eq. 1-2).

    y = [Vm, n]
    """
    Vm, n = y

    Cm      = params['Cm']
    GCa     = params['Gna']    # sodium ≈ calcium dans ML
    GK      = params['Gk']
    GL      = params['GL']
    ECa     = params['ECa']
    EK      = params['EK']
    EL      = params['EL']
    V1      = params['V1']
    V2      = params['V2']
    V3      = params['V3']
    V4      = params['V4']
    lam0 = float(params['lambda0'])

    # Eq. 1 : dVm/dt
    I_Ca = GCa * mss(Vm, V1, V2) * (Vm - ECa)
    I_K  = GK  * n               * (Vm - EK)
    I_L  = GL                    * (Vm - EL)
    dVm  = (Iex - I_Ca - I_K - I_L) / Cm

    # Eq. 2 : dn/dt
    dn = lambda_n(Vm, V3, V4, lam0) * (nss(Vm, V3, V4) - n)

    return [dVm, dn]


def simulate(Iex, params, T_sim=0.5, dt=1e-4):
    """
    Iex : float OU callable(t) — courant d'excitation
    """
    Vm0 = params['EK']
    n0  = nss(Vm0, params['V3'], params['V4'])
    y0  = [Vm0, n0]

    # Si Iex est une constante, on en fait une fonction
    if callable(Iex):
        iex_fn = Iex
    else:
        iex_fn = lambda t: Iex

    sol = solve_ivp(
        fun=lambda t, y: ml_ode(t, y, iex_fn(t), params),
        t_span=(0, T_sim),
        y0=y0,
        t_eval=np.arange(0, T_sim, dt),
        method='RK45',
        max_step=dt * 10
    )

    return sol.t, sol.y[0], sol.y[1]



def time_to_first_spike(Vm, t, threshold=0.0):
    """
    Détecte le premier spike dans Vm(t).

    Parameters
    ----------
    Vm        : np.ndarray — potentiel membranaire (V)
    t         : np.ndarray — vecteur temps (s)
    threshold : float      — seuil de détection (V)

    Returns
    -------
    St : float — latence au premier spike (s)
              — None si pas de spike détecté
    """
    # Détection des passages ascendants du seuil
    crossings = np.where(
        (Vm[:-1] < threshold) & (Vm[1:] >= threshold)
    )[0]

    if len(crossings) == 0:
        return None

    return t[crossings[0]]