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



def ml_ode_batch(t, y, iex_fns, params):
    """
    ODE ML vectorisée — numpy pur, pas de boucle Python.
    """
    N   = len(iex_fns)
    Vms = y[0::2]   # indices pairs   → Vm de chaque neurone
    ns  = y[1::2]   # indices impairs → n  de chaque neurone

    # Courants Iex pour tous les neurones en une fois
    Iex = np.array([f(t) for f in iex_fns])

    # Calcul vectorisé des composantes ML
    I_Ca = params['Gna'] * mss(Vms, params['V1'], params['V2']) \
           * (Vms - params['ECa'])
    I_K  = params['Gk']  * ns * (Vms - params['EK'])
    I_L  = params['GL']  * (Vms - params['EL'])

    dVms = (Iex - I_Ca - I_K - I_L) / params['Cm']
    dns  = lambda_n(Vms, params['V3'], params['V4'],
                    params['lambda0']) \
           * (nss(Vms, params['V3'], params['V4']) - ns)

    # Interleave dVms et dns
    dydt       = np.zeros(2 * N)
    dydt[0::2] = dVms
    dydt[1::2] = dns

    return dydt


def simulate_batch(iex_fns, params, T_sim=0.5, dt=1e-4):
    """
    Simule un batch de neurones ML en parallèle.

    Parameters
    ----------
    iex_fns : list — liste de N fonctions Iex_j(t)
    params  : dict — paramètres ML
    T_sim   : float — durée simulation (s)
    dt      : float — pas de temps (s)

    Returns
    -------
    t   : np.ndarray — vecteur temps (s)
    Vms : np.ndarray — potentiels membranaires (N, T)
    ns  : np.ndarray — variables potassium (N, T)
    """
    N = len(iex_fns)

    # Conditions initiales pour tous les neurones
    Vm0 = params['EK']
    n0  = nss(Vm0, params['V3'], params['V4'])
    y0  = np.array([Vm0, n0] * N)

    t_span = (0, T_sim)
    t_eval = np.arange(0, T_sim, dt)

    sol = solve_ivp(
        fun=lambda t, y: ml_ode_batch(t, y, iex_fns, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        max_step=dt * 10
    )

    # Extraire Vm et n pour chaque neurone
    Vms = np.array([sol.y[2*j]     for j in range(N)])
    ns  = np.array([sol.y[2*j + 1] for j in range(N)])

    return sol.t, Vms, ns


def time_to_first_spike_batch(Vms, t, threshold=0.0):
    """
    Détecte le premier spike pour chaque neurone du batch.

    Parameters
    ----------
    Vms       : np.ndarray — potentiels (N, T)
    t         : np.ndarray — vecteur temps (s)
    threshold : float      — seuil de détection (V)

    Returns
    -------
    spike_times : list — latences (s), None si pas de spike
    """
    spike_times = []
    for Vm in Vms:
        St = time_to_first_spike(Vm, t, threshold)
        spike_times.append(St)
    return spike_times


from joblib import Parallel, delayed

def simulate_single_neuron(iex_fn, params, t_arr):
    """
    Simule un seul neurone ML par Euler.
    Conçu pour être appelé en parallèle.
    """
    n_steps = len(t_arr)
    dt      = t_arr[1] - t_arr[0]
    Vm0     = params['EK']
    n0      = nss(Vm0, params['V3'], params['V4'])

    Vm = np.full(n_steps, Vm0)
    n  = np.full(n_steps, n0)

    for k in range(n_steps - 1):
        t     = t_arr[k]
        Iex_k = iex_fn(t)

        I_Ca = params['Gna'] \
               * mss(Vm[k], params['V1'], params['V2']) \
               * (Vm[k] - params['ECa'])
        I_K  = params['Gk'] * n[k] * (Vm[k] - params['EK'])
        I_L  = params['GL'] * (Vm[k] - params['EL'])

        dVm  = (Iex_k - I_Ca - I_K - I_L) / params['Cm']
        dn   = lambda_n(Vm[k], params['V3'],
                        params['V4'], params['lambda0']) \
               * (nss(Vm[k], params['V3'], params['V4']) - n[k])

        Vm[k+1] = Vm[k] + dt * dVm
        n[k+1]  = n[k]  + dt * dn

    return Vm, n


def simulate_batch_parallel(iex_fns, params,
                             T_sim=0.5, dt=1e-4,
                             n_jobs=-1):
    t_arr = np.arange(0, T_sim, dt)

    # Seuil : paralléliser seulement si assez de neurones
    if len(iex_fns) < 20:
        # Sequential pour les petits batches
        results = [simulate_single_neuron(f, params, t_arr)
                   for f in iex_fns]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_single_neuron)(
                iex_fn, params, t_arr)
            for iex_fn in iex_fns)

    Vms = np.array([r[0] for r in results])
    ns  = np.array([r[1] for r in results])
    return t_arr, Vms, ns