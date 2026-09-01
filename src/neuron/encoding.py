"""
TTFS encoding : canal → courant d'excitation → latence spike.
"""

import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d


def build_iex_to_st_table(simulate_fn, params,
                           iex_min=77e-6, iex_max=150e-6,
                           n_points=100):
    """
    Construit la table de correspondance Iex → St.

    Parameters
    ----------
    simulate_fn : callable — fonction simulate()
    params      : dict     — paramètres neurone
    iex_min     : float    — courant minimum (A/cm²)
    iex_max     : float    — courant maximum (A/cm²)
    n_points    : int      — nombre de points

    Returns
    -------
    iex_arr : np.ndarray — courants (A/cm²)
    st_arr  : np.ndarray — latences (s)
    """
    from src.neuron.morris_lecar import time_to_first_spike

    iex_arr = np.linspace(iex_min, iex_max, n_points)
    st_arr  = []

    for Iex in iex_arr:
        t, Vm, n = simulate_fn(Iex=Iex, params=params)
        St = time_to_first_spike(Vm, t)
        st_arr.append(St if St is not None else np.nan)

    st_arr = np.array(st_arr)

    # Garder seulement les points valides
    valid   = ~np.isnan(st_arr)
    return iex_arr[valid], st_arr[valid]


def build_st_to_iex_interpolator(iex_arr, st_arr):
    """
    Construit l'interpolateur inverse St → Iex.
    Permet de cibler une latence précise.

    Returns
    -------
    interpolateur : callable — St → Iex
    """
    # St décroit avec Iex → on inverse
    # st_arr est décroissant, on le retourne pour interp1d
    idx = np.argsort(st_arr)
    return interp1d(
        st_arr[idx], iex_arr[idx],
        kind='linear',
        bounds_error=False,
        fill_value=(iex_arr[idx][0], iex_arr[idx][-1])
    )


def canal_to_iex_uniform_st(x, st_to_iex_fn,
                              st_min, st_max,
                              mu=0.0, sigma=1.0):
    """
    Convertit une valeur canal en Iex
    via mapping uniforme sur St (pas sur Iex).

    x → CDF → St uniforme → Iex correspondant
    """
    # CDF gaussienne → valeur uniforme [0,1]
    p = norm.cdf(x, loc=mu, scale=sigma)

    # Mapper vers [St_min, St_max]
    # Note : St_max → spike tardif → valeur basse
    #        St_min → spike précoce → valeur haute
    # On inverse : x élevé → St petit (spike précoce)
    St_target = st_max - p * (st_max - st_min)

    # Trouver Iex correspondant
    return st_to_iex_fn(St_target), St_target


def canal_to_iex(x, iex_min, iex_max, mu=0.0, sigma=1.0):
    """
    Version simple (Option A) — gardée pour référence.
    """
    p = norm.cdf(x, loc=mu, scale=sigma)
    return iex_min + p * (iex_max - iex_min)


def iex_to_ttfs(Iex, simulate_fn, params):
    """
    Calcule la latence TTFS pour un courant donné.
    """
    from src.neuron.morris_lecar import time_to_first_spike
    t, Vm, n = simulate_fn(Iex=Iex, params=params)
    return time_to_first_spike(Vm, t)


def encode_feature(x, st_to_iex_fn, simulate_fn,
                   params, st_min, st_max):
    """
    Pipeline complet : valeur canal → latence spike.
    """
    Iex, St_target = canal_to_iex_uniform_st(
        x, st_to_iex_fn, st_min, st_max)
    St = iex_to_ttfs(Iex, simulate_fn, params)
    return St, Iex, St_target