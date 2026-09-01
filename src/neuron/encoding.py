"""
TTFS encoding : canal → courant d'excitation → latence spike.
"""

import numpy as np
from scipy.stats import norm


def canal_to_iex(x, iex_min, iex_max, mu=0.0, sigma=1.0):
    """
    Convertit une valeur de canal en courant d'excitation
    via la CDF gaussienne (Option B).

    Parameters
    ----------
    x       : float — valeur de canal (Re ou Im)
    iex_min : float — courant minimum (A/cm²)
    iex_max : float — courant maximum (A/cm²)
    mu      : float — moyenne de la distribution canal
    sigma   : float — écart-type de la distribution canal

    Returns
    -------
    Iex : float — courant d'excitation (A/cm²)
    """
    # CDF gaussienne → valeur uniforme dans [0, 1]
    p = norm.cdf(x, loc=mu, scale=sigma)

    # Mapper vers [iex_min, iex_max]
    return iex_min + p * (iex_max - iex_min)


def iex_to_ttfs(Iex, simulate_fn, params, T_sim=0.5, dt=1e-4):
    """
    Calcule la latence TTFS pour un courant donné.

    Parameters
    ----------
    Iex         : float    — courant d'excitation (A/cm²)
    simulate_fn : callable — fonction simulate() du modèle ML
    params      : dict     — paramètres du neurone
    T_sim       : float    — durée simulation (s)
    dt          : float    — pas de temps (s)

    Returns
    -------
    St : float — latence au premier spike (s)
              — None si pas de spike
    """
    from src.neuron.morris_lecar import time_to_first_spike

    t, Vm, n = simulate_fn(Iex=Iex, params=params,
                           T_sim=T_sim, dt=dt)
    return time_to_first_spike(Vm, t)


def encode_feature(x, simulate_fn, params, encoding_params):
    """
    Pipeline complet : valeur canal → latence spike.

    Parameters
    ----------
    x               : float — valeur de canal
    simulate_fn     : callable
    params          : dict  — paramètres neurone
    encoding_params : dict  — paramètres encodage (depuis yaml)

    Returns
    -------
    St  : float — latence (s), None si pas de spike
    Iex : float — courant utilisé (pour debug)
    """
    iex_min = encoding_params['Iex_min']
    iex_max = encoding_params['Iex_max']

    Iex = canal_to_iex(x, iex_min, iex_max)
    St  = iex_to_ttfs(Iex, simulate_fn, params)

    return St, Iex