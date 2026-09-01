"""
SpikeProp : backpropagation pour SNN TTFS.
Bohte et al. 2002, adapté pour neurone Morris-Lecar.
"""

import numpy as np


def spikeprop_loss(t_out, target_idx,
                   t_target=None, t_margin=5e-3):
    """
    Loss SpikeProp.

    L = (t_correct - t_target)²
      + Σ_{k≠correct} max(0, t_margin - (t_k - t_correct))²

    Parameters
    ----------
    t_out      : list  — spike times couche sortie (s)
    target_idx : int   — indice du neurone correct
    t_target   : float — latence cible du bon neurone (s)
                         None → utilise le min des St couche cachée
    t_margin   : float — marge de séparation (s)

    Returns
    -------
    loss  : float
    delta : list — ∂L/∂t_k pour chaque neurone de sortie
    """
    if t_target is None:
        # Cible : spike le plus tôt possible
        valid = [t for t in t_out if t is not None]
        t_target = min(valid) * 0.8 if valid else 10e-3

    loss  = 0.0
    delta = [0.0] * len(t_out)

    for k, t_k in enumerate(t_out):
        if t_k is None:
            continue

        if k == target_idx:
            # Neurone correct → spike tôt
            err       = t_k - t_target
            loss     += err ** 2
            delta[k]  = 2 * err
        else:
            # Neurones incorrects → spike tard
            t_correct = t_out[target_idx]
            if t_correct is None:
                continue
            margin_err = t_margin - (t_k - t_correct)
            if margin_err > 0:
                loss     += margin_err ** 2
                delta[k]  = -2 * margin_err

    return loss, delta


def compute_gradient_weights(delta_out, t_hid, t_out,
                              W_hid_out, D_hid_out,
                              Vm_dot_out, tau_syn=50e-3):
    """
    Calcule ∂L/∂W_hid_out via SpikeProp.

    ∂L/∂w_jk = ∂L/∂t_k × ∂t_k/∂w_jk

    ∂t_k/∂w_jk = -ε(t_k - t_j - d_jk)
                 / (dVm_k/dt)|_{t=t_k}

    Parameters
    ----------
    delta_out  : list  — ∂L/∂t_k pour chaque neurone sortie
    t_hid      : list  — spike times couche cachée (s)
    t_out      : list  — spike times couche sortie (s)
    W_hid_out  : array — poids (n_hidden, n_phases)
    D_hid_out  : array — délais (n_hidden, n_phases)
    Vm_dot_out : list  — dVm/dt au moment du spike (V/s)
    tau_syn    : float — constante de temps PSP (s)

    Returns
    -------
    dW : np.ndarray — gradient (n_hidden, n_phases)
    """
    n_hidden, n_phases = W_hid_out.shape
    dW = np.zeros_like(W_hid_out)

    for k in range(n_phases):
        if t_out[k] is None or delta_out[k] == 0:
            continue
        if Vm_dot_out[k] == 0:
            continue

        for j in range(n_hidden):
            if t_hid[j] is None:
                continue

            t_arrive = t_hid[j] + D_hid_out[j, k]
            if t_out[k] < t_arrive:
                continue

            # PSP au moment du spike de sortie
            psp = np.exp(-(t_out[k] - t_arrive) / tau_syn)

            # ∂t_k/∂w_jk
            dt_dw = -psp / Vm_dot_out[k]

            # ∂L/∂w_jk = ∂L/∂t_k × ∂t_k/∂w_jk
            dW[j, k] += delta_out[k] * dt_dw

    return dW


def estimate_vm_dot(t_out, iex_fns_out, params, dt=1e-4):
    """
    Estime dVm/dt au moment du spike pour chaque
    neurone de sortie.

    Utilise une simulation courte autour de t_spike.

    Parameters
    ----------
    t_out     : list — spike times couche sortie
    iex_fns_out : list — fonctions Iex de sortie
    params    : dict — paramètres ML
    dt        : float — pas de temps

    Returns
    -------
    vm_dot : list — dVm/dt au spike (V/s)
    """
    from src.neuron.morris_lecar import ml_ode, nss

    vm_dot = []
    for k, t_k in enumerate(t_out):
        if t_k is None:
            vm_dot.append(0.0)
            continue

        # État du neurone juste avant le spike
        Vm_thresh = 0.0   # seuil de détection
        Vm_pre    = Vm_thresh - 1e-3  # juste avant

        n_pre = nss(Vm_pre, params['V3'], params['V4'])
        Iex_k = iex_fns_out[k](t_k)

        dVm, _ = ml_ode(t_k, [Vm_pre, n_pre],
                         Iex_k, params)
        vm_dot.append(dVm)

    return vm_dot