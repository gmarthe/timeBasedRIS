"""
SNN pour configuration de RIS passive.
TTFS encoding + Morris-Lecar neurons + STDP learning.
"""

import numpy as np
import yaml
from src.neuron.morris_lecar import simulate, time_to_first_spike
from src.neuron.encoding import (
    build_iex_to_st_table,
    build_st_to_iex_interpolator,
    canal_to_iex_uniform_st,
    iex_to_ttfs,
    decode_wta
)


class SNN_RIS:
    """
    Réseau SNN pour configuration RIS.

    Architecture :
      Couche encodage : 6 neurones ML (un par feature)
      Couche cachée   : N_hidden neurones ML
      Couche sortie   : 4 neurones ML (WTA, un par phase)

    Apprentissage : STDP sur les poids
    Délais        : fixes, initialisés aléatoirement
    """

    def __init__(self, n_features=6, n_hidden=60,
                 n_phases=4, params=None,
                 encoding_params=None, seed=42):
        """
        Parameters
        ----------
        n_features      : int  — features par élément RIS
        n_hidden        : int  — neurones couche cachée
        n_phases        : int  — phases possibles (2**n_bits)
        params          : dict — paramètres neurone ML
        encoding_params : dict — paramètres encodage
        seed            : int  — graine aléatoire
        """
        np.random.seed(seed)

        self.n_features      = n_features
        self.n_hidden        = n_hidden
        self.n_phases        = n_phases
        self.params          = params
        self.encoding_params = encoding_params

        # ── Poids synaptiques ─────────────────────────────────
        iex_target = 150e-6
        self.W_enc_hid = np.random.uniform(
            0.5 * iex_target / n_features,   # min
            1.5 * iex_target / n_features,   # max
            (n_features, n_hidden))

        self.W_hid_out = np.random.uniform(
            0.5 * iex_target / n_hidden,
            1.5 * iex_target / n_hidden,
            (n_hidden, n_phases))

        # ── Délais synaptiques (fixes) ────────────────────────
        # Initialisés aléatoirement dans [0, d_max]
        d_max = 10e-3   # 10ms max
        self.D_enc_hid = np.random.uniform(
            0, d_max, (n_features, n_hidden))
        self.D_hid_out = np.random.uniform(
            0, d_max, (n_hidden, n_phases))

        # ── Table d'encodage TTFS ─────────────────────────────
        # Construite une seule fois à l'initialisation
        print('Construction table TTFS...')
        iex_min = encoding_params['Iex_min']
        iex_max = encoding_params['Iex_max']
        iex_arr, st_arr = build_iex_to_st_table(
            simulate, params,
            iex_min=iex_min, iex_max=iex_max,
            n_points=50)
        self.st_min     = st_arr.min()
        self.st_max     = st_arr.max()
        self.st_to_iex  = build_st_to_iex_interpolator(
            iex_arr, st_arr)
        self.tau_syn = encoding_params.get('tau_syn', 50e-3)
        print('Table OK | St range : [%.1f, %.1f] ms' % (
            self.st_min*1e3, self.st_max*1e3))

    def encode(self, x_vec):
        """
        Encode un vecteur de features en spike times.

        Parameters
        ----------
        x_vec : np.ndarray — features (n_features,)

        Returns
        -------
        t_enc : np.ndarray — spike times couche encodage (s)
                             None si pas de spike
        """
        t_enc = []
        for x in x_vec:
            Iex, _ = canal_to_iex_uniform_st(
                x, self.st_to_iex,
                self.st_min, self.st_max)
            St = iex_to_ttfs(Iex, simulate, self.params)
            t_enc.append(St)
        return t_enc

    def _psp(self, t, t_pre, d, tau_syn=None):
        """
        Post-Synaptic Potential exponentiel.

        ε(t - t_pre - d) = exp(-(t - t_pre - d) / τ_syn)
                           si t > t_pre + d, sinon 0

        Parameters
        ----------
        t       : float — temps courant (s)
        t_pre   : float — spike time pré-synaptique (s)
        d       : float — délai synaptique (s)
        tau_syn : float — constante de temps (s)

        Returns
        -------
        psp : float — valeur du PSP
        """
        t_arrive = t_pre + d
        if t < t_arrive:
            return 0.0
        return np.exp(-(t - t_arrive) / tau_syn)

    def _compute_iex_psp(self, t_pre_list, weights,
                      delays, tau_syn=None):
        if tau_syn is None:
            tau_syn = self.tau_syn

        #Capturer tau_syn dans la closure explicitement
        _tau = tau_syn

        def iex_fn(t):
            total = 0.0
            for i, t_pre in enumerate(t_pre_list):
                if t_pre is None:
                    continue
                total += weights[i] * self._psp(
                    t, t_pre, delays[i], _tau)  # ← _tau pas tau_syn
            return total

        return iex_fn

    def forward_hidden(self, t_enc, tau_syn=None):
        """
        Calcule les spike times de la couche cachée
        en simulant vraiment les neurones ML avec PSP.

        Parameters
        ----------
        t_enc   : list  — spike times couche encodage (s)
        tau_syn : float — constante de temps PSP (s)

        Returns
        -------
        t_hid : list — spike times couche cachée (s)
                       None si pas de spike
        """
        t_hid = []

        for j in range(self.n_hidden):
            # Fonction Iex(t) pour le neurone j
            iex_fn = self._compute_iex_psp(
                t_enc,
                self.W_enc_hid[:, j],
                self.D_enc_hid[:, j],
                tau_syn)

            # Simuler le neurone ML avec ce Iex(t)
            t, Vm, n = simulate(
                Iex=iex_fn,
                params=self.params)

            St = time_to_first_spike(Vm, t)
            t_hid.append(St)

        return t_hid

    def forward_output(self, t_hid, tau_syn=None):
        """
        Calcule les spike times de la couche de sortie
        en simulant vraiment les neurones ML avec PSP.

        Parameters
        ----------
        t_hid   : list  — spike times couche cachée (s)
        tau_syn : float — constante de temps PSP (s)

        Returns
        -------
        t_out : list — spike times couche sortie (s)
        """
        t_out = []

        for k in range(self.n_phases):
            iex_fn = self._compute_iex_psp(
                t_hid,
                self.W_hid_out[:, k],
                self.D_hid_out[:, k],
                tau_syn)

            t, Vm, n = simulate(
                Iex=iex_fn,
                params=self.params)

            St = time_to_first_spike(Vm, t)
            t_out.append(St)

        return t_out

    def predict(self, x_vec):
        """
        Pipeline complet : features → phase RIS.

        Parameters
        ----------
        x_vec : np.ndarray — features (n_features,)

        Returns
        -------
        phase_idx : int   — indice de phase (0-3)
        t_enc     : list  — spike times encodage
        t_hid     : list  — spike times cachée
        t_out     : list  — spike times sortie
        """
        t_enc     = self.encode(x_vec)
        t_hid     = self.forward_hidden(t_enc)
        t_out     = self.forward_output(t_hid)
        phase_idx = decode_wta(t_out)

        return phase_idx, t_enc, t_hid, t_out