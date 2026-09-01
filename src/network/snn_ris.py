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
        # Initialisés uniformément dans [0, 1]
        self.W_enc_hid = np.random.uniform(
            0, 1, (n_features, n_hidden))
        self.W_hid_out = np.random.uniform(
            0, 1, (n_hidden, n_phases))

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

    def forward_hidden(self, t_enc):
        """
        Calcule les spike times de la couche cachée.

        Principe :
          Chaque neurone caché j reçoit des spikes
          de la couche encodage, retardés par D_enc_hid.
          
          Le spike le plus précoce et le plus fort
          (poids élevé) dépolarise le neurone j.
          
          On approxime la latence du neurone j par :
          t_hid_j = min_i(t_enc_i + D_enc_hid[i,j])
                    pondéré par W_enc_hid[i,j]

        Parameters
        ----------
        t_enc : list — spike times couche encodage (s)

        Returns
        -------
        t_hid : np.ndarray — spike times couche cachée (s)
                             None si pas de spike
        """
        t_hid = []

        for j in range(self.n_hidden):
            # Temps d'arrivée de chaque spike entrant
            # sur le neurone j
            contributions = []
            for i, t_i in enumerate(t_enc):
                if t_i is None:
                    continue
                # Spike i arrive à t_i + délai
                t_arrive = t_i + self.D_enc_hid[i, j]
                w        = self.W_enc_hid[i, j]
                contributions.append((t_arrive, w))

            if not contributions:
                t_hid.append(None)
                continue

            # Approximation de la latence du neurone j :
            # moyenne pondérée des temps d'arrivée
            # (spike fort + précoce → neurone j spike tôt)
            t_arr = np.array([c[0] for c in contributions])
            w_arr = np.array([c[1] for c in contributions])

            # Normaliser les poids
            w_sum = w_arr.sum()
            if w_sum < 1e-10:
                t_hid.append(None)
                continue

            # Latence approximée :
            # contributions précoces ET fortes
            # font spiker le neurone tôt
            t_j = np.sum(w_arr * t_arr) / w_sum
            t_hid.append(t_j)

        return t_hid

    def forward_output(self, t_hid):
        """
        Calcule les spike times de la couche de sortie.
        Même principe que forward_hidden.

        Parameters
        ----------
        t_hid : list — spike times couche cachée (s)

        Returns
        -------
        t_out : list — spike times couche sortie (s)
        """
        t_out = []

        for k in range(self.n_phases):
            contributions = []
            for j, t_j in enumerate(t_hid):
                if t_j is None:
                    continue
                t_arrive = t_j + self.D_hid_out[j, k]
                w        = self.W_hid_out[j, k]
                contributions.append((t_arrive, w))

            if not contributions:
                t_out.append(None)
                continue

            t_arr = np.array([c[0] for c in contributions])
            w_arr = np.array([c[1] for c in contributions])
            w_sum = w_arr.sum()

            if w_sum < 1e-10:
                t_out.append(None)
                continue

            t_k = np.sum(w_arr * t_arr) / w_sum
            t_out.append(t_k)

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