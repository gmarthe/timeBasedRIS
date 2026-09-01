"""
SNN pour configuration de RIS passive.
TTFS encoding + Morris-Lecar neurons + STDP learning.
"""

import numpy as np
import yaml
from src.neuron.morris_lecar import simulate, simulate_batch_parallel, time_to_first_spike
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

    def __init__(self, n_features=6, n_hidden=10,
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

        # Paramètres LIF pour les couches cachée et sortie
        # ML avec GCa=0, GK=0 → comportement LIF
        self.lif_params = params.copy()
        self.lif_params['Gna'] = 0.0
        self.lif_params['Gk']  = 0.0

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
        self.n_jobs = 7
        print('Table OK | St range : [%.1f, %.1f] ms' % (
            self.st_min*1e3, self.st_max*1e3))

    def encode(self, x_vec):
        """
        Encode un vecteur de features en spike times.
        Version vectorisée — 1 seul solve_ivp pour les 6 neurones.
        """
        from src.neuron.morris_lecar import (
            simulate_batch_parallel, time_to_first_spike_batch)

        # Calculer les Iex pour chaque feature
        iex_list = []
        for x in x_vec:
            Iex, _ = canal_to_iex_uniform_st(
                x, self.st_to_iex,
                self.st_min, self.st_max)
            # Iex constant → fonction du temps
            iex_list.append(lambda t, I=Iex: I)

        # T_sim = T_sim par défaut (encodage ML complet)
        T_sim = self.params.get('T_sim', 0.5)

        # 1 seul appel pour tous les neurones d'encodage
        t, Vms, ns = simulate_batch_parallel(
            iex_list, self.params, T_sim=T_sim,
            n_jobs=self.n_jobs)

        return time_to_first_spike_batch(Vms, t)

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
        if tau_syn is None:
            tau_syn = self.tau_syn

        from src.neuron.morris_lecar import (
            simulate_batch_parallel, time_to_first_spike_batch)

        # T_sim = max sur tous les neurones
        T_sim = max(
            self._compute_t_sim_adaptive(
                t_enc, self.D_enc_hid[:, j])
            for j in range(self.n_hidden))

        # Construire les fonctions Iex pour chaque neurone
        iex_fns = [
            self._compute_iex_psp(
                t_enc,
                self.W_enc_hid[:, j],
                self.D_enc_hid[:, j],
                tau_syn)
            for j in range(self.n_hidden)]

        # 1 seul appel solve_ivp pour toute la couche
        t, Vms, ns = simulate_batch_parallel(
            iex_fns, self.params, T_sim=T_sim,
            n_jobs=self.n_jobs)

        return time_to_first_spike_batch(Vms, t)


    def forward_output(self, t_hid, tau_syn=None):
        if tau_syn is None:
            tau_syn = self.tau_syn

        from src.neuron.morris_lecar import (
            simulate_batch_parallel, time_to_first_spike_batch)

        T_sim = max(
            self._compute_t_sim_adaptive(
                t_hid, self.D_hid_out[:, k])
            for k in range(self.n_phases))

        iex_fns = [
            self._compute_iex_psp(
                t_hid,
                self.W_hid_out[:, k],
                self.D_hid_out[:, k],
                tau_syn)
            for k in range(self.n_phases)]

        t, Vms, ns = simulate_batch_parallel(
            iex_fns, self.params, T_sim=T_sim,
            n_jobs=self.n_jobs)

        return time_to_first_spike_batch(Vms, t)

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

    def _compute_t_sim_adaptive(self, t_pre_list, delays,
                                margin_factor=5):
        """
        Calcule T_sim adaptatif.

        T_sim = max(t_pre + d) + margin_factor × tau_syn

        Parameters
        ----------
        t_pre_list    : list  — spike times pré-synaptiques
        delays        : array — délais synaptiques
        margin_factor : int   — facteur de marge

        Returns
        -------
        T_sim : float — durée de simulation adaptative (s)
        """
        t_arrives = []
        for i, t_pre in enumerate(t_pre_list):
            if t_pre is None:
                continue
            t_arrives.append(t_pre + delays[i])

        if not t_arrives:
            return self.tau_syn * margin_factor

        return max(t_arrives) + margin_factor * self.tau_syn