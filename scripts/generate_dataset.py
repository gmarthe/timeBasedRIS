"""
Génère un dataset K=1 pour l'entraînement SpikeProp.
"""

import numpy as np
import yaml
import os

def get_niveaux(n_bits):
    return np.linspace(0, 2*np.pi, 2**n_bits, endpoint=False)

def quantifier_phase(phase, niveaux):
    return np.argmin(np.abs(np.angle(np.exp(1j*(phase-niveaux)))))

def generate_channels_k1(N, K_rician=3):
    def rician_ula(N):
        theta   = np.random.uniform(-np.pi/2, np.pi/2)
        a       = np.exp(1j*np.pi*np.arange(N)*np.sin(theta))
        scatter = (np.random.randn(N)+1j*np.random.randn(N))/np.sqrt(2)
        return (np.sqrt(K_rician/(K_rician+1))*a
              + np.sqrt(1/(K_rician+1))*scatter)
    h1 = rician_ula(N)
    h2 = rician_ula(N)
    hd = (np.random.randn()+1j*np.random.randn())/np.sqrt(2)*0.1
    return h1, h2, hd

def generate_dataset(n_samples, N=32, n_bits=2,
                     K_rician=3, seed=42):
    np.random.seed(seed)
    niveaux = get_niveaux(n_bits)

    # Features : [Re(h1), Im(h1), Re(h2), Im(h2), Re(hd), Im(hd)]
    X = np.zeros((n_samples, N, 6))
    # Labels : phase optimale par élément
    Y = np.zeros((n_samples, N), dtype=int)

    for i in range(n_samples):
        if i % 100 == 0:
            print('  %d/%d' % (i, n_samples))
        h1, h2, hd = generate_channels_k1(N, K_rician)
        phases  = np.angle(h2) - np.angle(h1)
        phi_idx = np.array([quantifier_phase(phases[n], niveaux)
                             for n in range(N)])
        for n in range(N):
            X[i,n] = [h1[n].real, h1[n].imag,
                      h2[n].real, h2[n].imag,
                      hd.real,    hd.imag]
        Y[i] = phi_idx

    return X, Y


if __name__ == '__main__':
    cfg      = yaml.safe_load(open('config/params.yaml'))
    N        = cfg['ris']['N']
    n_bits   = cfg['ris']['n_bits']
    K_rician = cfg['channel']['K_rician']

    os.makedirs('results', exist_ok=True)

    # Dataset d'entraînement
    print('Génération dataset train (500 samples)...')
    X_tr, Y_tr = generate_dataset(
        500, N, n_bits, K_rician, seed=42)
    np.savez('results/dataset_train.npz', X=X_tr, Y=Y_tr)
    print('Sauvegardé : results/dataset_train.npz')

    # Dataset de test
    print('Génération dataset test (100 samples)...')
    X_te, Y_te = generate_dataset(
        100, N, n_bits, K_rician, seed=123)
    np.savez('results/dataset_test.npz', X=X_te, Y=Y_te)
    print('Sauvegardé : results/dataset_test.npz')

    print('X_train:', X_tr.shape)
    print('Y_train:', Y_tr.shape)
    print('Classes :', np.unique(Y_tr))