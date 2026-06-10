#!/usr/bin/env python
# coding: utf-8

# # Comparaison des dynamiques Swendsen-Wang (Optimisée CPU / GPU)
# 
# Ce script implémente et compare trois dynamiques de clusters pour la détection de communautés sur un graphe signé (modèle de Sankararaman-Baccelli) comme décrit dans le Chapitre 11 de la thèse.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Paramètres
d = 2
p = 0.85
n = 10000
L = int(np.round(n ** (1/d)))
T = 1000
N_samples = 10

# Tentative d'import de CuPy pour la détection du GPU
try:
    import cupy as cp
    has_gpu = True
except ImportError:
    has_gpu = False

# ==========================================
# SÉLECTION AUTOMATIQUE DU MATÉRIEL (GPU / CPU)
# ==========================================
# Si un GPU est disponible, on l'utilise par défaut (optimisé même pour petits n grâce aux noyaux CUDA !)
use_gpu = has_gpu

# Décommentez la ligne ci-dessous pour forcer manuellement le matériel si besoin :
# use_gpu = True # ou False

if use_gpu:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix
    
    # Noyaux CUDA personnalisés pour éviter la construction de csr_matrix et les syncs CPU-GPU
    update_labels_kernel = cp.ElementwiseKernel(
        'int32 u, int32 v, bool frozen',
        'raw int32 labels',
        '''
        if (frozen) {
            int root_u = labels[u];
            while (root_u != labels[root_u]) {
                root_u = labels[root_u];
            }
            int root_v = labels[v];
            while (root_v != labels[root_v]) {
                root_v = labels[root_v];
            }
            if (root_u != root_v) {
                if (root_u < root_v) {
                    atomicMin(&labels[root_v], root_u);
                } else {
                    atomicMin(&labels[root_u], root_v);
                }
            }
        }
        ''',
        'update_labels_kernel'
    )

    flatten_kernel = cp.ElementwiseKernel(
        'int32 dummy',
        'raw int32 labels',
        '''
        int p = labels[i];
        while (p != labels[p]) {
            p = labels[p];
        }
        labels[i] = p;
        ''',
        'flatten_kernel'
    )
    
    xp = cp
    print(f"Exécution automatique configurée sur : GPU (CuPy) [n = {n}]")
else:
    import scipy.sparse
    import scipy.sparse.csgraph
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    xp = np
    print(f"Exécution automatique configurée sur : CPU (NumPy/SciPy) [n = {n}]")


def generate_graph(L):
    N = L * L
    # Génération de grille entièrement vectorisée (aucun python loop !)
    y, x = xp.meshgrid(xp.arange(L), xp.arange(L), indexing='ij')
    idx = y * L + x
    
    h_target = y * L + (x + 1) % L
    h_edges = xp.stack((idx, h_target), axis=-1).reshape(-1, 2)
    
    v_target = ((y + 1) % L) * L + x
    v_edges = xp.stack((idx, v_target), axis=-1).reshape(-1, 2)
    
    d_target = ((y + 1) % L) * L + (x + 1) % L
    d_edges = xp.stack((idx, d_target), axis=-1).reshape(-1, 2)
    
    edges = xp.concatenate((h_edges, v_edges, d_edges), axis=0)
    
    v_edge_white = N + y * L + (x + 1) % L
    d_edge_white = 2 * N + idx
    white_triangles = xp.stack((idx, v_edge_white, d_edge_white), axis=-1).reshape(-1, 3)
    
    v_edge_black = N + idx
    h_edge_black = ((y + 1) % L) * L + x
    d_edge_black = 2 * N + idx
    black_triangles = xp.stack((v_edge_black, h_edge_black, d_edge_black), axis=-1).reshape(-1, 3)
    
    return edges, white_triangles, black_triangles


def SW_step(sigma, W, edges, mode, up, triangles, type_A, au, freeze_prob):
    N_edges = len(edges)
    satisfied = (W * sigma[edges[:,0]] * sigma[edges[:,1]]) > 0
    
    if mode == 'edges':
        frozen = satisfied & (xp.random.rand(N_edges) < freeze_prob)
    else:
        sat_tri = satisfied[triangles]
        sat_count = xp.sum(sat_tri, axis=1)
        
        rand_vals = xp.random.rand(len(triangles))
        freeze_A = type_A & (sat_count == 3) & (rand_vals < au)
        freeze_B = (~type_A) & (sat_count == 2) & (rand_vals < au)
        
        frozen_int = xp.zeros(N_edges, dtype=xp.int32)
        
        # Exécution inconditionnelle de add.at pour éviter la synchronisation GPU-CPU
        xp.add.at(frozen_int, triangles.flatten(), xp.repeat(freeze_A, 3).astype(xp.int32))
            
        # Freeze B: Choix vectorisé sans synchronisation CPU-GPU ni indexation booléenne
        S0 = sat_tri[:, 0]
        S2 = sat_tri[:, 2]
        C1 = 1 - S0.astype(xp.int32)
        C2 = 1 + S2.astype(xp.int32)
        
        pick = xp.random.randint(0, 2, size=len(triangles))
        chosen_col = xp.where(pick == 0, C1, C2)
        chosen_edges = triangles[xp.arange(len(triangles)), chosen_col]
        
        xp.add.at(frozen_int, chosen_edges, freeze_B.astype(xp.int32))
        frozen = frozen_int > 0

    N_nodes = len(sigma)
    if use_gpu:
        labels = xp.arange(N_nodes, dtype=xp.int32)
        dummy = xp.arange(N_nodes, dtype=xp.int32)
        u = edges[:, 0]
        v = edges[:, 1]
        
        # 10 iterations is sufficient for grid graphs up to L=1000
        for _ in range(10):
            update_labels_kernel(u, v, frozen, labels)
            flatten_kernel(dummy, labels)
            
        lcc_frac = xp.max(xp.bincount(labels)) / N_nodes
    else:
        frozen_edges = edges[frozen]
        row = frozen_edges[:, 0]
        col = frozen_edges[:, 1]
        data = xp.ones_like(row, dtype=xp.float32)
        
        adj = csr_matrix((data, (row, col)), shape=(N_nodes, N_nodes))
        n_comp, labels = connected_components(adj, directed=False)
        
        lcc_frac = xp.max(xp.bincount(labels)) / N_nodes
    
    # Flip optimisé sans transfert de n_comp vers le CPU
    flip = xp.random.randint(0, 2, size=N_nodes) * 2 - 1
    sigma = sigma * flip[labels]
    
    return sigma, lcc_frac


if __name__ == '__main__':
    edges_gpu, white_tri_gpu, black_tri_gpu = generate_graph(L)
    up = float(np.log(p / (1 - p)))
    freeze_prob = 1.0 - np.exp(-up)
    au_white = float(1.0 - np.exp(-2.0 * up))
    au_half = float(1.0 - np.exp(-up))
    modes = ['edges', 'white', 'half-half']

    overlap_history = {m: xp.zeros((N_samples, T)) for m in modes}
    lcc_history = {m: xp.zeros((N_samples, T)) for m in modes}

    for sample in tqdm(range(N_samples), desc="Samples"):
        Sigma = xp.random.randint(0, 2, size=L*L) * 2 - 1
        
        same_comm = Sigma[edges_gpu[:,0]] == Sigma[edges_gpu[:,1]]
        correct_obs = xp.random.rand(len(edges_gpu)) < p
        W = xp.where(same_comm == correct_obs, up, -up)
        
        for mode in modes:
            sigma = xp.random.randint(0, 2, size=L*L) * 2 - 1
            
            # Précalculs statiques hors de la boucle temporelle (évite les allocations GPU répétées !)
            if mode == 'edges':
                triangles = None
                type_A = None
                au = 0.0
            elif mode == 'white':
                triangles = white_tri_gpu
                type_A = xp.prod(W[triangles], axis=1) > 0
                au = au_white
            else:
                triangles = xp.vstack((white_tri_gpu, black_tri_gpu))
                type_A = xp.prod(W[triangles], axis=1) > 0
                au = au_half
                
            ov_list = []
            lcc_list = []
            for t in range(T):
                sigma, lcc = SW_step(sigma, W, edges_gpu, mode, up, triangles, type_A, au, freeze_prob)
                ov = xp.abs(xp.mean(sigma * Sigma))
                ov_list.append(ov)
                lcc_list.append(lcc)
                
            # Affectation unique par ligne (élimine 2000 écritures par élément sur GPU)
            overlap_history[mode][sample] = xp.stack(ov_list)
            lcc_history[mode][sample] = xp.stack(lcc_list)

    mean_ov_gpu = {m: xp.mean(overlap_history[m], axis=0).get() if use_gpu and has_gpu else xp.mean(overlap_history[m], axis=0) for m in modes}
    std_ov_gpu = {m: xp.std(overlap_history[m], axis=0).get() if use_gpu and has_gpu else xp.std(overlap_history[m], axis=0) for m in modes}
    mean_lcc_gpu = {m: xp.mean(lcc_history[m], axis=0).get() if use_gpu and has_gpu else xp.mean(lcc_history[m], axis=0) for m in modes}
    std_lcc_gpu = {m: xp.std(lcc_history[m], axis=0).get() if use_gpu and has_gpu else xp.std(lcc_history[m], axis=0) for m in modes}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = {'edges': 'blue', 'white': 'orange', 'half-half': 'green'}
    labels = {'edges': 'SW-Edges', 'white': 'SW-Triangles (Blancs)', 'half-half': 'SW-Triangles (Half-Half)'}

    for mode in modes:
        ax1.plot(mean_ov_gpu[mode], label=labels[mode], color=colors[mode])
        ax1.fill_between(range(T), mean_ov_gpu[mode] - std_ov_gpu[mode], mean_ov_gpu[mode] + std_ov_gpu[mode], color=colors[mode], alpha=0.2)
        
        ax2.plot(mean_lcc_gpu[mode], label=labels[mode], color=colors[mode])
        ax2.fill_between(range(T), mean_lcc_gpu[mode] - std_lcc_gpu[mode], mean_lcc_gpu[mode] + std_lcc_gpu[mode], color=colors[mode], alpha=0.2)

    ax1.set_title('Overlap (Recouvrement) vs Itérations')
    ax1.set_xlabel('Itération')
    ax1.set_ylabel('Overlap moyen')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Taille de la plus grande composante (LCC) vs Itérations')
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Proportion de points dans la LCC')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
