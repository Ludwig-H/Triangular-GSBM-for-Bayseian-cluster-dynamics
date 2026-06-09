import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

d = 2
p = 0.8
n = 100
L = int(np.round(n ** (1/d)))
T = 10
N_samples = 1
device = torch.device('cpu')

def generate_graph(L, device):
    N = L * L
    edges = []
    # 0 to N-1: Horizontal
    for y in range(L):
        for x in range(L):
            edges.append([y*L + x, y*L + (x+1)%L])
    # N to 2N-1: Vertical
    for y in range(L):
        for x in range(L):
            edges.append([y*L + x, ((y+1)%L)*L + x])
    # 2N to 3N-1: Diagonal
    for y in range(L):
        for x in range(L):
            edges.append([y*L + x, ((y+1)%L)*L + (x+1)%L])
            
    edges = torch.tensor(edges, dtype=torch.long, device=device)
    
    white_triangles = []
    black_triangles = []
    for y in range(L):
        for x in range(L):
            idx = y*L + x
            
            # White triangle
            h_edge = idx
            v_edge = N + y*L + (x+1)%L
            d_edge = 2*N + idx
            white_triangles.append([h_edge, v_edge, d_edge])
            
            # Black triangle
            v_edge_b = N + idx
            h_edge_b = ((y+1)%L)*L + x
            d_edge_b = 2*N + idx
            black_triangles.append([v_edge_b, h_edge_b, d_edge_b])
            
    return edges, torch.tensor(white_triangles, dtype=torch.long, device=device), torch.tensor(black_triangles, dtype=torch.long, device=device)

def SW_step(sigma, W, edges, mode, up, triangles_A, triangles_B=None):
    N = len(sigma)
    frozen = torch.zeros(len(edges), dtype=torch.bool, device=sigma.device)
    satisfied = (W * sigma[edges[:,0]] * sigma[edges[:,1]]) > 0
    
    if mode == 'edges':
        freeze_prob = 1 - torch.exp(torch.tensor(-up, device=sigma.device))
        frozen = satisfied & (torch.rand(len(edges), device=sigma.device) < freeze_prob)
    else:
        if mode == 'white':
            triangles = triangles_A
            au = 1 - torch.exp(torch.tensor(-2 * up, device=sigma.device))
        else: # 'half-half'
            triangles = torch.cat((triangles_A, triangles_B), dim=0)
            au = 1 - torch.exp(torch.tensor(-up, device=sigma.device))
            
        W_tri = W[triangles]
        type_A = torch.prod(W_tri, dim=1) > 0
        sat_tri = satisfied[triangles]
        sat_count = torch.sum(sat_tri, dim=1)
        
        rand_vals = torch.rand(len(triangles), device=sigma.device)
        freeze_A = type_A & (sat_count == 3) & (rand_vals < au)
        freeze_B = (~type_A) & (sat_count == 2) & (rand_vals < au)
        
        if freeze_A.any():
            frozen[triangles[freeze_A].flatten()] = True
            
        B_idx = torch.where(freeze_B)[0]
        if len(B_idx) > 0:
            sat_B = sat_tri[B_idx]
            rand_pick = torch.randint(0, 2, (len(B_idx),), device=sigma.device)
            sat_indices = torch.nonzero(sat_B)
            col_indices = sat_indices[:, 1].view(len(B_idx), 2)
            chosen_col = col_indices[torch.arange(len(B_idx)), rand_pick]
            chosen_edges = triangles[B_idx, chosen_col]
            frozen[chosen_edges] = True

    # Connected components using scipy (CPU)
    frozen_cpu = frozen.cpu().numpy()
    edges_cpu = edges.cpu().numpy()
    
    row = edges_cpu[frozen_cpu, 0]
    col = edges_cpu[frozen_cpu, 1]
    data = np.ones(len(row), dtype=bool)
    
    adj = csr_matrix((data, (row, col)), shape=(N, N))
    n_comp, labels = connected_components(adj, directed=False)
    
    lcc_frac = np.max(np.bincount(labels)) / N
    
    flip = torch.randint(0, 2, (n_comp,), device=sigma.device) * 2 - 1
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=sigma.device)
    sigma = sigma * flip[labels_tensor]
    
    return sigma, lcc_frac

edges, white_tri, black_tri = generate_graph(L, device)
up = np.log(p / (1 - p))
modes = ['edges', 'white', 'half-half']

for mode in modes:
    Sigma = torch.randint(0, 2, (L*L,), device=device) * 2 - 1
    same_comm = Sigma[edges[:,0]] == Sigma[edges[:,1]]
    correct_obs = torch.rand(len(edges), device=device) < p
    W = torch.where(same_comm == correct_obs, torch.tensor(up, device=device), torch.tensor(-up, device=device))
    
    sigma = torch.randint(0, 2, (L*L,), device=device) * 2 - 1
    for t in range(T):
        sigma, lcc = SW_step(sigma, W, edges, mode, up, white_tri, black_tri)
        
print("Success!")
