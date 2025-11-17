import numpy as np
from scipy.special import gamma  # For toy χ approx

# S1: Toy X - graph of loop states (5 nodes, edges as extensions)
nodes = 5  # Partial loops
edges = np.array([[0,1], [1,2], [2,3], [3,4], [4,0], [0,2], [1,3]])  # Toy arrows y→x
tau = np.log([2,3,5,7,11,13,17])[edges[:,1] - edges[:,0]]  # Toy log p weights (mimic primitives)

# S2: τ additive (sum along paths)

# S3: μ approx - equilibrium eigenvector at s=1/2
def transfer_matrix(s):
    mat = np.zeros((nodes, nodes))
    for i, (y, x) in enumerate(edges):
        mat[x, y] = np.exp(-s * tau[i])  # L_s entry
    return mat

L_half = transfer_matrix(0.5)
eigvals, eigvecs = np.linalg.eig(L_half.T)  # Perron approx
mu = np.abs(eigvecs[:, np.argmax(np.abs(eigvals))])  # Positive eigenvector

# S4: χ toy approx - Γ(s/2) π^{-s/2} etc.
def chi(s):
    return np.pi**(-s/2) * gamma(s/2 + 1)  # Simplified archimedean

# ~L_s conjugated
def conjugated_L(s, it=0):
    L_s = transfer_matrix(s + 1j*it)
    chi_s = chi(s + 1j*it)
    chi_1ms = chi(1 - s + 1j*it)
    conj = np.sqrt(chi_1ms / chi_s) * L_s  # ~L_s approx (matrix mult for toy)
    return conj

# U1 Test: Singular vals at σ=1/2 vs off
svd_half = np.linalg.svd(conjugated_L(0.5), compute_uv=False)[0]  # Top singular at 1/2
svd_off = np.linalg.svd(conjugated_L(0.6), compute_uv=False)[0]  # Off-line deviate

print(f"Top singular at 1/2: {svd_half:.2f} (should ≈1)")
print(f"Top singular off (0.6): {svd_off:.2f} (should ≠1)")

# Expected: If ≈1 at 1/2 and drifts off, lens holds—test RH neutrality proxy.